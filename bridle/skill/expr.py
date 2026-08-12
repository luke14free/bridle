"""bridle.skill.expr — the tier-2 reward expression language.

A skill's reward is normally assembled from the fixed 9-term vocabulary (`vocab.py`). A row that
vocabulary cannot express falls back to a raw expression string — `expr: "2 * (1 - tanh(6 * abs(h
- 0.015)))"` — authored by a local 27-30B LLM, not a human. That authorship is exactly why this
module exists instead of just calling `eval()`:

  SAFETY. The string is parsed against an `ast` node whitelist and NEVER passed to `eval`/`exec`.
  `ast.Attribute` and `ast.Subscript` are refused outright (not merely "not in the grammar we
  generate") — that is what makes `().__class__.__bases__[0].__subclasses__()` a parse-time refusal
  instead of a possibility that depends on the model never trying it. See the adversarial checks
  driven from `bridle/tests/test_expr.py` and `docs/` for the attacks this closes.

  BATCH SEMANTICS. The exact same parsed `Expr` is evaluated once with plain Python floats (unit
  tests, `Expr.names` compile-time checks) and later with batched CUDA tensors — 4096 parallel
  envs — during PPO training. The two consequences that follow, and that a naive port from Python
  gets wrong:
    - `where(c, a, b)` is written branch-free as `c * a + (1 - c) * b`. A Python `if c: a else: b`
      would call `bool()` on the whole batch and silently take ONE branch for all 4096 envs. The
      condition is made NUMERIC first (`_numeric`), because a torch comparison yields a bool tensor
      and `1 - bool_tensor` raises — without that step the claim in this paragraph was false for
      `clamp`/`min`/`max` on a raw tensor (measured 2026-08-13).
    - `tanh`/`exp`/`log`/`sqrt` dispatch to the value's own method first (`x.tanh()` — torch tensors
      have these) and fall back to `math.*` only when that method is absent (plain floats/ints).
      That is the one piece of code that makes "the same expression string means the same thing for
      a CPU float and a CUDA tensor" true, rather than merely intended.
"""
import ast
import math
import operator

__all__ = ["ExprError", "ALLOWED_CALLS", "parse", "Expr"]


class ExprError(Exception):
    """Raised for anything wrong with an expression string: parse-time whitelist violations,
    syntax errors, and evaluation-time failures (undefined name, div-by-zero, ...). Callers — the
    skill compiler, the training loop — only ever need to catch this one type; a raw `SyntaxError`
    or `KeyError` escaping from a sandboxed evaluator would be a bug in this module, not a legitimate
    outcome.
    """


# The only functions an expression may call. Deliberately small: every entry has both a Python-float
# and a torch-tensor-shaped meaning (see module docstring). Extending this set is a design decision,
# not a bug fix — anything else belongs in the fixed 9-term vocabulary or a new primitive.
ALLOWED_CALLS = frozenset({"abs", "tanh", "exp", "log", "sqrt", "clamp", "min", "max", "where"})

# ast node types a reward expression may contain. Everything not listed here is refused at parse
# time, before an Expr object is ever constructed — in particular ast.Attribute and ast.Subscript
# are absent ON PURPOSE, which is what turns dunder-chain escapes (`x.__class__`, `x[0]`) into a
# parse-time ExprError rather than something that merely doesn't appear in the grammar we intended.
_ALLOWED_NODE_TYPES = frozenset({
    ast.Expression,
    ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Constant,
    ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
})

# Human-readable labels for common rejected constructs, so the error names what was actually typed
# instead of a bare Python class name like "ListComp". Anything not in this table falls back to
# type(node).__name__, which is still informative (e.g. "Global", "Yield").
_NODE_LABELS = {
    ast.Attribute: "attribute access (e.g. x.attr)",
    ast.Subscript: "subscript (e.g. x[0])",
    ast.ListComp: "list comprehension",
    ast.SetComp: "set comprehension",
    ast.DictComp: "dict comprehension",
    ast.GeneratorExp: "generator expression",
    ast.Lambda: "lambda",
    ast.IfExp: "conditional expression (x if y else z)",
    ast.JoinedStr: "f-string",
    ast.FormattedValue: "f-string interpolation",
    ast.Dict: "dict literal",
    ast.Set: "set literal",
    ast.List: "list literal",
    ast.Tuple: "tuple literal",
    ast.Starred: "starred expression (*args)",
    ast.keyword: "keyword argument",
    ast.BoolOp: "boolean 'and'/'or' (use multiplication/where instead — see module docstring)",
}


def _allowed_calls_clause() -> str:
    return f"allowed calls are: {', '.join(sorted(ALLOWED_CALLS))}"


def _reject(node: ast.AST) -> None:
    label = _NODE_LABELS.get(type(node), type(node).__name__)
    raise ExprError(
        f"'{label}' is not allowed in a reward expression; "
        f"allowed constructs are arithmetic (+ - * / **), comparisons, and calls to "
        f"one of the whitelisted functions ({_allowed_calls_clause()})"
    )


def _validate_call(node: ast.Call) -> None:
    if not isinstance(node.func, ast.Name):
        # e.g. `__import__('os').system(...)` or `obj.method()` — the callee is an Attribute/
        # Subscript/other expression, not a bare name. Refuse before even looking at args.
        raise ExprError(
            "a call's target must be a plain function name, not an attribute or expression "
            f"({_allowed_calls_clause()})"
        )
    if node.func.id not in ALLOWED_CALLS:
        raise ExprError(f"'{node.func.id}' is not a whitelisted call ({_allowed_calls_clause()})")


def _validate_constant(node: ast.Constant) -> None:
    # bool is a subclass of int in Python, so `type(x) in (int, float)` (not isinstance) is what
    # excludes True/False/None/strings/bytes while still admitting plain numeric literals.
    if type(node.value) not in (int, float):
        raise ExprError(
            f"only numeric constants are allowed in a reward expression, got {node.value!r}"
        )


# `2 ** 300000` parses instantly but is a ~100k-digit int the moment it's evaluated — real time and
# memory spent before the value ever reaches a tensor, usually because a model typo'd an extra digit
# into an exponent. 64 is generous for any reward-shaping exponent that has ever appeared in practice
# (even `x ** 10` is already an unusually sharp shaping term) while still being nowhere near a value
# that could occur by legitimate accident.
_MAX_LITERAL_EXPONENT = 64


def _validate_pow(node: ast.BinOp) -> None:
    # Only the fully-literal case (`2 ** 300000`, both base AND exponent numeric constants) is
    # guarded here, and only by inspecting the constant's already-parsed value — this never computes
    # the power itself, so the guard itself can't be the expensive part. A variable exponent
    # (`x ** y`, or even `x ** 300000` with a non-constant base) is a legitimate runtime value whose
    # magnitude depends on what `x`/`y` are bound to at evaluate() time — that is the caller's
    # concern (e.g. clamping inputs), not something this parser can or should judge.
    left, right = node.left, node.right
    if not (isinstance(left, ast.Constant) and type(left.value) in (int, float)):
        return
    if not (isinstance(right, ast.Constant) and type(right.value) in (int, float)):
        return
    if abs(right.value) > _MAX_LITERAL_EXPONENT:
        raise ExprError(
            f"literal exponent {right.value!r} exceeds the maximum allowed literal exponent "
            f"({_MAX_LITERAL_EXPONENT}) in a reward expression; a value this large is almost always "
            f"a typo (e.g. an extra digit) rather than an intended reward shaping term"
        )


def _validate(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        node_type = type(node)
        if node_type not in _ALLOWED_NODE_TYPES:
            _reject(node)
        elif node_type is ast.Call:
            _validate_call(node)
        elif node_type is ast.Constant:
            _validate_constant(node)
        elif node_type is ast.BinOp and isinstance(node.op, ast.Pow):
            _validate_pow(node)


def parse(src: str) -> "Expr":
    """Parse a reward-expression string into an `Expr`, or raise `ExprError` naming the offending
    construct. Never calls `eval`/`exec`/`compile` in exec/eval-execution mode on the source — only
    `ast.parse(..., mode="eval")` to get a tree, which is then walked and validated, never executed
    by the Python interpreter.
    """
    if not isinstance(src, str):
        raise ExprError(f"expression source must be a string, got {type(src).__name__}")
    if not src.strip():
        raise ExprError("expression source is empty")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"syntax error in expression {src!r}: {exc}") from exc

    _validate(tree)

    # Free variables = every Name node NOT in CALL POSITION (i.e. not the `func` of an ast.Call).
    # This must be a positional check, not a filter on `node.id in ALLOWED_CALLS` — the latter would
    # drop a Name from `.names` just because it happens to share a spelling with a builtin, even when
    # used as a plain value (`min - 1` uses `min` as a variable, not a call; filtering by id would
    # make that free variable invisible to the compiler's "every name resolves to a declared measure"
    # check, and it would then blow up at evaluate() time instead — the exact failure mode `.names`
    # exists to prevent). So: collect the *node identities* (via id()) of every Name sitting in
    # call-func position, then exclude exactly those specific AST nodes, not every Name of that id.
    #
    # DECISION: a measure MAY be named `min`/`max`/`abs`/etc. Call vs. value position is unambiguous
    # in this grammar (`min(...)` is always a Call.func, a bare `min` is always a value) — there is no
    # real ambiguity for an author to trip over, so refusing the name at parse time would only add
    # friction without closing any actual hole.
    call_func_ids = {
        id(node.func) for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    names = frozenset(
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and id(node) not in call_func_ids
    )
    return Expr(tree, src, names)


# ── evaluation ────────────────────────────────────────────────────────────────────────────────
# Every helper below is written using only +, -, *, /, **, comparisons, and generic dispatch — never
# a Python `if`/`and`/`or` on the *value* being evaluated — because those short-circuit to a single
# Python bool and would silently pick one branch for an entire batched tensor. See module docstring.

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_COMPARE_OPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}


def _method_or_math(method_name, math_fn):
    """Build a unary call (tanh/exp/log/sqrt) that dispatches to the operand's own method first,
    falling back to `math.<fn>` when that method doesn't exist. Plain floats/ints have no `.tanh()`
    etc., so they always fall through to `math`; a torch tensor has `.tanh()` etc., so it takes that
    path and the operation stays batched/on-device instead of being forced through a Python float.
    This is the one piece of code that makes one expression string mean the same thing for a CPU
    float in a unit test and a CUDA tensor of 4096 envs in training.
    """
    def call(x):
        method = getattr(x, method_name, None)
        if callable(method):
            return method()
        return math_fn(x)
    return call


_tanh = _method_or_math("tanh", math.tanh)
_exp = _method_or_math("exp", math.exp)
_log = _method_or_math("log", math.log)
_sqrt = _method_or_math("sqrt", math.sqrt)


def _numeric(c):
    """A condition as a NUMBER, whatever it arrived as, so `1 - c` below is arithmetic rather than a
    type error.

    A comparison on a torch tensor yields a BOOL tensor, and `1 - bool_tensor` is a RuntimeError
    ("Subtraction, the `-` operator, with a bool tensor is not supported"), so `_min2`/`_max2`/
    `_clamp` — which all build their condition as `a < b` — could not evaluate a raw tensor at all
    (measured 2026-08-13), while the module docstring above promised the same expression means the
    same thing for a CPU float and a batched CUDA tensor. `c * 1` is the one operation that fixes it
    without changing any value: exact for a float, identity for an int, `True/False -> 1/0` for a
    Python bool, bool -> int64 for a tensor. `_eval_compare` already normalises this way (`result =
    1; result = result * ok`), which is why an `expr:` whose condition is a COMPARISON worked while
    the same condition handed straight to `clamp`/`min`/`max` did not.
    """
    return c * 1


def _where(c, a, b):
    """Branch-free select. `if c: a else: b` would call `bool(c)` on the WHOLE batch and take one
    branch for all 4096 parallel envs at once; `c*a + (1-c)*b` selects elementwise instead, which is
    the only form that means the same thing for a scalar and a tensor. `c` may be a Python bool, a
    0/1 float, or a bool/float tensor — `_numeric` is what makes the last of those true rather than
    merely intended.
    """
    c = _numeric(c)
    return c * a + (1 - c) * b


def _min2(a, b):
    return _where(a < b, a, b)


def _max2(a, b):
    return _where(a > b, a, b)


def _clamp(x, lo, hi):
    return _min2(_max2(x, lo), hi)


_CALL_IMPLS = {
    "abs": abs,   # builtin abs() already dispatches via __abs__ — the same protocol tanh/exp/log/
                  # sqrt implement by hand above, since Python has no __tanh__/__exp__/__log__.
    "tanh": _tanh,
    "exp": _exp,
    "log": _log,
    "sqrt": _sqrt,
    "clamp": _clamp,
    "min": _min2,
    "max": _max2,
    "where": _where,
}
assert set(_CALL_IMPLS) == ALLOWED_CALLS, "ALLOWED_CALLS and _CALL_IMPLS have drifted apart"


def _eval_compare(node: ast.Compare, env: dict):
    # Chained comparisons (`0 < x < 1`) combine pairwise results with multiplication, not `and`, for
    # the same batch reason `_where` is branch-free: `and` forces a single Python bool out of a
    # tensor with more than one element ("ambiguous truth value"), but multiplying elementwise
    # bool/float results reduces correctly across a whole batch.
    #
    # `result` is seeded with the numeric identity `1`, not `None`/the bare first `ok`, so that even
    # an UNCHAINED single comparison (`x > 1`) comes out of the first multiplication as a numeric
    # type, not a raw Python `bool` (or a bool tensor). A bare comparison can be an entire reward row
    # (`expr: "x > 1"`) — a Python `bool` there is a batch-semantics trap in the making even though
    # `bool` arithmetic happens to work today, so we normalize to numeric unconditionally rather than
    # rely on that coincidence.
    left = _eval(node.left, env)
    result = 1
    for op, comparator in zip(node.ops, node.comparators):
        right = _eval(comparator, env)
        ok = _COMPARE_OPS[type(op)](left, right)
        result = result * ok
        left = right
    return result


def _eval(node: ast.AST, env: dict):
    node_type = type(node)
    if node_type is ast.Expression:
        return _eval(node.body, env)
    if node_type is ast.Constant:
        return node.value
    if node_type is ast.Name:
        if node.id in env:
            return env[node.id]
        raise ExprError(f"undefined name '{node.id}' — declared measures are: {sorted(env)}")
    if node_type is ast.BinOp:
        return _BIN_OPS[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if node_type is ast.UnaryOp:
        return _UNARY_OPS[type(node.op)](_eval(node.operand, env))
    if node_type is ast.Compare:
        return _eval_compare(node, env)
    if node_type is ast.Call:
        # node.func is guaranteed a bare Name with .id in ALLOWED_CALLS by parse()'s validation —
        # nothing else could have survived _validate_call above.
        args = [_eval(a, env) for a in node.args]
        return _CALL_IMPLS[node.func.id](*args)
    # Unreachable: parse() walks and rejects every other node type before an Expr is constructed.
    raise ExprError(f"internal: unhandled node type {node_type.__name__}")


class Expr:
    """A parsed, validated reward expression. Construct via `parse()`, never directly."""

    def __init__(self, tree: ast.Expression, source: str, names: frozenset):
        self._tree = tree
        self.source = source
        self.names = names

    def evaluate(self, env: dict):
        """Evaluate against `env` (name -> float or tensor). Raises `ExprError` — never `KeyError`
        or a raw arithmetic exception — on any failure, so a caller only needs to catch one type.
        """
        try:
            return _eval(self._tree, env)
        except ExprError:
            raise
        except ZeroDivisionError as exc:
            raise ExprError(f"division by zero evaluating {self.source!r}: {exc}") from exc
        except Exception as exc:  # defensive: a sandboxed evaluator should never leak a raw exception
            raise ExprError(f"error evaluating {self.source!r}: {exc}") from exc

    def __repr__(self):
        return f"Expr({self.source!r})"
