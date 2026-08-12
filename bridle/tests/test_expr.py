"""Unit test for bridle.skill.expr — the tier-2 reward expression language.

WHY THIS EXISTS: a reward row that the 9-term vocabulary cannot express falls back to an `expr:`
string, which a 27-30B model may author. Two properties must hold. It must be SAFE — parsed against
an ast whitelist, never eval'd, so a malformed or hostile string cannot execute anything. And it must
be TOTAL over its declared names — the compiler asks for `Expr.names` to check every free variable
resolves to a real measure, so an author's typo fails at compile time instead of at step 0 of a
multi-hour run.

Evaluation here is plain arithmetic and is tested with Python floats. In production the same
expression evaluates over batched GPU tensors: the operators are identical, which is exactly why the
language is restricted to operators that mean the same thing for both.

Run: python -m pytest bridle/tests/test_expr.py
     PYTHONPATH=. python bridle/tests/test_expr.py
"""
import math
import sys

from bridle.skill.expr import ExprError, parse

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except Exception:
        return False
    return False


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def run_checks():
    # ── arithmetic ──
    e = parse("2 * (1 - tanh(6 * abs(h - 0.015)))")
    check("free names discovered", e.names == frozenset({"h"}))
    check("source preserved verbatim", e.source == "2 * (1 - tanh(6 * abs(h - 0.015)))")
    check("evaluates", close(e.evaluate({"h": 0.015}), 2.0))
    check("evaluates off-setpoint", close(e.evaluate({"h": 0.115}),
                                          2 * (1 - math.tanh(6 * 0.1))))

    # the real descend hover row, gated
    d = parse("2.5 * (1 - tanh(6 * abs(sdz - hover))) * grasped")
    check("multi-name expression", d.names == frozenset({"sdz", "hover", "grasped"}))
    check("gate of 0 zeroes the row", close(d.evaluate({"sdz": 0.0, "hover": 0.015, "grasped": 0.0}), 0.0))
    check("gate of 1 keeps it", d.evaluate({"sdz": 0.015, "hover": 0.015, "grasped": 1.0}) > 2.49)

    # ── every whitelisted call ──
    check("clamp lower", close(parse("clamp(-3, 0, 1)").evaluate({}), 0.0))
    check("clamp upper", close(parse("clamp(3, 0, 1)").evaluate({}), 1.0))
    check("min", close(parse("min(2, 5)").evaluate({}), 2.0))
    check("max", close(parse("max(2, 5)").evaluate({}), 5.0))
    check("exp", close(parse("exp(0)").evaluate({}), 1.0))
    check("sqrt", close(parse("sqrt(9)").evaluate({}), 3.0))
    check("log", close(parse("log(1)").evaluate({}), 0.0))
    check("where picks the true branch", close(parse("where(x > 1, 10, 20)").evaluate({"x": 5}), 10.0))
    check("where picks the false branch", close(parse("where(x > 1, 10, 20)").evaluate({"x": 0}), 20.0))
    check("power", close(parse("2 ** 3").evaluate({}), 8.0))
    check("unary minus", close(parse("-x").evaluate({"x": 4}), -4.0))
    # A comparison is normalized to a numeric type unconditionally (see _eval_compare), because a
    # bare comparison can be an entire reward row and a Python `bool` there is a batch-semantics
    # trap. This check asserts the actual claim directly — `(x > 1) * 5 == 5.0` alone would pass
    # whether evaluate() returned `True` or `1.0`, since Python's `bool` is an `int` subclass and
    # arithmetic can't tell them apart; asserting `type(...) is not bool` can.
    check("bare comparison evaluates to a numeric type, not a bare bool",
          type(parse("x > 1").evaluate({"x": 2})) is not bool)
    check("comparison composes arithmetically", close(parse("(x > 1) * 5").evaluate({"x": 2}), 5.0))

    # ── Expr.names: call position vs. value position ──
    # A Name is a free variable unless it sits in CALL POSITION (the `func` of a Call) — filtering by
    # id string instead (`node.id in ALLOWED_CALLS`) would make a measure literally named `min`
    # invisible to this compile-time check. DECISION (see expr.py parse()): such a name is ALLOWED —
    # call vs. value position is unambiguous in the grammar, so there's no real hole to close by
    # refusing it, only friction to add.
    check("a builtin-shaped name used as a VALUE is a free variable",
          parse("min - 1").names == frozenset({"min"}))
    check("a builtin-shaped name used as a CALL is still not reported",
          parse("min(a, b)").names == frozenset({"a", "b"}))
    check("mixed call+value use of the same name reports only the value use",
          parse("min(min, 2)").names == frozenset({"min"}))

    # ── unbounded exponent literals refused at parse time ──
    check("small literal power still parses", parse("2 ** 8").evaluate({}) == 256)
    check("variable exponent is NOT guarded (caller's concern, not the parser's)",
          close(parse("x ** y").evaluate({"x": 2, "y": 10}), 1024.0))
    check("huge literal exponent is refused at parse time", raises(ExprError, parse, "2 ** 300000"))
    try:
        parse("2 ** 300000")
    except ExprError as e:
        check("exponent guard names the offending value", "300000" in str(e))

    # ── SAFETY: everything outside the whitelist is refused AT PARSE TIME ──
    for src, why in [
        ("__import__('os').system('rm -rf /')", "import"),
        ("open('/etc/passwd').read()", "open"),
        ("(1).__class__.__bases__", "dunder attribute"),
        ("[x for x in range(10)]", "comprehension"),
        ("lambda x: x", "lambda"),
        ("x if y else z", "conditional expression"),
        ("f'{x}'", "f-string"),
        ("x.attr", "attribute access"),
        ("print(1)", "non-whitelisted call"),
        ("x[0]", "subscript"),
        ("{'a': 1}", "dict literal"),
    ]:
        check(f"refused at parse: {why}", raises(ExprError, parse, src))

    # a refusal must NAME the construct, or a 27B author cannot self-correct
    try:
        parse("print(1)")
    except ExprError as e:
        msg = str(e)
        check("error names the offending call", "print" in msg)
        check("error lists what IS allowed", "tanh" in msg or "allowed" in msg.lower())

    # ── evaluation-time failures are ExprError, not KeyError ──
    check("missing name is an ExprError", raises(ExprError, parse("a + b").evaluate, {"a": 1}))
    check("syntax error is an ExprError", raises(ExprError, parse, "2 * (1 +"))
    check("empty source is an ExprError", raises(ExprError, parse, "   "))


def test_bridle():
    run_checks()
    assert not FAILS, f"{len(FAILS)} check(s) failed: {FAILS}"


def main():
    run_checks()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
