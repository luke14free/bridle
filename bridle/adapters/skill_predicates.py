"""bridle.adapters.skill_predicates — the predicate mini-language a skill document's truths parse to.

WHAT THIS IS: a parser and evaluator, not an env adapter. A `gate:`, a `predicate:` and a `success:`
line in a skill document are all the same little language — a bare vocabulary name, or a call over
vocabulary names (`and_(grasped, above_z(z=0.06))`), plus the bracket sugar `all[...]`/`any[...]`
the design doc §4 and the acceptance fixture write. This module owns that language end to end: the
`ast` whitelist, the desugarer, argument resolution, and the seventeen predicates themselves.

WHY IT IS NOT IN `skill_env.py`. It re-implements the whitelist discipline `bridle/skill/expr.py`
already owns for reward expressions, for the same stated reason (the author is a 27-30B model, so
`().__class__.__bases__` has to be a PARSE-time refusal rather than something that depends on never
being tried) — and that is a different job from reading a quantity off a simulator. It is also the
part most likely to grow: `forall`/`for_n` below are placeholders until the scene block is
synthesised into env objects, and quantifiers landing here should not enlarge the adapter.

WHAT IT KNOWS ABOUT THE ENV: nothing directly. Every predicate takes a `ctx` — a
`skill_env.MeasureContext` — and reads the scene only through `ctx.measure(...)`, `ctx.slots`,
`ctx.goal`, `ctx.object_p` and `ctx.attr(...)`. That is the whole seam, and it is why the dependency
runs one way: `skill_env` imports this module, never the reverse. `SkillEnvError` and `_norm` are
DEFINED here, at the bottom of that edge, and re-exported by `skill_env` so every existing
`except skill_env.SkillEnvError` keeps catching the same class object.

Every predicate returns a 0.0/1.0 FLOAT tensor, not a bool one, so it can be multiplied into a gate
and subtracted from 1 without `RuntimeError: Subtraction, the `-` operator, with a bool tensor is
not supported`. `compile._numeric` now normalises a bool CONDITION at the source, but a predicate is
a VALUE — it is multiplied into gates and subtracted from 1 by term math that never goes near
`_numeric` — so returning a float here is still the contract, not a redundancy.

Torch is imported lazily inside the functions that need it, matching `adapters/preflight.py` and
`skill_env.py`: this module must import on a box with no torch so that the `PREDICATE_FNS` key-set
assert at the bottom runs in `bridle/tests/`.
"""
import ast

from bridle.skill.vocab import PREDICATES


class SkillEnvError(Exception):
    """A reading this adapter cannot take against the env it was handed.

    Same shape of message as `SpecError`/`CompileError` — what was asked for, what is missing, what
    to write instead — because the reader is the same 27-30B author, one tier further down. A measure
    that cannot be read RAISES rather than returning a plausible substitute: a wrong-but-finite
    number trains a policy, logs clean, and is indistinguishable from a right one.
    """


def _norm(v):
    import torch
    return torch.linalg.norm(v, dim=-1)


# ── predicates ──────────────────────────────────────────────────────────────────────────────────
# A predicate field is a bare name or a nested call over existing names (`spec._check_predicate`),
# and `success:` additionally uses the bracket form the design doc's §4 example writes,
# `all[a, b, c]`. Both are parsed with `ast` against a whitelist — never `eval` — for the same reason
# `expr.py` is: the author is a 27-30B model and `().__class__.__bases__` has to be a parse-time
# refusal rather than something that depends on it never being tried.
#
# Every predicate returns a 0.0/1.0 FLOAT tensor, not a bool one, so it can be multiplied into a
# gate and subtracted from 1 without torch's bool-tensor subtraction error (module docstring).

#: `ast.List`/`ast.Tuple` are admitted for ONE reason: `and_`/`or_` declare `terms: list[predicate]`,
#: so `and_(terms=[grasped, above_z(z=0.06)])` is a spelling the vocabulary's own type invites even
#: though every chassis writes the positional form. Nothing else is: no attribute access, no
#: subscript, no arithmetic — same whitelist discipline as `expr.py`, for the same reason.
_PRED_NODES = frozenset({ast.Expression, ast.Call, ast.Name, ast.Constant, ast.Load, ast.keyword,
                         ast.UnaryOp, ast.USub, ast.UAdd, ast.List, ast.Tuple})

_BRACKET_SUGAR = {"all": "and_", "any": "or_"}


def _desugar_brackets(text):
    """`all[a, b]` -> `and_(a, b)`, `any[...]` -> `or_(...)`.

    The `success:` grammar is the one thing `spec.py` explicitly does NOT parse ("that grammar
    belongs to whoever evaluates it"), and the document form in the design doc §4 and in the
    acceptance fixture is the bracket one. It is sugar and nothing more: the bracket lowers to the
    `and_`/`or_` that already exist in PREDICATES, so there is one set of semantics, not two.

    THE MATCH IS ANCHORED AT A WORD BOUNDARY. It was not, and `all`/`any` are suffixes of ordinary
    words: `overall[x]` rewrote to `overand_(x)` and `many[q]` to `mor_(q)` (measured 2026-08-13).
    Neither is a legal predicate either way, so nothing was mis-EVALUATED — but the refusal named a
    predicate the author never typed, and this file's whole premise is that the error message is the
    API for an author who cannot introspect it.
    """
    out, depth_stack = [], []
    i = 0
    while i < len(text):
        matched = None
        # A sugar word only counts at the start of an identifier: `all[` yes, `overall[` no.
        boundary = i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
        for word, replacement in (_BRACKET_SUGAR.items() if boundary else ()):
            if text.startswith(word + "[", i):
                matched = (word, replacement)
                break
        if matched:
            out.append(matched[1] + "(")
            depth_stack.append(len(out))
            i += len(matched[0]) + 1
            continue
        char = text[i]
        if char == "[":
            depth_stack.append(None)
            out.append(char)
        elif char == "]":
            opened = depth_stack.pop() if depth_stack else None
            out.append(")" if opened is not None else "]")
        else:
            out.append(char)
        i += 1
    return "".join(out)


class _Args:
    """The arguments of one predicate call, resolvable by keyword OR position.

    `and_(grasped, above_z(z=0.06))` writes its operands positionally and `above_z(z=0.06)` writes
    its own by keyword; both spellings appear in `vocab.CHASSIS`, so both resolve here.
    """

    def __init__(self, ctx, node, source):
        self.ctx = ctx
        self.source = source
        self.positional = list(node.args) if isinstance(node, ast.Call) else []
        self.keyword = {kw.arg: kw.value for kw in node.keywords} if isinstance(node, ast.Call) else {}

    def _node(self, index, name):
        if name in self.keyword:
            return self.keyword[name]
        if index is not None and index < len(self.positional):
            return self.positional[index]
        return None

    def predicate(self, index, name):
        node = self._node(index, name)
        if node is None:
            raise SkillEnvError(f"{self.source}: `{name}` is required and names a predicate")
        return _eval_predicate_node(self.ctx, node, self.source)

    def all_predicates(self):
        nodes, queue = [], list(self.positional) + [self.keyword[k] for k in sorted(self.keyword)]
        for node in queue:
            # `and_(terms=[a, b])` and `and_(a, b)` are the same conjunction; a declared
            # `list[predicate]` is flattened rather than being a second grammar.
            nodes.extend(node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node])
        if not nodes:
            raise SkillEnvError(f"{self.source}: needs at least one operand predicate")
        return [_eval_predicate_node(self.ctx, n, self.source) for n in nodes]

    def number(self, index, name, default=None, required=False):
        node = self._node(index, name)
        if node is None:
            if required:
                raise SkillEnvError(f"{self.source}: `{name}` is required and is a number")
            return default
        value = _literal(node)
        if value is None:
            raise SkillEnvError(
                f"{self.source}: `{name}` has to be a number here, and {ast.unparse(node)!r} is not "
                f"one. A `params.X` reference is substituted before this point; an expression over "
                f"scene attributes (`bin.inner_radius - 0.3*object.half_size`) needs the scene "
                f"binding this phase does not build (phase2-decisions, scope limit)")
        return float(value)

    def flag(self, index, name, default):
        node = self._node(index, name)
        if node is None:
            return default
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        raise SkillEnvError(f"{self.source}: `{name}` is true or false, got {ast.unparse(node)!r}")

    def identifier(self, index, name, default=None):
        node = self._node(index, name)
        if node is None:
            return default
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        raise SkillEnvError(f"{self.source}: `{name}` names a point, got {ast.unparse(node)!r}")


def _literal(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _literal(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    return None


def _f(mask, ctx):
    """A bool tensor as the 0.0/1.0 float every predicate returns."""
    import torch
    return mask.to(dtype=torch.float32) if torch.is_tensor(mask) else \
        torch.full((ctx.num_envs,), float(mask), device=ctx.device)


def _p_grasped(ctx, args):
    """PRIVILEGED sim ground truth (`agent.is_grasping`, contact + force + angle). Allowed as a
    training-time gate; never in a deployed switching rule — the zero-privilege rule, CLAUDE.md."""
    held = ctx.info.get("is_grasped") if isinstance(ctx.info, dict) else None
    if held is None:
        held = ctx.attr("agent", "the grasp predicate").is_grasping(ctx.held)
    return _f(held, ctx)


def _p_not_grasped(ctx, args):
    return 1.0 - _p_grasped(ctx, args)


def _p_above_z(ctx, args):
    return _f(ctx.measure("object_z") > args.number(0, "z", required=True), ctx)


def _p_below_height(ctx, args):
    return _f(ctx.measure("object_z") < args.number(0, "z", required=True), ctx)


def _anchor_xy(ctx, name):
    import torch
    if name is None:
        return ctx.goal[..., :2]
    value = getattr(ctx.env, name, None)
    if value is None:
        raise SkillEnvError(f"anchor {name!r} is not an attribute of {type(ctx.env).__name__}; the "
                            f"scene block is not bound to env objects in this phase, so an anchor "
                            f"has to be an env attribute holding an (N,3) or (N,2) point")
    point = value.pose.p if hasattr(value, "pose") else torch.as_tensor(value, device=ctx.device)
    return point[..., :2]


def _p_within_radius(ctx, args):
    anchor = _anchor_xy(ctx, args.identifier(0, "anchor"))
    radius = args.number(1, "radius_expr", required=True)
    return _f(_norm(ctx.object_p[..., :2] - anchor) < radius, ctx)


def _p_in_cylinder(ctx, args):
    """The container-interior test. `in_cylinder` declares radius and floor but NO anchor
    (vocab.PREDICATES), so the anchor is the goal point — the only centre an unqualified container
    test can mean in this corpus."""
    radius = args.number(0, "radius", required=True)
    floor = args.number(1, "floor", default=0.0)
    inside = _norm(ctx.object_p[..., :2] - _anchor_xy(ctx, None)) < radius
    return _f(inside & (ctx.object_p[..., 2] > floor), ctx)


def _p_at_rest(ctx, args):
    """Either bound may be omitted. NEVER gate on angular alone for a grasped object — CLAUDE.md
    gotcha (2), ~98% contact-solver noise — but that is a document-level choice this cannot police."""
    import torch
    linear = args.number(0, "linear")
    angular = args.number(1, "angular")
    if linear is None and angular is None:
        raise SkillEnvError("at_rest: give at least one of `linear` / `angular`; with neither it is "
                            "a check that can never fail")
    ok = torch.ones(ctx.num_envs, dtype=torch.bool, device=ctx.device)
    if linear is not None:
        ok = ok & (ctx.measure("object_linear_velocity") < linear)
    if angular is not None:
        ok = ok & (ctx.measure("object_angular_velocity") < angular)
    return _f(ok, ctx)


def _up_axis(q):
    """The object's own +z axis in world coordinates, from a wxyz quaternion `(N,4)`."""
    import torch
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1)


def _p_undisturbed(ctx, args):
    """Moved less than `drift` and tilted less than `tilt` SINCE RESET — so the tilt is measured
    against the reset orientation's own up-axis, not against world vertical: an object that spawns
    on its side is not "tilted" until something tips it."""
    import torch
    drift = args.number(0, "drift", required=True)
    tilt = args.number(1, "tilt", required=True)
    up = _up_axis(ctx.held.pose.q)
    anchor = ctx.slots.slot("frame.object_up0", init=lambda: up, width=3)
    cosine = torch.clamp((up * anchor).sum(dim=-1), -1.0, 1.0)
    ok = (ctx.measure("object_xy_drift_from_reset") < drift) & (torch.acos(cosine) < tilt)
    return _f(ok, ctx)


def _p_height_above_resting_in(ctx, args):
    """`height_above_resting` in [0, band]. descend uses a band INSTEAD of an at-rest gate because a
    held cube being positioned is never stationary and the at-rest gate never latched (eval
    2026-06-03: descend_low_once=1.0 while obj_at_rest=0.06). Reads the measure it NAMES — if the
    band is meant against the destination seat, bind `resting_surface_z` (see
    `_m_height_above_resting`) rather than expecting this predicate to switch measures silently.

    IT IS BOUNDED BELOW, and its sibling `below_resting_height(band)` is not. Use this one when
    below-the-surface is a FAILURE; use the sibling when it merely still counts as low, which is how
    `descend_env.py`'s own `low` gate is written. The two disagree on 37 of 64 forced
    full-criterion descend states (measured 2026-08-12)."""
    band = args.number(0, "band", required=True)
    height = ctx.measure("height_above_resting")
    return _f((height >= 0.0) & (height <= band), ctx)


def _p_below_resting_height(ctx, args):
    """`height_above_resting < band`, UNBOUNDED BELOW — `descend_env.py`'s `low` gate.

    The lower bound `height_above_resting_in` imposes is wrong for THIS criterion on purpose: a cube
    pressed BELOW its resting height is still low, and the crush penalty (descend's
    `-3.0*clamp(-sdz, min=0)`, which exists because pressing to dz=0 broke 16/16 grasps on
    2026-06-04), not the success gate, is what handles pressing.

    Measured 2026-08-12: routing descend's criterion through `height_above_resting_in` instead
    disagreed on 37 of 4456 sampled states on this component alone, on 64/64 of the states below the
    seat, and on 37 of 64 at FULL criterion level once `centered` is forced true.

    Reads the measure it NAMES — if the band is meant against a destination seat, bind
    `resting_surface_z` (see `_m_height_above_resting`) rather than expecting this predicate to
    switch measures silently.
    """
    band = args.number(0, "band", required=True)
    height = ctx.measure("height_above_resting")
    return _f(height < band, ctx)


def _p_and(ctx, args):
    out = None
    for term in args.all_predicates():
        out = term if out is None else out * term
    return out


def _p_or(ctx, args):
    """De Morgan rather than `max`, so the result stays a product of floats: `1 - prod(1 - p)`."""
    out = None
    for term in args.all_predicates():
        complement = 1.0 - term
        out = complement if out is None else out * complement
    return 1.0 - out


def _p_not(ctx, args):
    return 1.0 - args.predicate(0, "term")


def _p_sustained(ctx, args):
    """`predicate` has held for `k` steps.

    `consecutive=True` (7 primitives): one failing step resets the streak. `consecutive=False`
    (grab/sphere_grab): the count ACCUMULATES and never resets on a slip. That is not cosmetic — the
    cumulative version false-passed flaky grips before 2026-06-25 (vocab.PREDICATES).

    The streak advances at most once per control step even when the same predicate is read from both
    `evaluate()` and `compute_dense_reward()` — see `StateSlots.fresh_rows`."""
    import torch
    value = args.predicate(0, "predicate")
    k = args.number(1, "k", default=1.0)
    consecutive = args.flag(2, "consecutive", True)
    name = f"pred.sustained.{args.source}"
    streak = ctx.slots.slot(name, init=lambda: torch.zeros_like(value))
    advanced = (streak + 1.0) * value if consecutive else streak + value
    fresh = ctx.slots.fresh_rows(name, ctx.elapsed)
    streak.copy_(advanced if fresh is None else torch.where(fresh > 0, advanced, streak))
    return _f(streak >= k, ctx)


def _p_latched(ctx, args):
    """OR-accumulated: once true it stays true for the episode. move_to_target/move_over_bin's
    success — the bonus it feeds pays every remaining step. Idempotent, so it needs no step guard."""
    import torch
    value = args.predicate(0, "predicate")
    latch = ctx.slots.slot(f"pred.latched.{args.source}", init=lambda: torch.zeros_like(value))
    latch.copy_(torch.maximum(latch, value))
    return latch.clone()


def _unimplemented_quantifier(name, over_doc):
    def fn(ctx, args):
        raise SkillEnvError(
            f"`{name}` quantifies over a COLLECTION ({over_doc}) and this phase does not build one: "
            f"the `scene:` block is parsed and fingerprinted but is not synthesised into env objects "
            f"(phase2-decisions, scope limit), so there is nothing to enumerate. Write the criterion "
            f"over the single held object, or wait for the scene-generation phase")
    return fn


PREDICATE_FNS = {
    "grasped": _p_grasped,
    "not_grasped": _p_not_grasped,
    "above_z": _p_above_z,
    "below_height": _p_below_height,
    "within_radius": _p_within_radius,
    "in_cylinder": _p_in_cylinder,
    "at_rest": _p_at_rest,
    "undisturbed": _p_undisturbed,
    "height_above_resting_in": _p_height_above_resting_in,
    "below_resting_height": _p_below_resting_height,
    "and_": _p_and,
    "or_": _p_or,
    "not_": _p_not,
    "sustained": _p_sustained,
    "latched": _p_latched,
    "forall": _unimplemented_quantifier("forall", "e.g. 'bricks_in_bin'"),
    "for_n": _unimplemented_quantifier("for_n", "e.g. 'bricks_in_bin'"),
}

assert set(PREDICATE_FNS) == set(PREDICATES), (
    f"PREDICATE_FNS and vocab.PREDICATES disagree: missing "
    f"{sorted(set(PREDICATES) - set(PREDICATE_FNS))}, extra "
    f"{sorted(set(PREDICATE_FNS) - set(PREDICATES))}")


def _eval_predicate_node(ctx, node, source):
    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
    else:
        raise SkillEnvError(f"{source}: {ast.unparse(node)!r} is not a predicate — write a bare name "
                            f"from the vocabulary, or a call over them like "
                            f"`and_(grasped, above_z(z=0.06))`")
    fn = PREDICATE_FNS.get(name)
    if fn is None:
        raise SkillEnvError(f"{source}: unknown predicate {name!r} — legal: "
                            f"{', '.join(sorted(PREDICATES))}")
    # The unparsed call text is the slot identity for a stateful predicate: two `sustained(...)`
    # rows over different operands must not share one streak counter, and the same one read twice
    # must not become two.
    return fn(ctx, _Args(ctx, node, ast.unparse(node)))


def _eval_predicate_text(ctx, text):
    source = _desugar_brackets(text)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise SkillEnvError(f"predicate {text!r} does not parse: {exc}") from exc
    for node in ast.walk(tree):
        if type(node) not in _PRED_NODES:
            raise SkillEnvError(
                f"predicate {text!r} contains {type(node).__name__}, which a predicate expression "
                f"may not: it is a bare name or a call over names, nothing else")
    return _eval_predicate_node(ctx, tree.body, text)
