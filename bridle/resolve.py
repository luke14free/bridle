"""Does this skill work on THIS rig? — and if not, what has to happen.

The heart of the product. Given a skill's trained contract and the contract you intend to run it
under, decide:

    RUN      the differences cannot affect the policy      -> use the weights
    ADAPT    the policy is recoverable by further training -> fine-tune / re-distil
    RETRAIN  the policy was trained for a different problem -> regenerate from the recipe

WHY A SEVERITY TABLE AND NOT "EQUAL OR NOT". A bare fingerprint comparison can only say *different*,
which forces a full retrain for a change that could not possibly matter — a longer step budget, a
looser deploy-side gate. That would make the honest path so expensive nobody takes it. Conversely,
treating everything as adaptable would ship a policy trained for a different action space. Both
failure modes are worse than the truth, and the truth is per-field.

EVERY SEVERITY BELOW IS AN EMPIRICAL CLAIM, and the ones that matter were paid for:

    execution.hold_steps      6 vs 16 was worth 0.40 vs 0.83 on identical seeds (p=0.00012).
                              The policy is not wrong, the requirement moved -> ADAPT.
    release.height_above_resting
                              the descend reward is an ATTRACTOR at this height; moving it changes
                              what the policy is optimising -> RETRAIN. Getting this wrong is
                              precisely the 0/20 stack failure: a policy trained for platform tops,
                              executed against cube tops, with nothing objecting.
    release.centering_tolerance
                              a DEPLOY-side gate, evaluated after the rollout. The policy never saw
                              it -> RUN. (Its training-side twin, success_tolerance, is RETRAIN.)

When a field is unknown to the table, the verdict is RETRAIN. An unrecognised difference is not
evidence of safety.
"""
from dataclasses import dataclass, field

RUN, ADAPT, RETRAIN = "run", "adapt", "retrain"
_ORDER = {RUN: 0, ADAPT: 1, RETRAIN: 2}

#: field prefix -> severity. Longest matching prefix wins, so a specific rule beats a general one.
SEVERITY = {
    # ── the rig: what the robot IS ────────────────────────────────────────────────────────────
    "rig.name": RUN,                    # documentation only
    "rig.embodiment": RETRAIN,          # different arm, different everything
    "rig.dof": RETRAIN,                 # the action vector changes shape
    "rig.control_mode": RETRAIN,        # delta-pos and abs-pos policies are not interchangeable
    "rig.control_hz": ADAPT,            # same dynamics, different discretisation
    "rig.gripper.kind": RETRAIN,
    "rig.gripper.dim": RETRAIN,         # wrong index commands the wrong joint
    "rig.gripper.stroke_m": ADAPT,      # bounds what is graspable; policy is recoverable
    "rig.gripper.jaw_closed_below": RUN,  # a SIGNAL threshold, re-fitted per rig; not learned
    "rig.sensors": RETRAIN,             # observation modality change
    "rig.cameras": ADAPT,               # a vision student memorises the view -> needs re-distil,
                                        # but the teacher and the task are unchanged
    # ── actuation ─────────────────────────────────────────────────────────────────────────────
    "actuation.gripper_dim": RETRAIN,
    "actuation.action_lo": RETRAIN,
    "actuation.action_hi": RETRAIN,
    # ── execution: how the loop runs ──────────────────────────────────────────────────────────
    "execution.budget": RUN,            # more or fewer steps to attempt; the policy is unchanged
    "execution.hold_steps": ADAPT,
    "execution.linger_steps": RUN,      # a deploy-side exit rule, not a training signal
    "execution.goal_tolerance": RUN,
    "execution.force_threshold": RUN,
    "execution.gripper": RETRAIN,       # freeze-on-latch vs always-zero changes the action stream
    "execution.terminate": RETRAIN,     # a different termination is a different task
    "execution.terminate_pre_step": RETRAIN,
    # ── grasp ─────────────────────────────────────────────────────────────────────────────────
    "grasp.latch_on": ADAPT,
    "grasp.signal.kind": ADAPT,         # privileged -> proprio changes WHEN the latch fires
    "grasp.signal.force_threshold_n": RUN,   # fitted per rig, not learned
    "grasp.signal.jaw_closed_below": RUN,
    # ── release: placement geometry ───────────────────────────────────────────────────────────
    "release.height_above_resting": RETRAIN,
    "release.success_tolerance": RETRAIN,      # training's success criterion
    "release.success_height_band": RETRAIN,
    "release.centering_tolerance": RUN,        # deploy-side gate only
    "release.destination_top_rule": RUN,       # how the GOAL is computed, not how the policy acts
    "release.assumed_half_m": RUN,
    "release.platform_top_z_m": RUN,
    "release.ramp_steps": RUN,                 # post-release jaw motion; the rollout is over
    # ── identity ──────────────────────────────────────────────────────────────────────────────
    "name": RUN,
}


def severity_of(field_path: str) -> str:
    """Severity for a dotted field. Longest matching prefix wins; unknown fields are RETRAIN."""
    best, best_len = None, -1
    for prefix, sev in SEVERITY.items():
        if (field_path == prefix or field_path.startswith(prefix + ".")) and len(prefix) > best_len:
            best, best_len = sev, len(prefix)
    return best if best is not None else RETRAIN


@dataclass
class Resolution:
    """What to do with this skill on this rig, and why."""

    verdict: str
    reasons: list = field(default_factory=list)     # (field, severity, trained, target)

    @property
    def ok(self) -> bool:
        return self.verdict == RUN

    def explain(self) -> str:
        if self.verdict == RUN and not self.reasons:
            return "RUN — the contracts are identical."
        head = {RUN: "RUN — every difference is inert for the policy.",
                ADAPT: "ADAPT — the policy is recoverable by further training.",
                RETRAIN: "RETRAIN — trained for a different problem."}[self.verdict]
        lines = [f"    [{sev:7s}] {f}: trained={was!r} target={now!r}"
                 for f, sev, was, now in self.reasons]
        return head + ("\n" + "\n".join(lines) if lines else "")


def _flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def resolve_contracts(trained, target) -> Resolution:
    """Compare two Contracts and decide run / adapt / retrain.

    Fast path: identical fingerprints resolve to RUN without walking any fields.
    """
    from dataclasses import asdict

    if trained.fingerprint() == target.fingerprint():
        return Resolution(RUN, [])

    was, now = _flat(asdict(trained)), _flat(asdict(target))
    reasons, verdict = [], RUN
    for k in sorted(set(was) | set(now)):
        if was.get(k) == now.get(k):
            continue
        sev = severity_of(k)
        reasons.append((k, sev, was.get(k), now.get(k)))
        if _ORDER[sev] > _ORDER[verdict]:
            verdict = sev
    # Report worst-first: the field that forces the verdict should be the first thing read.
    reasons.sort(key=lambda r: (-_ORDER[r[1]], r[0]))
    return Resolution(verdict, reasons)


def resolve(app, rig, target_contract=None) -> Resolution:
    """Resolve an App against a Rig.

    An app with no artifacts always RETRAINs — there is nothing to run. That is the normal state for
    a freshly authored recipe, and the reason a recipe is the primary artifact rather than a fallback.
    """
    target = target_contract if target_contract is not None else app.contract_for(rig)
    best = None
    for art in getattr(app, "artifacts", ()) or ():
        r = resolve_contracts(art.contract, target)
        if best is None or _ORDER[r.verdict] < _ORDER[best.verdict]:
            best = r
        if best.verdict == RUN:
            break
    if best is None:
        return Resolution(RETRAIN, [("artifacts", RETRAIN, "none", "none")])
    return best
