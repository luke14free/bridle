"""Fit a proprioceptive GraspSignal to recorded traces. Pure arithmetic — no sim, no torch.

The thresholds in a `GraspSignal` are physical claims about a specific robot ("this much force with
the jaws this closed means something is held"). Guessing them is how you get a signal that reads
"holding" while the open jaw plows the table. So they are FITTED against traces in which the
privileged `grasped` flag was recorded alongside the raw force and jaw readings — the one legitimate
use of privileged state, at training/calibration time, never in the deployed loop.

Read `Trace` rows that carry `grasped`, `force_n` and `jaw_pos` (Runner records the last two when
given an `observe_fn`), sweep a grid, and return the thresholds that best reproduce the privileged
verdict. What "best" means is deliberately asymmetric — see `fit`.
"""
from dataclasses import replace

from bridle.signals import grasped_from_proprio


def _score(rows, force_thr, jaw_thr, false_positive_weight):
    """Agreement with the privileged verdict, penalising false POSITIVES harder.

    A false positive is "I think I'm holding it" when nothing is held: the gripper freezes on empty
    air, the carry proceeds, and the whole episode is spent moving nothing. A false negative merely
    spends more of the step budget before latching. The costs are not symmetric, so the fit must not
    pretend they are.
    """
    tp = fp = fn = tn = 0
    for r in rows:
        truth = bool(r["grasped"])
        pred = (float(r["force_n"]) >= force_thr and float(r["jaw_pos"]) <= jaw_thr)
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif truth:
            fn += 1
        else:
            tn += 1
    n = max(tp + fp + fn + tn, 1)
    cost = (fp * false_positive_weight + fn) / n
    return {"force_thr": force_thr, "jaw_thr": jaw_thr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "agreement": (tp + tn) / n, "cost": cost}


def fit(rows, force_grid=None, jaw_grid=None, false_positive_weight=3.0):
    """Best (force_threshold_n, jaw_closed_below) for these rows, plus the confusion counts.

    Returns the full scored record so the caller can SEE what it bought — an agreement number with
    no confusion matrix behind it hides exactly the asymmetry that matters. Raises if the rows carry
    no positive examples: a fit against all-negatives would "succeed" at any threshold.
    """
    rows = [r for r in rows
            if r.get("force_n") is not None and r.get("jaw_pos") is not None
            and r.get("grasped") is not None]
    if not rows:
        raise ValueError("no rows carrying grasped/force_n/jaw_pos — was Runner given an observe_fn?")
    if not any(r["grasped"] for r in rows):
        raise ValueError(f"{len(rows)} rows but none with grasped=True: nothing to fit against. "
                         "Capture traces from episodes that actually grasp.")
    if force_grid is None:
        force_grid = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    if jaw_grid is None:
        jaw_grid = [-0.30, -0.40, -0.50, -0.55, -0.60, -0.65, -0.70, -0.75, -0.80]
    scored = [_score(rows, f, j, false_positive_weight) for f in force_grid for j in jaw_grid]
    # Ties broken toward the LOOSER jaw threshold then the LOWER force: among equally-costly fits,
    # prefer the one that latches earliest, since a late latch burns step budget.
    scored.sort(key=lambda s: (s["cost"], -s["jaw_thr"], s["force_thr"]))
    return scored[0]


def apply_fit(contract, fitted):
    """Return `contract` with a PROPRIO grasp signal carrying the fitted thresholds.

    Also flips `latch_on` to "any" when it was "target": a proprioceptive signal cannot identify the
    object, and `Contract.validate()` rejects the combination rather than let it look supported.
    """
    g = contract.grasp
    if g is None:
        raise ValueError(f"{contract.describe()} declares no grasp phase to calibrate")
    signal = replace(g.signal, kind="proprio",
                     force_threshold_n=float(fitted["force_thr"]),
                     jaw_closed_below=float(fitted["jaw_thr"]))
    out = replace(contract,
                  grasp=replace(g, signal=signal,
                                latch_on=("any" if g.latch_on == "target" else g.latch_on)))
    out.validate()
    return out
