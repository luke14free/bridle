"""Grasp detection from proprioception only. Pure arithmetic — no sim, no torch.

WHY THIS IS THE ONLY GRASP SIGNAL THE LIBRARY SHIPS (2026-08-12 decision). The simulator's
`is_grasping` is object-aware and convenient and it is *privileged state*: a real SO-101 has no such
reading. A library that standardises it bakes a sim-only oracle into every contract written against
it, and the deployed policy inherits a dependency that cannot exist on hardware (CLAUDE.md's
zero-privilege rule).

TWO GATES, NOT ONE. Force and jaw position each fail alone, in opposite directions:

    force alone   an OPEN jaw pressing the TABLE reads 150-240N — coord_deploy's recorded warning:
                  it told the policy it was holding before it had ever closed (~0.5 success live).
    jaw alone     a fully-closed EMPTY gripper looks exactly like a closed loaded one.

Requiring both — loaded AND closed on something — is what makes the pair informative.

WHAT THIS SIGNAL CANNOT DO: say WHICH object is held. Force and jaw position are object-agnostic, so
`latch_on="target"` is unimplementable here and `Contract.validate()` rejects it. That is an honest
limit of proprioception, not a gap to paper over.
"""


def grasped_from_proprio(force_n, jaw_pos, signal) -> bool:
    """True iff the fingers are LOADED and CLOSED — i.e. something is held.

    `jaw_pos` is in the robot's own jaw-joint units (SO-101: open ~0, closing negative), matching
    `GraspSignal.jaw_closed_below`.
    """
    return float(force_n) >= signal.force_threshold_n and float(jaw_pos) <= signal.jaw_closed_below
