"""Placement geometry derived from a Release contract. Pure arithmetic — no sim, no torch.

This is library logic, not app logic, and putting it here is the fix for a specific failure. On
2026-08-11 `macro_place` computed the destination's top as `detected_z + 0.014` — a hardcoded
constant standing in for a RULE. The constant came from a platform (top 0.03 + an assumed cube half
0.014), so it silently carried a platform assumption onto a cube: a 0.012-half cube is aimed 2mm
high and a 0.016-half cube 2mm low, and nothing anywhere said so. Expressing it as
`Release.destination_top_rule` makes the assumption a value you can read, print, fingerprint, and
test — and makes switching to the measured half-size a one-field change.
"""


def destination_top_z(release, detected_z, detected_half=None) -> float:
    """World z of the destination's TOP surface — the plane the held object comes to rest on.

    `detected_half` is required only by the "detected_half" rule; the other rules ignore it, so a
    caller that has no size estimate can still be served (by a contract that declares it doesn't
    need one). A missing half under "detected_half" raises rather than silently falling back — a
    silent fallback to an assumed size is exactly the bug this rule exists to end.
    """
    rule = release.destination_top_rule
    if rule == "platform_constant":
        return float(release.platform_top_z_m)
    if rule == "assumed_half":
        return float(detected_z) + float(release.assumed_half_m)
    if rule == "detected_half":
        if detected_half is None:
            raise ValueError(
                "destination_top_rule='detected_half' needs a measured half-size, and none was "
                "supplied. Refusing to substitute the assumed half — that substitution is the "
                "2026-08-11 stack bug."
            )
        return float(detected_z) + float(detected_half)
    raise ValueError(f"unknown destination_top_rule {rule!r}")


def resting_center_z(top_z, held_half) -> float:
    """World z the held object's CENTRE settles at once it rests on `top_z`."""
    return float(top_z) + float(held_half)


def release_center_z(release, top_z, held_half) -> float:
    """World z the held object's CENTRE should be at when the jaws open.

    `resting + release.height_above_resting`. With the deployed 0.015 that is a 1.5cm drop.

    ⚠ THE DROP ITSELF IS HARMLESS, including onto a 2.4cm cube. Swept 2026-08-12 (lego-arm
    scripts/probe_stack_basin.py, 176 cells): a zero-velocity release stacks or tips on the static
    tipping condition alone, and the basin is FLAT in gap from 0mm to 22mm — restitution is 0 and the
    cube masses ~2.8g. The episode once cited here (released at +1.44cm, cube 4.6cm away on the
    table) had a lateral error of 0.73cm, i.e. WELL INSIDE the basin: what moved that cube was the
    release action, not the height. `is_supported` below is the function that matters for a stack.
    """
    return resting_center_z(top_z, held_half) + float(release.height_above_resting)


def is_supported(release, xy_error, base_half) -> bool:
    """Would the held object actually be SUPPORTED if released at this lateral error?

    True iff its centre of mass falls inside the base's top face. This is physics, not policy: a
    2.4cm cube supports a load only within ~1.2cm of centre, while the deployed release gate admits
    3.5cm. Two releases at 1.99cm and 2.06cm with essentially perfect height both slid off
    (2026-08-11). The gate tolerance lives in `release.centering_tolerance`; this function is what
    that tolerance SHOULD be checked against.
    """
    return float(xy_error) <= float(base_half)
