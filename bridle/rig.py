"""Your setup, as data. Frozen, validated, fingerprinted — no sim, no torch.

A `Rig` is the answer to "what robot is this, and what can it see and feel?". It exists because a
skill trained against one rig is not a skill on another, and the difference has to be *statable*
before it can be detected. Without this, "download a skill and run it" is a coin flip that looks like
a working robot right up until the numbers are bad.

Deliberately NOT a URDF parser or a scene description. A Rig captures only what changes whether a
POLICY is valid: what it commands (action space, control rate), what it grips with, and what it
observes. Geometry that a policy never perceives does not belong here — every field added is a field
that can spuriously invalidate a checkpoint.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field

GRIPPER_KINDS = ("parallel_jaw",)          # v0.1 scope; suction/multi-finger are not modelled
CONTROL_MODES = ("pd_joint_delta_pos", "pd_joint_pos", "pd_ee_delta_pose")
SENSORS = ("proprio", "rgb", "depth", "force")


@dataclass(frozen=True)
class Camera:
    """One camera, in the terms a policy actually depends on.

    Pose and FOV matter because a vision policy memorises the view: the same scene from 10cm to the
    left is a different observation, and a student trained on one will not transfer to the other.
    `name` matters because multi-camera policies index channels by camera ORDER (base+wrist = 6ch),
    so swapping two cameras silently permutes the input.
    """

    name: str
    width: int
    height: int
    pos: tuple = (0.0, 0.0, 0.0)
    target: tuple = (0.0, 0.0, 0.0)
    fov_deg: float = 60.0


@dataclass(frozen=True)
class Gripper:
    kind: str
    #: Action dimension that commands the gripper. Wrong index = the policy opens the elbow.
    dim: int
    #: Fully-open span, metres. Bounds what the gripper can pick at all.
    stroke_m: float
    #: Jaw joint position at/below which the fingers count as CLOSED, in the robot's own units.
    #: Fitted per rig (bridle.calibrate), never assumed — see bridle.signals.
    jaw_closed_below: float = -0.60


@dataclass(frozen=True)
class Rig:
    """A robot setup. `name` is documentation; every other field can invalidate a checkpoint."""

    name: str
    embodiment: str                     # "so101", "panda", ...
    dof: int                            # arm DOF excluding the gripper
    control_mode: str
    control_hz: float
    gripper: Gripper
    cameras: tuple = ()
    sensors: tuple = ("proprio",)

    def validate(self) -> None:
        if self.gripper.kind not in GRIPPER_KINDS:
            raise ValueError(f"gripper.kind must be one of {GRIPPER_KINDS}, got {self.gripper.kind!r}")
        if self.control_mode not in CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {CONTROL_MODES}, got {self.control_mode!r}")
        if self.dof <= 0:
            raise ValueError(f"dof must be > 0, got {self.dof}")
        if self.control_hz <= 0:
            raise ValueError(f"control_hz must be > 0, got {self.control_hz}")
        if self.gripper.stroke_m <= 0:
            raise ValueError(f"gripper.stroke_m must be > 0, got {self.gripper.stroke_m}")
        for s in self.sensors:
            if s not in SENSORS:
                raise ValueError(f"sensor {s!r} not one of {SENSORS}")
        if any(c.width <= 0 or c.height <= 0 for c in self.cameras):
            raise ValueError("camera width/height must be > 0")
        names = [c.name for c in self.cameras]
        if len(names) != len(set(names)):
            raise ValueError(f"camera names must be unique, got {names}")
        if "rgb" in self.sensors and not self.cameras:
            raise ValueError("sensors declare 'rgb' but no cameras are defined")

    def fingerprint(self) -> str:
        """Stable 12-hex digest. sha256 over canonical JSON — never hash(), which is salted per
        process and would make a stamped checkpoint unverifiable in the next run."""
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def describe(self) -> str:
        return f"{self.name}@{self.fingerprint()}"

    @classmethod
    def so101(cls, cameras=("base",)) -> "Rig":
        """The SO-101 arm as deployed in the reference codebase.

        The camera geometry is the "operator view" that every vision student there was trained
        against; changing it invalidates those students, which is exactly what `resolve` should say.
        """
        cams = []
        if "base" in cameras:
            cams.append(Camera(name="base", width=128, height=128,
                               pos=(-0.5, -0.15, 0.5), target=(0.25, 0.05, 0.08), fov_deg=30.0))
        if "wrist" in cameras:
            cams.append(Camera(name="wrist", width=128, height=128,
                               pos=(0.0, 0.0, 0.0), target=(0.0, 0.0, 0.1), fov_deg=60.0))
        return cls(
            name="so101-default",
            embodiment="so101",
            dof=5,
            control_mode="pd_joint_delta_pos",
            control_hz=20.0,
            gripper=Gripper(kind="parallel_jaw", dim=5, stroke_m=0.035, jaw_closed_below=-0.60),
            cameras=tuple(cams),
            sensors=("proprio", "rgb", "force"),
        )
