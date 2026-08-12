"""Per-step execution record.

The 2026-08-11 root cause was found by printing grip command, jaw position, contact force and
cube pose every step: the force collapsed 17.5 -> 0.5N while the cube moved at a CONSTANT 2.2cm
from the TCP, i.e. carried rather than held. No aggregate metric could have shown that.
"""
import json


class Trace:
    """Append-only structured record of one rollout. Cheap enough to leave on."""

    def __init__(self, primitive: str):
        self.primitive = primitive
        self.rows: list[dict] = []

    def record(self, step: int, **fields) -> None:
        self.rows.append({"step": step, **fields})

    def summary(self) -> dict:
        latched_at = next((r["step"] for r in self.rows if r.get("latched")), None)
        return {"n_steps": len(self.rows), "latched_at": latched_at}

    def to_jsonl(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(json.dumps({"primitive": self.primitive, **self.summary()}) + "\n")
            for r in self.rows:
                f.write(json.dumps(r) + "\n")
