"""Bind a checkpoint to the contract it was trained under. Pure dicts — no torch.

THE FAILURE THIS EXISTS TO END. A policy trained against flat platform tops was deployed against
2.4cm cube tops. Nothing objected; it ran, it looked like it was working, and it produced
pick-and-stack 0/20 for two days while three people looked for a bug in the policy. There was no bug
in the policy — it was executing a contract it had never been trained under.

A checkpoint carries its contract's fingerprint. Loading it under a different contract is an error
at STARTUP, with a field-level diff, instead of a silent 0/20 discovered by trace three days later.

MIGRATION (2026-08-12). Every checkpoint that exists today is unstamped, so `verify` cannot demand a
stamp yet without bricking the deployed system. `on_missing="warn"` is the default and prints loudly;
newly trained checkpoints stamp themselves; once the deployed set is all stamped, the default flips
to "error". An unstamped checkpoint is not "fine", it is UNKNOWN — and the warning says so.
"""
from dataclasses import asdict

KEY = "bridle_contract"


class ContractMismatch(RuntimeError):
    """Raised when a checkpoint's contract is not the one it is about to be executed under."""


def stamp(state: dict, contract) -> dict:
    """Record `contract` in a checkpoint dict (mutates and returns it).

    Stores the full contract alongside the digest, not just the digest: a bare hash can only say
    "different", and "different" without "how" sends you reading two files side by side. The stored
    copy is what makes `verify`'s diff possible.
    """
    contract.validate()
    state[KEY] = {"fingerprint": contract.fingerprint(),
                  "name": contract.name,
                  "contract": asdict(contract)}
    return state


def stamped_fingerprint(state: dict):
    rec = state.get(KEY)
    return rec.get("fingerprint") if isinstance(rec, dict) else None


def _flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def diff(state: dict, contract) -> dict:
    """Field-level differences between a checkpoint's stored contract and `contract`.

    Returns {dotted.field: (checkpoint_value, runtime_value)}. Empty means they agree.
    """
    rec = state.get(KEY) or {}
    was, now = _flat(rec.get("contract")), _flat(asdict(contract))
    keys = set(was) | set(now)
    return {k: (was.get(k), now.get(k)) for k in sorted(keys) if was.get(k) != now.get(k)}


def verify(state: dict, contract, on_missing: str = "warn") -> None:
    """Raise `ContractMismatch` unless `state` was trained under `contract`.

    on_missing="warn"  (default, migration) an unstamped checkpoint prints a warning and proceeds.
    on_missing="error" an unstamped checkpoint is itself a failure. Use once the fleet is stamped.
    """
    if on_missing not in ("warn", "error"):
        raise ValueError(f"on_missing must be 'warn' or 'error', got {on_missing!r}")
    fp = stamped_fingerprint(state)
    if fp is None:
        msg = (f"checkpoint carries NO contract stamp — cannot tell whether it was trained under "
               f"{contract.describe()}. This is UNKNOWN, not OK: the 2026-08-11 stack failure was "
               f"exactly a policy executed under a contract it never trained on.")
        if on_missing == "error":
            raise ContractMismatch(msg)
        print(f"[bridle] WARNING: {msg}", flush=True)
        return
    if fp == contract.fingerprint():
        return
    d = diff(state, contract)
    lines = "\n".join(f"    {k}: checkpoint={was!r}  runtime={now!r}" for k, (was, now) in d.items())
    if not lines:
        lines = "    (digest differs but no field diff — stored contract missing or older schema)"
    raise ContractMismatch(
        f"checkpoint was trained under {rec_name(state)}@{fp} but is being run under "
        f"{contract.describe()}.\n  Differing fields:\n{lines}"
    )


def rec_name(state: dict) -> str:
    rec = state.get(KEY) or {}
    return rec.get("name") or "contract"
