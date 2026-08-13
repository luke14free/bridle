"""bridle.adapters.snapshot_ref — checking a skill document's `init:` claims against the capture.

THE SEAM, AND WHY IT IS HERE AND NOT IN CORE. `bridle.skill` is stdlib-only and knows nothing about
storage formats: it validates the SHAPE of an `init:` claim and carries it (`spec._parse_init`).
Turning `snapshot: descend_init_move` into a path, opening a `.npz`, canonicalising it and reading
the provenance its capture tool wrote is a BACKEND concern, so it lives beside the other backend
seam — `bridle.adapters.env_ref`, which resolves `env_id:` for exactly the same reason. A different
snapshot store (a different capture format, a remote artifact registry) is a NEW MODULE beside this
one; the schema, the refusals and the CLI wiring above this line do not move.

WHAT IT CHECKS, IN THE ORDER THEY ARE WORTH.

  1. PROVENANCE (`init.after`). A skill trains on the states its predecessors actually leave it —
     `descend` is `reach -> grab -> lift -> move_to_target -> descend`, so its initiation set is
     move's handoff. The capture files already record which primitive they came from, and until
     today nothing compared that to the document. On 2026-06-10 the bytes behind `descend_init` were
     replaced by a `grab` handoff and the file SAID SO in its own metadata (`"prim": "grab"`) while
     the document and the env docstring both said move; that contradiction sat unread for two months
     and cost roughly a day of GPU across two dead runs. This comparison is one string against one
     string, needs no simulator and no GPU, and would have refused in the first second.

  2. CONTENT (`init.sha256`). The narrower case provenance cannot see: the same predecessor,
     recaptured or edited into different data. A digest catches it; it costs one read of the file.

THREE OUTCOMES, NEVER TWO. `OK`, `MISMATCH` (a refusal) and `NOT_CHECKED` — the last for a claim
that could not be evaluated at all: no file reachable from here, or a capture that records no
provenance. A checker that reported the third as a pass would be a check that passes by not running,
which is the shape this project has been bitten by repeatedly (`env_ref`'s module docstring records
the `env_id: ThisEnvDoesNotExist-v9` version of it).

TWO METADATA DIALECTS ARE READ, because two capture tools wrote them and neither is going to be
rewritten to satisfy this module:

    {"primitive": "move_handoff_probe", "source": "chain_handoff",
     "program": "reach_grab_lift_move", "after": "move_to_target", "captured": 1678, ...}
    {"source": "coord_chain_handoff", "prim": "grab", "n": 4456}

`after` and `prim` are the same fact under two names. A file carrying neither is NOT CHECKED.
"""
import glob
import hashlib
import json
import os
import zipfile

__all__ = [
    "OK", "MISMATCH", "NOT_CHECKED", "Finding",
    "snapshot_digest", "snapshot_provenance", "resolve_snapshot", "check_init",
]

OK = "ok"
MISMATCH = "mismatch"
NOT_CHECKED = "not_checked"

#: The metadata keys that name the PREDECESSOR, best dialect first. See the module docstring.
#: `primitive:` is deliberately NOT here: in the `chain_handoff` dialect it names the CAPTURE TOOL
#: (`move_handoff_probe`), not the skill whose handoff was captured, and reading it as a predecessor
#: would invent an agreement or a mismatch out of a field that means something else.
_PREDECESSOR_KEYS = ("after", "prim")

#: Where a snapshot key is materialised, relative to a repo root. `lego_arm.snapshots.SNAPSHOTS_DIR`
#: is `<repo>/snapshots/<key>.npz`; kept as a string here so this module imports with no lego-arm on
#: the path (a `bridle skill check` run from anywhere must still work, and report NOT CHECKED rather
#: than explode, when the store is not reachable).
_STORE_DIRNAME = "snapshots"
_SUFFIXES = (".npz", "")


class Finding:
    """One checked claim: `(status, label, detail)`, plus the path it was checked against.

    A list of these — not a bool — because a document can make several claims and the reader needs
    to see which one was OK, which was NOT CHECKED and which refused. Collapsing them loses exactly
    the distinction the module exists to preserve.
    """

    __slots__ = ("status", "label", "detail", "path")

    def __init__(self, status, label, detail, path=None):
        self.status = status
        self.label = label
        self.detail = detail
        self.path = path

    @property
    def refused(self):
        return self.status == MISMATCH

    def line(self):
        head = {OK: "OK", MISMATCH: "MISMATCH", NOT_CHECKED: "NOT CHECKED"}[self.status]
        return f"{self.label}: {head} — {self.detail}"

    def __repr__(self):
        return f"<Finding {self.status} {self.label}: {self.detail}>"


# ── the file ────────────────────────────────────────────────────────────────────────────────────

def snapshot_digest(path) -> str:
    """sha256 over a CANONICAL FORM of the capture's contents, not over the file's bytes.

    WHY NOT THE BYTES. A `.npz` is a zip, and a zip stores a compression choice, a member order and
    per-member timestamps. Hashing the bytes would make a re-save of the IDENTICAL arrays read as a
    CHANGED initiation set — a false refusal every time a capture is copied through a tool that
    rewrites the container, which is the fastest way to get a guard switched off. Measured on
    `snapshots/descend_init_move.npz` (2026-08-13): `np.savez` of the same arrays differs from the
    shipped `np.savez_compressed` file byte-for-byte, and both produce the digest below. (Timestamps
    happen not to bite today — numpy pins every member's zip date to 1980-01-01 — but that is
    numpy's choice to change, not a property to depend on.) So the digest covers the members' NAMES
    and their uncompressed CONTENTS, in sorted name order, each length-prefixed so no concatenation
    of two members can collide with another pair:

        sha256( for name in sorted(members): name || 0x00 || len(data) || 0x00 || data )

    WHAT IS THEREFORE INVARIANT: zip member mtimes, the zip's member ORDER, the compression method
    and level, `savez` versus `savez_compressed`. WHAT IS NOT, and the honest limit of this
    function: the member payloads are `.npy` blobs, so the digest still moves if a numpy version
    writes a different `.npy` header for the same array (header padding, a dtype spelled
    differently), or if an array is re-created with a different dtype or shape. It is a canonical
    form of the CONTAINER, not a canonical form of the tensors. Re-saving the same arrays from the
    same numpy is the assumption; re-generating a capture is a genuinely different initiation set
    and is supposed to move the digest.

    NO NUMPY, NO PICKLE. `zipfile` is stdlib and reads the member payloads without interpreting
    them, so this runs in an interpreter with no scientific stack and never unpickles an object
    array to compute a digest. A non-zip file (some other capture format) is hashed byte-for-byte,
    and the docstring above is then simply the whole story for it.
    """
    h = hashlib.sha256()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                data = z.read(name)
                h.update(name.encode("utf-8"))
                h.update(b"\0")
                h.update(str(len(data)).encode("ascii"))
                h.update(b"\0")
                h.update(data)
        return h.hexdigest()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _npy_member(blob):
    """`(descr, payload)` for one `.npy` blob — its dtype string and the bytes after the header.

    Twelve lines of stdlib instead of a numpy import, because the whole point of this adapter is
    that a document can be checked in an interpreter that has no scientific stack. The format is
    fixed and versioned: magic, (major, minor), a 2- or 4-byte little-endian header length, then an
    ASCII dict literal, then the data.
    """
    if not blob.startswith(b"\x93NUMPY"):
        return None, blob
    major = blob[6]
    if major == 1:
        hlen = int.from_bytes(blob[8:10], "little")
        start = 10
    else:
        hlen = int.from_bytes(blob[8:12], "little")
        start = 12
    header = blob[start:start + hlen].decode("latin-1")
    descr = ""
    marker = "'descr':"
    if marker in header:
        rest = header.split(marker, 1)[1].lstrip()
        quote = rest[0] if rest[:1] in ("'", '"') else None
        if quote:
            descr = rest[1:].split(quote, 1)[0]
    return descr, blob[start + hlen:]


def _meta_member(path):
    """The `meta` member of a capture as `(descr, payload)`, or `(None, None)`."""
    if not zipfile.is_zipfile(path):
        return None, None
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if os.path.splitext(os.path.basename(name))[0] == "meta":
                return _npy_member(z.read(name))
    return None, None


#: Pickle opcodes that push a literal (a dict key or a scalar value) rather than build something.
#: Used to read a pickled `meta` dict WITHOUT unpickling it — see `snapshot_provenance`.
_PICKLE_LITERALS = frozenset((
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
    "SHORT_BINSTRING", "BINSTRING", "STRING",
    "BININT", "BININT1", "BININT2", "INT", "LONG", "LONG1", "LONG4", "FLOAT", "BINFLOAT",
))


def snapshot_provenance(path) -> dict:
    """`{key: value}` of the provenance a capture recorded, or `{}` when it recorded none.

    NEVER UNPICKLES, and the two dialects are why that takes any code at all:

      `chain_handoff` writes `np.array(json.dumps(meta))` — a 0-d unicode array, so the payload is
      UTF-32 and the JSON is decoded straight out of it.

      `coord_chain_handoff` writes `np.array(meta)` on a dict — a 0-d OBJECT array, i.e. a pickle.
      `pickletools.genops` DISASSEMBLES a pickle without executing it, so the dict's keys and values
      are read off the opcode stream: `EMPTY_DICT`, `MARK`, then key, value, key, value, then
      `SETITEMS`. Anything outside that span (numpy's own `_reconstruct` preamble) is skipped. The
      alternative — `np.load(allow_pickle=True)` — executes whatever the file says in order to read
      a label off it, which is not a trade a validator should make.

    A capture in neither dialect yields `{}`, and the caller must report that as NOT CHECKED rather
    than as agreement.
    """
    descr, payload = _meta_member(path)
    if payload is None:
        return {}
    if descr and descr[1:2] == "U":
        codec = "utf-32-be" if descr.startswith(">") else "utf-32-le"
        text = payload.decode(codec, "replace").rstrip("\x00")
        try:
            loaded = json.loads(text)
        except ValueError:
            return {}
        return {str(k): loaded[k] for k in loaded} if isinstance(loaded, dict) else {}

    import pickletools
    literals, depth, collecting = [], 0, False
    try:
        for op, arg, _pos in pickletools.genops(payload):
            if op.name == "EMPTY_DICT":
                collecting, depth = True, 0
            elif collecting and op.name == "MARK":
                depth += 1
            elif collecting and op.name in ("SETITEMS", "SETITEM"):
                break
            elif collecting and depth and op.name in _PICKLE_LITERALS:
                literals.append(arg)
    except Exception:                                             # noqa: BLE001 — a malformed or
        return {}                                                 # unknown pickle is "no metadata"
    return {str(literals[i]): literals[i + 1] for i in range(0, len(literals) - 1, 2)}


def resolve_snapshot(name, search_dir=None):
    """`descend_init_move` -> `<repo>/snapshots/descend_init_move.npz`, or None.

    Walks UP from `search_dir` (the document's own directory) looking for a `snapshots/` store, so a
    document at `primitives/<p>/skill.yaml` finds the repo's store without either of them naming a
    path. `BRIDLE_SNAPSHOT_DIR` overrides, for a store that is not under the document. Returning
    None is a legitimate state — the machine reading a document need not hold its captures — and the
    caller reports NOT CHECKED for it, never a pass.
    """
    if not name:
        return None
    if os.path.isabs(name) and os.path.exists(name):
        return name
    dirs = []
    override = os.environ.get("BRIDLE_SNAPSHOT_DIR")
    if override:
        dirs.append(override)
    here = os.path.abspath(search_dir or os.getcwd())
    while True:
        dirs.append(os.path.join(here, _STORE_DIRNAME))
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    for d in dirs:
        for suffix in _SUFFIXES:
            candidate = os.path.join(d, name + suffix)
            if os.path.isfile(candidate):
                return candidate
        hits = sorted(glob.glob(os.path.join(d, name + ".*")))
        if hits:
            return hits[0]
    return None


# ── the check ───────────────────────────────────────────────────────────────────────────────────

def check_init(init, search_dir=None) -> list:
    """Every `init:` claim, checked against the capture on disk. Returns a list of `Finding`.

    An empty list means the document claimed nothing checkable (no `after:`, no `sha256:`) — the
    caller decides whether to say so. A document with claims but no reachable file gets NOT CHECKED
    findings, not an empty list, because "we could not look" and "there was nothing to look for" are
    different sentences and a reader must not have to guess which one silence meant.
    """
    init = dict(init or {})
    name = init.get("snapshot")
    after = init.get("after")
    digest = init.get("sha256")
    chain = list(init.get("chain") or ())
    findings = []

    if chain and after:
        # Checked in `spec._parse_init` (chain[-1] == after) — restated here as an OK line so the
        # reader sees that the chain was read and not merely tolerated.
        findings.append(Finding(OK, "init.chain",
                                f"{' -> '.join(chain)} -> (this skill); its last step is "
                                f"`after: {after}`"))

    if not (after or digest):
        return findings

    if not name:
        findings.append(Finding(NOT_CHECKED, "init.snapshot",
                                "the document makes a claim about its initiation set but does not "
                                "name one, so nothing was opened"))
        return findings

    path = resolve_snapshot(name, search_dir)
    if path is None:
        detail = (f"no capture named {name!r} is reachable from {search_dir or os.getcwd()!r} "
                  f"(looked for {_STORE_DIRNAME}/{name}.npz up the tree; set BRIDLE_SNAPSHOT_DIR to "
                  f"point elsewhere) — the claim was NOT evaluated, not satisfied")
        if after:
            findings.append(Finding(NOT_CHECKED, "init.after", detail))
        if digest:
            findings.append(Finding(NOT_CHECKED, "init.sha256", detail))
        return findings

    if after:
        findings.append(_check_after(after, path))
    if digest:
        findings.append(_check_digest(digest, path))
    return findings


def _check_after(after, path):
    meta = snapshot_provenance(path)
    recorded, key = None, None
    for k in _PREDECESSOR_KEYS:
        if isinstance(meta.get(k), str) and meta[k].strip():
            recorded, key = meta[k].strip(), k
            break
    if recorded is None:
        return Finding(NOT_CHECKED, "init.after",
                       f"{os.path.basename(path)} records no predecessor (looked for "
                       f"{', '.join(_PREDECESSOR_KEYS)}; it has "
                       f"{', '.join(sorted(meta)) or 'no metadata at all'}) — the document's "
                       f"`after: {after}` could not be confirmed and is NOT confirmed", path)
    if recorded != after:
        return Finding(MISMATCH, "init.after",
                       f"the document says this initiation set is {after}'s handoff; "
                       f"{os.path.basename(path)} records `{key}: {recorded}`. A skill trains on "
                       f"the states its predecessor leaves it, so this is a different task from "
                       f"the one the document describes — the 2026-06-10 swap (move's handoff "
                       f"replaced by grab's) is exactly this line", path)
    return Finding(OK, "init.after",
                   f"{os.path.basename(path)} records `{key}: {recorded}`, as declared", path)


def _check_digest(declared, path):
    actual = snapshot_digest(path)
    if actual != declared:
        return Finding(MISMATCH, "init.sha256",
                       f"{os.path.basename(path)} hashes to {actual} but the document declares "
                       f"{declared}. The content of the initiation set changed underneath a "
                       f"document that claims to describe it: either re-capture was intended (then "
                       f"update `init.sha256:` and say why) or the wrong file is in place", path)
    return Finding(OK, "init.sha256", f"{os.path.basename(path)} matches {actual[:16]}…", path)
