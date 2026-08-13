"""Unit test for bridle.adapters.snapshot_ref — the `init:` claims, checked against the capture.

WHAT IT HAS TO PROVE, AND WHAT IT COST NOT TO HAVE. On 2026-06-10 the bytes behind
`init: {snapshot: descend_init}` were replaced: a `move_to_target` handoff (cube carried above the
destination, mean 8.4 cm away) became a `grab` handoff (cube freshly gripped at the pickup point,
mean 29.6 cm away, 0.000 of starts inside the ~6 cm the re-centring reward was tuned for). Nothing
objected — the plan fingerprint covers the reward's ops, not its inputs — and two full training runs
died on it (32M steps best 0.0625; a Python-reward control at 27.9M best 0.25) against a June
lineage at 0.9375. The file SAID what it was the whole time: its own metadata records `prim: grab`.

So the three legs below are:

  after   the document's declared predecessor vs. the one the capture recorded. Two dialects
          (`after:` and `prim:`), one of them inside a PICKLED metadata blob that must be read
          without unpickling it.
  sha256  the content digest, canonical over the container so a like-for-like re-save is not a
          false alarm, and different for two genuinely different captures.
  three   outcomes, never two: a claim that could not be evaluated (no file, no provenance) is
          NOT_CHECKED and must never read as OK.

The real artifacts are used when they are present (`/home/luca/lego-arm/snapshots/`), because a
guard demonstrated only against fixtures is a guard nobody has seen bite. Synthetic zips cover the
rest so this file still asserts something on a machine without the repo.

Run: PYTHONPATH=. python bridle/tests/test_snapshot_ref.py
"""
import io
import json
import os
import pickle
import sys
import tempfile
import zipfile

from bridle.adapters.snapshot_ref import (
    MISMATCH, NOT_CHECKED, OK, check_init, resolve_snapshot, snapshot_digest, snapshot_provenance,
)

FAILS = []

LEGO_SNAPS = "/home/luca/lego-arm/snapshots"


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def _npy(obj, unicode_json=False):
    """A minimal `.npy` blob for a 0-d array, in the two dialects the capture tools produce."""
    if unicode_json:
        text = json.dumps(obj)
        payload = text.encode("utf-32-le")
        descr = f"<U{len(text)}"
    else:
        payload = pickle.dumps(obj, protocol=2)
        descr = "|O"
    header = ("{'descr': '%s', 'fortran_order': False, 'shape': (), }" % descr).encode("latin-1")
    header += b" " * ((64 - (10 + len(header)) % 64) % 64 - 1) + b"\n"
    return (b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + payload)


def _fake_npz(path, meta=None, unicode_json=False, body=b"data"):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("state.npy", body)
        if meta is not None:
            z.writestr("meta.npy", _npy(meta, unicode_json=unicode_json))


def run_checks():
    tmp = tempfile.mkdtemp()

    # ── the digest is canonical over the CONTAINER, not the bytes ───────────────────────────────
    a, b = os.path.join(tmp, "a.npz"), os.path.join(tmp, "b.npz")
    _fake_npz(a, meta={"after": "move_to_target"}, unicode_json=True)
    with zipfile.ZipFile(b, "w", zipfile.ZIP_STORED) as z:      # different compression, same content
        with zipfile.ZipFile(a) as src:
            for name in reversed(src.namelist()):                # ...and a different member order
                z.writestr(name, src.read(name))
    check("a re-save with identical contents, different compression and member order, has the "
          "same digest", snapshot_digest(a) == snapshot_digest(b))
    check("...and the raw bytes really did differ (so the test is not vacuous)",
          open(a, "rb").read() != open(b, "rb").read())

    c = os.path.join(tmp, "c.npz")
    _fake_npz(c, meta={"after": "move_to_target"}, unicode_json=True, body=b"DIFFERENT")
    check("a capture with different data has a different digest",
          snapshot_digest(c) != snapshot_digest(a))

    # ── both metadata dialects are read, and neither is unpickled ───────────────────────────────
    check("the JSON/unicode dialect is read",
          snapshot_provenance(a).get("after") == "move_to_target")
    d = os.path.join(tmp, "d.npz")
    _fake_npz(d, meta={"source": "coord_chain_handoff", "prim": "grab", "n": 4456})
    check("the pickled-dict dialect is read WITHOUT unpickling",
          snapshot_provenance(d).get("prim") == "grab"
          and snapshot_provenance(d).get("source") == "coord_chain_handoff")
    e = os.path.join(tmp, "e.npz")
    _fake_npz(e, meta=None)
    check("a capture with no metadata yields {} — not a guess", snapshot_provenance(e) == {})

    # ── check_init: the three outcomes ──────────────────────────────────────────────────────────
    os.environ["BRIDLE_SNAPSHOT_DIR"] = tmp
    try:
        good = {"snapshot": "a", "after": "move_to_target", "sha256": snapshot_digest(a)}
        f = {x.label: x for x in check_init(good, search_dir=tmp)}
        check("a matching predecessor and a matching digest are both OK",
              f["init.after"].status == OK and f["init.sha256"].status == OK)

        swapped = dict(good, snapshot="d")               # the 2026-06-10 swap, in miniature
        f = {x.label: x for x in check_init(swapped, search_dir=tmp)}
        check("a capture recording a DIFFERENT predecessor is a MISMATCH",
              f["init.after"].status == MISMATCH and "grab" in f["init.after"].detail)
        check("...and it is a refusal", f["init.after"].refused)
        check("the digest of that other capture also refuses",
              f["init.sha256"].status == MISMATCH)

        f = {x.label: x for x in check_init({"snapshot": "e", "after": "move_to_target"},
                                            search_dir=tmp)}
        check("a capture that records NO provenance is NOT_CHECKED, never OK",
              f["init.after"].status == NOT_CHECKED and not f["init.after"].refused)

        f = {x.label: x for x in check_init({"snapshot": "nope", "after": "move_to_target",
                                             "sha256": "0" * 64}, search_dir=tmp)}
        check("an unreachable capture is NOT_CHECKED on every claim, never OK",
              f["init.after"].status == NOT_CHECKED and f["init.sha256"].status == NOT_CHECKED)

        check("a document that claims nothing is checked as nothing (empty, not OK)",
              check_init({"snapshot": "a"}, search_dir=tmp) == [])
        check("...and so is an absent init block", check_init(None) == [])

        chain = {"snapshot": "a", "after": "move_to_target",
                 "chain": ["reach", "grab", "lift", "move_to_target"]}
        labels = [x.label for x in check_init(chain, search_dir=tmp)]
        check("a declared chain is reported alongside the predecessor it ends at",
              "init.chain" in labels and "init.after" in labels)
    finally:
        os.environ.pop("BRIDLE_SNAPSHOT_DIR", None)

    # ── the real artifacts, which are the whole point ───────────────────────────────────────────
    broken = os.path.join(LEGO_SNAPS, "descend_init.npz")
    corrected = os.path.join(LEGO_SNAPS, "descend_init_move.npz")
    if not (os.path.isfile(broken) and os.path.isfile(corrected)):
        print(f"  NOTE  {LEGO_SNAPS} not present — the synthetic legs above are all this "
              f"interpreter can assert")
        return
    check("the corrected capture records `after: move_to_target`",
          snapshot_provenance(corrected).get("after") == "move_to_target")
    check("the capture swapped in on 2026-06-10 records `prim: grab` — it said so all along",
          snapshot_provenance(broken).get("prim") == "grab")
    check("the two captures do not share a digest",
          snapshot_digest(broken) != snapshot_digest(corrected))

    declared = {"snapshot": "descend_init_move", "after": "move_to_target",
                "sha256": snapshot_digest(corrected)}
    f = {x.label: x for x in check_init(declared, search_dir=os.path.dirname(LEGO_SNAPS))}
    check("descend's real document passes both claims",
          f["init.after"].status == OK and f["init.sha256"].status == OK)
    f = {x.label: x for x in check_init(dict(declared, snapshot="descend_init"),
                                        search_dir=os.path.dirname(LEGO_SNAPS))}
    check("pointing that same document at the broken capture refuses on BOTH claims",
          f["init.after"].status == MISMATCH and f["init.sha256"].status == MISMATCH)


def test_bridle():
    FAILS.clear()
    run_checks()
    assert not FAILS, f"{len(FAILS)} check(s) failed: {FAILS}"


def main():
    run_checks()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
