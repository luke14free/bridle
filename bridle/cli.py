"""`bridle` — the out-of-the-box entry point.

    bridle tui --model local:qwen3-32b        # agent TUI + simulator window
    bridle skills                             # LIST trained apps: what runs on this rig (plural)
    bridle plan descend_to_target             # why a skill needs adapting or rebuilding
    bridle skill vocab                        # AUTHOR a new one: the reward vocabulary (singular)
    bridle skill check primitives/x/skill.yaml   # schema -> compile, before a GPU-second is spent
    bridle skill diagnose runs/<exp>/reward_terms.jsonl   # ...and what the GPU said afterwards

The TUI and the viewer come up together: the terminal is where you talk to the agent, the browser
window is where you watch the robot. Neither replaces the other.

EXECUTORS. bridle knows which skills are valid; it does not know how to drive YOUR simulator. Without
`--executor mod:func` the TUI runs in DRY mode, where a skill call is reported but nothing moves.
That is the honest default: silently doing nothing while looking like it worked is the failure this
whole project exists to prevent.
"""
import argparse
import importlib
import os
import sys
import warnings

from bridle.agent import AgentSession
from bridle.llm import PRESETS, from_spec
from bridle.resolve import RUN
from bridle.rig import Rig
from bridle.store import Store


def _load(path):
    """`package.module:function` -> the callable."""
    mod, _, fn = path.partition(":")
    if not fn:
        raise SystemExit(f"--executor wants 'module:function', got {path!r}")
    return getattr(importlib.import_module(mod), fn)


def dry_executor(name, args):
    return True, f"[dry] would run {name}({args}) — no executor configured, nothing moved"


def cmd_skills(a):
    store, rig = Store(a.store), Rig.so101(cameras=tuple(a.cameras.split(",")))
    rows = []
    for app in store.apps():
        try:
            p = store.plan(app, rig)
            rows.append((p.action, app.name, p.reason if p.action != "blocked"
                         else ", ".join(p.blockers)))
        except Exception as e:
            rows.append(("error", app.name, f"{type(e).__name__}: {e}"))
    order = {RUN: 0, "adapt": 1, "retrain": 2, "blocked": 3, "error": 4}
    print(f"rig: {rig.describe()}\n")
    for action, name, why in sorted(rows, key=lambda r: (order.get(r[0], 9), r[1])):
        print(f"  {action:8s} {name:26s} {why[:70]}")
    print(f"\n{sum(1 for r in rows if r[0] == RUN)}/{len(rows)} skills run on this rig")
    return 0


def cmd_plan(a):
    store, rig = Store(a.store), Rig.so101(cameras=tuple(a.cameras.split(",")))
    plan = store.plan(store.get(a.app), rig)
    print(plan.explain())
    return 0


def cmd_tui(a):
    from bridle import tui as tui_mod
    from bridle.ui import Viewer

    store = Store(a.store)
    rig = Rig.so101(cameras=tuple(a.cameras.split(",")))
    executor = _load(a.executor) if a.executor else dry_executor
    provider = from_spec(a.model, base_url=a.base_url)
    session = AgentSession(provider, store, rig, executor)

    viewer = None
    if not a.no_viewer:
        viewer = Viewer(store, rig, port=a.viewer_port).start()
    if not a.executor:
        session.events.put(type("E", (), {"kind": "status", "text":
            "DRY MODE — no --executor, so skills are reported but nothing moves", "data": {},
            "t": 0})())
    tui_mod.run(session, viewer_url=(viewer.url if viewer else None),
                model_label=a.model, models=(a.models.split(",") if a.models else [a.model]))
    if viewer:
        viewer.stop()
    return 0


def cmd_relaunch(a):
    """Reproduce a deployed lineage with named overrides, gated by preflight."""
    # bridle declares `dependencies = []`; Store._load_text already does the lazy `import yaml`
    # with a json fallback, so use it rather than adding a third YAML-reading path to a codebase
    # whose thesis is that two records of one fact is the disease.
    from bridle.store import Store
    from bridle.adapters.preflight import collect, readable_env
    from bridle.lineage import NAMESPACES, EmptyDiff, UnknownOverride, format_diff
    from bridle.preflight import DYNAMIC, evaluate, format_effective, format_failures
    from bridle.relaunch import (
        WarmStartRefused, build_plan, install_and_start, nonempty_run_dir_message, run_dir_for,
        seed_warm_start, systemd_unit, vacuous_preflight_message,
    )

    # Validated UP FRONT, before any manifest/preflight work: a bogus --warm-start path used to
    # reach `shutil.copyfile` unvalidated at the very end of this command (raising a raw traceback),
    # or — when the preflight for this app has no DYNAMIC asserts — never get opened at all and
    # silently seed a run that then trains from a checkpoint that never existed. Fail clean, first.
    if a.warm_start and not os.path.isfile(a.warm_start):
        print(f"refused: --warm-start {a.warm_start} does not exist")
        return 1

    store = Store(a.store)
    manifest_path = os.path.join(os.path.expanduser(a.store), f"{a.app}.yaml")
    try:
        with open(manifest_path) as f:
            manifest = store._load_text(f.read())
    except FileNotFoundError:
        print(f"refused: no app manifest at {manifest_path}")
        return 1
    if not manifest:
        print(f"refused: {manifest_path} parsed to nothing (empty or malformed)")
        return 1

    bad_sets = [kv for kv in (a.set or []) if "=" not in kv]
    if bad_sets:
        print(f"refused: --set wants K=V, got: {', '.join(bad_sets)!r}")
        return 1
    overrides = dict(kv.split("=", 1) for kv in a.set or [])

    # I6: relaunch never told the operator when the inherited env came from a hand-transcribed
    # record instead of a captured process. capture_training() tags a real capture "source:
    # captured"; a manifest built by hand before that existed (or backfilled after the fact, like
    # place_coord_v3) says "source: backfilled" and was never verified against a live process.
    if (manifest.get("training") or {}).get("source") == "backfilled":
        print(f"warning: {a.app}'s training record is BACKFILLED — transcribed by hand from prose/"
              "journal notes, not captured from a live process, and never independently verified. "
              "Treat the inherited env with extra scrutiny.")

    # C1(a): resolve the preflight path from the MANIFEST, not a guess off the app name. Apps are
    # not named after the primitive they train (`place_coord_v3` trains `descend_to_target`), so
    # `primitives/<app>/preflight.yaml` found nothing for any real app and silently ran with zero
    # asserts. Priority: an explicit --preflight override, then the manifest's own `preflight:` key,
    # then the app-name guess as a last resort (for an app that has genuinely never had one authored).
    manifest_pf = manifest.get("preflight")
    if a.preflight:
        pf_path, pf_source = a.preflight, "--preflight"
    elif manifest_pf:
        pf_path = manifest_pf if os.path.isabs(manifest_pf) else os.path.join(a.cwd, manifest_pf)
        pf_source = f"{a.app}.yaml `preflight:`"
    else:
        pf_path = os.path.join(a.cwd, "primitives", a.app, "preflight.yaml")
        pf_source = "guessed from the app name (manifest has no `preflight:` key)"
    if os.path.isfile(pf_path):
        doc = store._load_text(open(pf_path).read()) or {}
    else:
        doc = {}
        print(f"warning: no preflight.yaml at {pf_path} ({pf_source}) — running kind asserts only")

    module = a.module or doc.get("module", "")
    try:
        plan = build_plan(manifest, doc, overrides, a.exp, readable_env(module))
    except (EmptyDiff, UnknownOverride, ValueError) as e:
        # These three already carry an actionable message (lineage.py / relaunch.py write them for
        # a human to read). The callers of this command include an automated agent, for which a raw
        # traceback is not a contract — so refuse cleanly instead of letting it propagate, the same
        # way a preflight failure below already returns 1 with a clean message.
        print(f"refused: {e}")
        return 1

    print(f"\nrelaunch {plan.app} as {plan.exp}\n\nchange:")
    print(format_diff(plan.changes))
    print("\neffective asserts:")
    print(format_effective(plan.asserts))

    # C1(b): zero asserts must NEVER render as "preflight OK" — to an automated caller that is
    # indistinguishable from six asserts having actually been checked and passed. Refuse unless the
    # operator explicitly says a vacuous preflight is acceptable.
    if not plan.asserts:
        msg = vacuous_preflight_message(plan.asserts, pf_path, pf_source)
        if not a.allow_vacuous_preflight:
            print(f"\n{msg}")
            return 1
        print(f"\n{msg} — proceeding anyway (--allow-vacuous-preflight)")

    # I3: refuse to write into an existing lineage's run directory. Every launcher
    # (train_coord_prim.sh, teacher_train.sh, ...) resumes from whichever ckpt sorts highest by
    # number in runs/<exp>/ — starting a DIFFERENT relaunch into that same directory is checkpoint
    # contamination, not a fresh run, and is exactly the class of mistake this command exists to end.
    run_dir = run_dir_for(module, plan.exp, a.cwd)
    if run_dir is None:
        print(f"refused: could not derive the run directory from module {module!r} (expected "
              "'primitives.<name>...') — refusing rather than silently skipping the "
              "non-empty-run-directory check")
        return 1
    nonempty_msg = nonempty_run_dir_message(run_dir, a.resume)
    if nonempty_msg:
        print(nonempty_msg)
        return 1

    # Delete every namespaced variable that survived from the calling shell but is NOT part of this
    # plan, before updating with plan.env. Preflight's STATIC tier imports the target module
    # in-process and reads os.environ directly, but systemd_unit() below only emits `Environment=`
    # lines for plan.env — so a stray namespaced var left in the process environment would let
    # preflight validate a configuration that is not the one the launched unit actually gets.
    # Preflight is worthless if it measures a different config than the one that runs.
    for k in [k for k in os.environ if k.startswith(NAMESPACES) and k not in plan.env]:
        del os.environ[k]
    os.environ.update(plan.env)
    values = collect(plan.asserts, doc.get("env_id", ""), module, a.warm_start,
                     doc.get("eval_envs", 64), doc.get("eval_steps", 64),
                     from_scratch=a.from_scratch)
    failures = evaluate(plan.asserts, values, from_scratch=a.from_scratch)
    if failures:
        print("\n" + format_failures(failures))
        return 1
    print("\npreflight OK")

    unit_text = systemd_unit(plan.exp, plan.launcher, plan.env, a.cwd)
    if a.dry_run:
        print("\n--dry-run: the systemd unit that would be written:\n")
        print(unit_text)
        print("(dry run — omit --dry-run to start it)")
        return 0

    # I2: seed the warm-start ckpt so the launcher's own resume logic (highest-numbered ckpt in
    # runs/<exp>/) actually picks it up. Without this, --warm-start's help text ("ckpt to seed and
    # to preflight against") was half true: preflight measured the warm policy, but the unit still
    # trained from scratch. F2: `seed_warm_start` refuses rather than clobbering a ckpt_1.pt that
    # may hold real training progress; existence of `a.warm_start` itself was already checked above.
    if a.warm_start:
        try:
            seeded = seed_warm_start(a.warm_start, run_dir, force=a.force_warm_start)
        except WarmStartRefused as e:
            print(str(e))
            return 1
        print(f"seeded {seeded} from --warm-start {a.warm_start}")

    handle = install_and_start(f"bridle-{plan.exp}", unit_text)
    print(f"launched: systemd --user {handle}")
    return 0


def cmd_lineage(a):
    """Assert every record that claims to describe the deployed environment agrees with it.

    THE RULE IS AGREEMENT BETWEEN RECORDS, NOT TIDINESS WITHIN ONE. Repeated assignment is not
    reported: a drop-in overriding the base unit is the systemd mechanism working as designed, and
    playground-coord.service.d/deploy-widegrab.conf assigns GRAB_COORD_REFRESH_R and
    DINO_GRAB_CORNER_COORD twice on purpose, as a chronological log where a later line reverts an
    earlier decision with the measurement preserved inline. Flagging either would report correct
    code as broken, and a checker that fires on correct code gets ignored.

    What IS a violation (live on 2026-08-12): scripts/_pgenv.sh says in its header that it mirrors
    playground-coord.service, and carries the SHADOWED COORD_CKPT_grab — so every offline test run
    through it loaded a different grab policy than the live service.
    """
    import glob
    import subprocess

    from bridle.lineage import check_ckpt_pins, compare_records, resolve_env
    from bridle.store import Store

    store = Store(a.store)
    bad = []

    # 1. Resolve the service environment the way systemd does: `systemctl cat` already emits the
    #    base unit followed by its drop-ins in application order, so a single ordered pass is the
    #    same resolution systemd performs.
    #
    #    An unperformed check is a violation, not a pass: if this fails for ANY reason (systemctl
    #    missing from PATH, the service stopped/renamed/uninstalled), `effective` stays {} and
    #    steps 2-3 below cannot compare against a live env. Printing "0 violation(s)" in that case
    #    would be indistinguishable from a genuine clean run, so every such path prints a distinct
    #    UNVERIFIED line and counts toward the nonzero exit — "cannot verify" must never render as
    #    "verified".
    try:
        unit = subprocess.run(["systemctl", "--user", "cat", "playground-coord.service"],
                              capture_output=True, text=True)
    except FileNotFoundError as e:
        print(f"UNVERIFIED systemctl not found on PATH ({e}); cannot resolve the live env")
        bad.append("systemctl")
        effective = {}
    else:
        if unit.returncode != 0:
            print("UNVERIFIED playground-coord.service: `systemctl --user cat` exited "
                  f"{unit.returncode} ({unit.stderr.strip() or 'no stderr'}); cannot resolve the "
                  "live env")
            bad.append("playground-coord.service")
            effective = {}
        else:
            effective = resolve_env([("playground-coord.service", unit.stdout.splitlines())])

    # 2. Every record that claims to mirror it must agree. This must run — and be reported as
    #    UNVERIFIED if it can't — independent of whether step 1 succeeded: a missing _pgenv.sh is
    #    its own unperformed check even when the live env resolved fine.
    pgenv = os.path.join(a.cwd, "scripts/_pgenv.sh")
    if not os.path.isfile(pgenv):
        print(f"UNVERIFIED scripts/_pgenv.sh not found at {pgenv}; cannot check it against the "
              "live env")
        bad.append("scripts/_pgenv.sh")
    elif not effective:
        print("UNVERIFIED scripts/_pgenv.sh: live env did not resolve (see above); comparison "
              "skipped")
        bad.append("scripts/_pgenv.sh")
    else:
        claimed = resolve_env([(pgenv, open(pgenv).read().splitlines())])
        for m in compare_records(effective, claimed, "scripts/_pgenv.sh", prefix="COORD_CKPT_"):
            eff_repr = m.effective if m.effective is not None else "(not set at all by the live env)"
            print(f"MISMATCH   {m.record}: {m.key} claims {m.claimed}, live env resolves to "
                  f"{eff_repr}")
            bad.append(m.key)

    # 3. Every app's ckpt exists, and where the live env pins that primitive, they agree. One
    #    unparsable manifest, or one with a malformed `ckpt` field, must not abort the scan of
    #    every other manifest — it is reported as its own violation and the scan continues.
    manifest_ckpts = []  # [(app_name, absolute_ckpt_path), ...], fed to check_ckpt_pins below
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(a.store), "*.yaml"))):
        try:
            m = store._load_text(open(path).read()) or {}
            ck = m.get("ckpt")
            if not ck:
                continue
            if not isinstance(ck, str):
                raise TypeError(f"ckpt must be a string, got {type(ck).__name__}: {ck!r}")
            full = ck if os.path.isabs(ck) else os.path.join(a.cwd, ck)
        except Exception as e:
            print(f"UNVERIFIED {path}: {type(e).__name__}: {e}; manifest could not be checked")
            bad.append(path)
            continue
        if not os.path.isfile(full):
            print(f"MISSING    {m.get('name')}: ckpt does not exist: {ck}")
            bad.append(m.get("name"))
        manifest_ckpts.append((m.get("name"), full))

    # 4. Where the live env pins a primitive's checkpoint (COORD_CKPT_<prim>), some manifest must
    #    agree — this is the check step 3's own comment has always claimed and, until now, never
    #    ran (see bridle.lineage.check_ckpt_pins for the live 2026-08-12/13 grab defect this catches).
    for v in check_ckpt_pins(effective, manifest_ckpts):
        label = "MISMATCH" if v.kind == "mismatch" else "UNREGISTERED"
        print(f"{label:<10} {v.key}: live env resolves to {v.effective} — {v.note}")
        bad.append(f"{v.key}:{v.kind}")

    print(f"\n{len(bad)} violation(s)")
    return 1 if bad else 0


# ── bridle skill: the three things the author of a skill.yaml needs ─────────────────────────────
# THE AUTHOR IS A LOCAL 27-30B MODEL, so these three outputs are its entire API surface: the
# vocabulary it writes against, the refusal when it gets one wrong, and the resolved plan showing
# what it actually asked for. Measured stakes: one-shot authoring is 58.3% +/- 47.3%, refinement
# against typed feedback is 97.6%, and stripping the typed content of that feedback collapses it to
# 11.5%. A refusal that names no path, and a plan that hides the defaults the chassis supplied, are
# both the stripped condition.

#: `--terminate-on-success` is tri-state and `unknown` is a real answer, not a missing one: it
#: selects the CONSERVATIVE branch of compile's horizon-integrated warning rather than asserting
#: something about the env. Guessing here would put a number in a warning that nobody measured.
_TERMINATION = {"yes": True, "no": False, "unknown": None}

#: The one fully worked skill document in either repo, named by `bridle skill vocab` so the payload
#: it prints is not the only thing an author has ever seen. A PATH and not an inlined copy: the
#: prose payload is already near its token ceiling, and this file is ~400 lines of annotated YAML
#: whose annotations (what each `why` was ported from, where the two files legitimately diverge) are
#: most of its value. Not read by this process — `skill check` reads it when asked to.
_WORKED_EXAMPLE = "/home/luca/lego-arm/primitives/descend_to_target/skill.yaml"


def _skill_document(a):
    """`(doc, None)` or `(None, refusal text)`.

    YAML is read through `Store._load_text` — the lazy `import yaml` with a json fallback — because
    `bridle` declares `dependencies = []` and a second YAML-reading path in this repo would be two
    records of one fact. `Store(a.store)` is constructed only for that loader; its root is the app
    store and is not otherwise touched here.
    """
    if not os.path.isfile(a.file):
        return None, f"no skill document at {a.file}"
    try:
        with open(a.file) as f:
            text = f.read()
    except OSError as e:
        return None, f"{a.file} could not be read: {e}"
    try:
        doc = Store(a.store)._load_text(text)
    except Exception as e:
        return None, (f"{a.file} does not parse as YAML (or as JSON, when PyYAML is absent): "
                      f"{type(e).__name__}: {e}")
    if doc is None:
        return None, f"{a.file} parsed to nothing — the file is empty, or entirely comments"
    if not isinstance(doc, dict):
        return None, (f"{a.file} parsed to a {type(doc).__name__}, but a skill document is a "
                      f"mapping of top-level fields (name, kind, contract, env_id, scene, reward, "
                      f"success)")
    return doc, None


def _skill_refusal(a, stage, error):
    """One refusal, in the form the author can act on without reading any Python: the stage that
    said no, the dotted path, the legal set (carried inside the message by `SpecError`/`CompileError`
    themselves), and the nearest legal spelling on its own line."""
    from bridle.skill.report import wrap

    lines = [f"skill {a.skill_cmd} FAILED — {stage}", "", wrap(str(error), "  "), "",
             f"  path: {error.path}"]
    if getattr(error, "suggestion", None) is not None:
        lines.append(f"  fix:  the nearest legal spelling is {error.suggestion!r}")
    lines += ["", wrap(
        "This is the FIRST refusal only. The checks are ordered and a later one may be reading a "
        "value this one rejected, so fix this path and run the command again rather than guessing "
        "at what else might be wrong.", "  ")]
    return "\n".join(lines)


def _skill_diagnose(a):
    """`bridle skill diagnose <reward_terms.jsonl>` — tier 3, read back after the GPU has spoken.

    THE VERB EXISTS BECAUSE THERE IS NOW SOMETHING TO READ. It was deliberately not added while
    `diagnose` had no producer: a verb with no input is worse than no verb, because it advertises a
    capability whose only possible output is "nothing to say". The input is the JSONL
    `bridle.adapters.skill_telemetry` appends one record to per emission —
    `<primitive>/runs/<exp>/reward_terms.jsonl` under `scripts/train_from_skill.py --train`.

    THE LAST RECORD BY DEFAULT, because a window describes recent behaviour and the freshest one is
    the question a reader has. `--index` reaches back (negative indices count from the end) and
    `--list` prints the whole series' headline numbers, which is how a trend gets read without a
    dashboard.

    The record carries the diagnostics the run itself emitted; they are RE-DERIVED here from the
    stored statistics rather than reprinted, so this command and the training loop cannot come to
    disagree about what the same numbers mean. The stored copy is what `--stored` prints, for
    exactly the case where they DO differ (the rules moved since the run).
    """
    import json

    from bridle.skill.diagnose import diagnose, format_diagnostics

    if not os.path.isfile(a.file):
        print(f"skill diagnose FAILED — no diagnostics file at {a.file}\n\n"
              f"  This reads the JSONL a training run writes, one record per emission:\n"
              f"    <primitive>/runs/<exp>/reward_terms.jsonl\n"
              f"  produced by `scripts/train_from_skill.py <skill.yaml> --train` (lego-arm). A run "
              f"started with --no-diagnostics writes none.")
        return 1
    records = []
    with open(a.file) as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError as e:
                print(f"  note: line {n} of {a.file} is not JSON ({e}) — skipped")
    if not records:
        print(f"skill diagnose — {a.file} holds no records yet. The first emission lands after "
              f"`--diag-every` control steps of the training env (default 400 = one per eval).")
        return 1

    if a.list:
        print(f"{len(records)} emission(s) in {a.file}")
        print(f"  {'#':>4}  {'steps':>9}  {'episodes':>9}  {'success':>8}  {'ep_len':>7}  tags")
        for i, r in enumerate(records):
            rate = "n/a" if r.get("success_rate") is None else f"{r['success_rate']:.4f}"
            ln = "n/a" if r.get("ep_len") is None else f"{r['ep_len']:.1f}"
            tags = ",".join(sorted({d["tag"] for d in r.get("diagnostics") or []})) or "-"
            print(f"  {i:>4}  {r.get('total_steps', 0):>9}  "
                  f"{(r.get('window') or {}).get('episodes', 0):>9}  {rate:>8}  {ln:>7}  {tags}")
        return 0

    try:
        r = records[a.index]
    except IndexError:
        print(f"skill diagnose FAILED — --index {a.index} is outside the {len(records)} record(s) "
              f"in {a.file}; `--list` prints them all")
        return 1

    window = r.get("window") or {}
    print(f"skill diagnose — {a.file}")
    print(f"  record {a.index} of {len(records)}   {r.get('label') or ''}"
          + (f"   plan@{r['plan']}" if r.get("plan") else ""))
    print(f"  window: {window.get('steps')} control steps, {window.get('env_steps')} env-steps, "
          f"{window.get('episodes')} episodes ended, at {r.get('total_steps')} total steps")
    rate, ep_len = r.get("success_rate"), r.get("ep_len")
    print(f"  success rate: " + ("NOT MEASURED — no episode ended in this window, so the "
                                 "composition checks could not run" if rate is None else
                                 f"{rate:.4f}   mean episode length {ep_len:.1f}"))
    stats = r.get("term_stats") or {}
    print(f"\n  {'row':<52}{'min':>12}{'mean':>12}{'max':>12}")
    for name, s in stats.items():
        print(f"  {name:<52}{s.get('min', float('nan')):>12.4f}"
              f"{s.get('mean', float('nan')):>12.4f}{s.get('max', float('nan')):>12.4f}")
    print()
    if a.stored:
        stored = r.get("diagnostics") or []
        print(f"  AS EMITTED BY THE RUN ITSELF ({len(stored)}):")
        for d in stored:
            print(f"    [{d['tag']}] {d['row']}: {d['message']}")
        return 0
    print(format_diagnostics(diagnose(stats, rate, ep_len, horizon=r.get("horizon"))))
    return 0


def _skill_env_check(a, spec):
    """Does `env_id:` name an environment that exists? `(status, detail)`.

    IT USED TO NAME ANYTHING. `env_id` was an unresolved free string, so a document with
    `env_id: ThisEnvDoesNotExist-v9` reported `skill check OK, exit 0` and a stamped plan
    fingerprint — the same defect class as the `success:` criterion that reached the GPU unchecked
    (2026-08-13, review I2). The resolution itself lives in `bridle.adapters.env_ref` because it
    needs the simulator's registry and `bridle.skill` is stdlib-only.

    THE THIRD OUTCOME IS THE POINT. With no simulator importable the answer is `NOT CHECKED`, which
    is printed as such and does NOT fail the run — this checker must never render "could not
    verify" as "verified", and it must not refuse a document merely because the machine reading it
    has no GPU stack installed.
    """
    if getattr(a, "no_env_check", False):
        return "not_checked", "--no-env-check was passed, so `env_id` was not resolved"
    try:
        from bridle.adapters.env_ref import check_env_ref
    except Exception as e:                                        # noqa: BLE001
        return "not_checked", f"bridle.adapters.env_ref unavailable ({type(e).__name__}: {e})"
    return check_env_ref(spec.env_id, search_dir=os.path.dirname(os.path.abspath(a.file)))


def _skill_init_check(a, spec):
    """Do the document's `init:` claims match the capture on disk? `(findings, refused)`.

    THE CHEAPEST REAL CHECK IN THIS COMMAND. It opens one file, reads its recorded provenance and
    hashes its contents — no simulator, no GPU, no import of the primitive. The bug it exists to
    refuse cost ~a day of GPU across two dead runs: the bytes behind `snapshot: descend_init` were
    replaced on 2026-06-10 by a different primitive's handoff, and the file SAID SO in its own
    metadata while the document did not, for two months (`bridle.adapters.snapshot_ref`).

    SAME THREE OUTCOMES AS `env_id` ABOVE, for the same reason. A MISMATCH refuses. A claim that
    could not be evaluated — no capture reachable from this machine, no provenance recorded in the
    file, no adapter importable — prints NOT CHECKED and does not fail: this checker must never
    render "could not verify" as "verified", and must not refuse a document merely because the
    machine reading it does not hold the captures.
    """
    if not spec.init:
        return [], False
    try:
        from bridle.adapters.snapshot_ref import NOT_CHECKED, Finding, check_init
    except Exception as e:                                        # noqa: BLE001
        return [type("_F", (), {"line": lambda self: (
            f"init: NOT CHECKED — bridle.adapters.snapshot_ref unavailable "
            f"({type(e).__name__}: {e})"), "refused": False})()], False
    findings = check_init(spec.init, search_dir=os.path.dirname(os.path.abspath(a.file)))
    if not findings:
        name = spec.init.get("snapshot")
        findings = [Finding(NOT_CHECKED, "init",
                            f"`init:` names {name!r} and claims nothing about it. Declare "
                            f"`after:` (the predecessor whose handoff this is) and/or `sha256:` "
                            f"(its content digest) and this line becomes a check."
                            if name else "`init:` is empty")]
    return findings, any(f.refused for f in findings)


def _skill_contract_check(spec):
    """Does `contract:` name a `bridle.contract.Contract` factory? Reported, never refused.

    ADVISORY ON PURPOSE, and the asymmetry with `env_id` above is deliberate rather than an
    oversight. `Contract` exposes a handful of named factories (`stack`, `grab`, `for_prim`) and a
    skill may legitimately name a contract this repo has no factory for — several existing documents
    and fixtures do. A refusal would therefore reject working documents to catch a typo, so the
    typo is printed instead and the reader decides.
    """
    try:
        from bridle.contract import Contract
    except Exception as e:                                        # noqa: BLE001
        return f"NOT CHECKED — bridle.contract unavailable ({type(e).__name__}: {e})"
    factory = getattr(Contract, spec.contract, None)
    if callable(factory):
        return f"{spec.contract} — resolves to Contract.{spec.contract}()"
    named = sorted(n for n in dir(Contract)
                   if not n.startswith("_") and callable(getattr(Contract, n, None)))
    return (f"{spec.contract} — NOT a Contract factory (advisory, not a refusal). "
            f"Named factories: {', '.join(named)}")


def cmd_skill(a):
    """`vocab` | `check <file>` | `compile <file>` | `diagnose <file>` — the three feedback tiers.

    Tiers 1 and 2 (`check`, `compile`) need no simulator, so a clean `check` says the document is
    well-formed and internally consistent, NOT that the reward trains. Tier 3 is what the GPU said
    afterwards: `diagnose` reads the per-term statistics a training run logged and re-derives the
    typed findings from them.
    """
    from bridle.skill.vocab import vocab_document

    if a.skill_cmd == "diagnose":
        return _skill_diagnose(a)

    if a.skill_cmd == "vocab":
        if getattr(a, "json_schema", False):
            # `spec.json_schema()` had NO consumer anywhere in the branch (2026-08-13 review, C2)
            # while its own docstring called it "the machine-readable half of what a 27-30B author
            # is handed" — a half nothing handed over. This flag is that consumer, and it is the
            # payload for a constrained-decoding harness rather than for a prose prompt.
            import json

            from bridle.skill.spec import json_schema
            print(json.dumps(json_schema(), indent=2))
            return 0
        print(vocab_document())
        # `vocab_document`'s budget always assumed "a task description and one worked example"
        # ALONGSIDE it, and nothing in this branch supplied one, so the payload named no document
        # an author could read. It is NAMED rather than inlined: the prose payload is already at
        # ~7.4k of its 8k estimated-token ceiling, and the example is 400 lines of annotated YAML.
        print(f"\n<!-- One fully worked, deployed example, annotated row by row: {_WORKED_EXAMPLE}\n"
              f"     Its plan fingerprint is printed by `bridle skill compile`. -->")
        return 0

    from bridle.skill.compile import CompileError, compile_spec
    from bridle.skill.report import format_plan, format_warnings, wrap
    from bridle.skill.spec import SpecError, parse_spec

    doc, refusal = _skill_document(a)
    if refusal is not None:
        print(f"skill {a.skill_cmd} FAILED — the document could not be read\n\n  {refusal}")
        return 1
    try:
        spec = parse_spec(doc)
    except SpecError as e:
        print(_skill_refusal(a, "tier 1, schema (parse_spec)", e))
        return 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # re-printed in full by `report.format_warnings`
            plan = compile_spec(spec, horizon=a.horizon,
                                terminate_on_success=_TERMINATION[a.terminate_on_success])
    except CompileError as e:                   # FloodingError included — it is a CompileError
        print(_skill_refusal(a, "tier 2, compile (compile_spec)", e))
        return 1

    if a.skill_cmd == "compile":
        print(format_plan(doc, spec, plan, horizon=a.horizon,
                          terminate_on_success=a.terminate_on_success))
        return 0

    env_status, env_detail = _skill_env_check(a, spec)
    contract_line = _skill_contract_check(spec)
    init_findings, init_refused = _skill_init_check(a, spec)
    if init_refused:
        # Printed BEFORE the env tier and returning immediately: this refusal is about the states
        # the run would start from, and a document whose initiation set is a different task from the
        # one it describes is not made acceptable by its env resolving.
        body = "\n".join(wrap(f.line(), "  ") for f in init_findings)
        print("skill check FAILED — tier 1.5, the initiation set is not the one the document "
              "describes\n\n" + body + "\n\n" + wrap(
                  f"The reward compiled — plan@{plan.fingerprint()} — and it is not the reward that "
                  f"is wrong. A skill trains on the states its predecessor leaves it, so an "
                  f"initiation set swapped underneath a document silently changes the task: on "
                  f"2026-06-10 exactly this swap (move_to_target's handoff replaced by grab's, mean "
                  f"8.4 cm from target replaced by mean 29.6 cm) cost two full training runs and "
                  f"~a day of GPU with nothing objecting. Point `init.snapshot:` at the right "
                  f"capture, or — if the capture was deliberately regenerated — update the claim "
                  f"and say why.", "  "))
        return 1
    if env_status == "unknown":
        tail = (f"The reward compiled — plan@{plan.fingerprint()} — but a document that names an "
                f"env nothing can build trains nothing. Fix `env_id:`, or point the check at an "
                f"interpreter that can import the module registering it. `--no-env-check` skips "
                f"this tier and says so.")
        print("skill check FAILED — tier 1.5, the environment does not exist\n\n"
              + wrap(env_detail, "  ") + "\n\n" + wrap(tail, "  "))
        return 1

    print(f"skill check OK — {spec.name} ({spec.kind} chassis) — plan@{plan.fingerprint()}, "
          f"{len(plan.ops)} ops")
    print(f"  env_id    {spec.env_id} — " + ("resolved: " + env_detail if env_status == "ok"
                                             else "NOT CHECKED: " + env_detail))
    print(f"  contract  {contract_line}")
    for f in init_findings:
        print(f"  {f.line()}")
    print(format_warnings(plan))
    print("\n" + wrap(
        "Passed tiers 1-2 of 3 (schema -> compile -> preflight). No simulator was started, so this "
        "says the document is well-formed and internally consistent — not that the reward trains. "
        "`bridle skill compile` prints the resolved plan, including every default the chassis "
        "supplied on your behalf.", "  "))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bridle", description=__doc__.split("\n")[0])
    ap.add_argument("--store", default=os.environ.get(
        "BRIDLE_STORE", "/home/luca/lego-arm/composer/store/apps"))
    ap.add_argument("--cameras", default="base", help="comma-separated camera names on your rig")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tui", help="agent TUI + simulator window")
    t.add_argument("--model", default="local:qwen3-32b",
                   help=f"<preset>:<model>; presets: {', '.join(sorted(PRESETS))}")
    t.add_argument("--models", default="", help="comma-separated specs to cycle with ^N")
    t.add_argument("--base-url", default=None)
    t.add_argument("--executor", default=None, help="module:function that actually drives a skill")
    t.add_argument("--viewer-port", type=int, default=8799)
    t.add_argument("--no-viewer", action="store_true")
    t.set_defaults(fn=cmd_tui)

    # `skills` and `skill` differ by one character and do unrelated things — one lists what is
    # already trained, the other validates a document for something that is not. argparse resolves
    # them exactly so neither can shadow the other, and `skill` is the mandated verb, so the cost to
    # remove is the READER's: the intended reader is a 27-30B model, and a wrong guess between two
    # commands separated by a trailing `s` costs a whole round trip. Each help line therefore says
    # what its command does in full and names the other one.
    s = sub.add_parser("skills", help="LIST the already-trained apps in the store and whether each "
                                      "runs on this rig (plural; see `skill` to author a new one)")
    s.set_defaults(fn=cmd_skills)

    p = sub.add_parser("plan", help="why a skill needs adapting or rebuilding")
    p.add_argument("app")
    p.set_defaults(fn=cmd_plan)

    r = sub.add_parser("relaunch", help="retrain a deployed lineage with one variable changed")
    r.add_argument("app")
    r.add_argument("--set", action="append", metavar="K=V",
                   help="override one variable; repeatable. An empty diff is refused.")
    r.add_argument("--exp", required=True, help="name for the new lineage (COORD_EXP/BRIDLE_EXP)")
    r.add_argument("--module", default=None, help="python module that registers the env")
    r.add_argument("--preflight", default=None,
                   help="path to preflight.yaml; overrides the manifest's own `preflight:` key")
    r.add_argument("--warm-start", default=None, help="ckpt to seed and to preflight against")
    r.add_argument("--force-warm-start", action="store_true",
                   help="overwrite an existing runs/<exp>/ckpt_1.pt with --warm-start")
    r.add_argument("--from-scratch", action="store_true", help="skip asserts needing a warm start")
    r.add_argument("--allow-vacuous-preflight", action="store_true",
                   help="proceed even though the resolved preflight contributed zero asserts")
    r.add_argument("--resume", action="store_true",
                   help="allow writing into an existing, non-empty runs/<exp>/ directory")
    r.add_argument("--cwd", default="/home/luca/lego-arm")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_relaunch)

    sk = sub.add_parser("skill", help="AUTHOR a new skill.yaml: print the vocabulary, check a "
                                      "document, compile it to a plan. Trains nothing and lists "
                                      "nothing (singular; see `skills` for what is already trained)")
    sk_sub = sk.add_subparsers(dest="skill_cmd", required=True)
    vc = sk_sub.add_parser("vocab", help="print the document grammar plus every term, parameter, "
                                         "measure and chassis a skill.yaml may use — the payload "
                                         "you put in the authoring model's prompt")
    vc.add_argument("--json-schema", action="store_true",
                    help="emit `spec.json_schema()` instead: the same surface as JSON Schema, for a "
                         "constrained-decoding harness rather than a prose prompt")
    vc.set_defaults(fn=cmd_skill)
    for verb, helptext in (
            ("check", "validate one skill.yaml (schema, then compile) and print the first refusal "
                      "with its dotted path; exits 1 if it does not pass. No simulator, no training"),
            ("compile", "print the reward plan a valid skill.yaml resolves to — every row in fold "
                        "order, every chassis-supplied default, and the plan fingerprint")):
        sp = sk_sub.add_parser(verb, help=helptext)
        sp.add_argument("file", help="path to a skill.yaml")
        sp.add_argument("--horizon", type=int, default=None,
                        help="the env's max_episode_steps. Omitted, the horizon-integrated flooding "
                             "check reports that it could NOT be computed — no default is invented")
        sp.add_argument("--terminate-on-success", choices=tuple(_TERMINATION), default="unknown",
                        help="does success end the episode? `unknown` counts the bonus once, which "
                             "is the conservative branch")
        sp.add_argument("--no-env-check", action="store_true",
                        help="do not resolve `env_id` against the simulator registry. The check "
                             "then reports NOT CHECKED rather than passing silently — and it "
                             "imports no simulator, which is the reason to want it")
        sp.set_defaults(fn=cmd_skill)
    #: TIER 3, and the reason it is a separate parser: its `file` is not a skill.yaml. It reads the
    #: JSONL a training run writes (`scripts/train_from_skill.py --train`), so `--horizon` and
    #: `--terminate-on-success` — which parameterise COMPILE-time checks — have no meaning here.
    dg = sk_sub.add_parser("diagnose",
                           help="read the per-term reward statistics a training run logged and "
                                "print the typed diagnostics: which row is flooding the return, "
                                "which is constant and unoptimizable, whether the policy is "
                                "earning without succeeding")
    dg.add_argument("file", help="path to a run's reward_terms.jsonl (written by "
                                 "scripts/train_from_skill.py --train, under runs/<exp>/)")
    dg.add_argument("--index", type=int, default=-1,
                    help="which emission to read; negative counts from the end (default -1, the "
                         "most recent window)")
    dg.add_argument("--list", action="store_true",
                    help="one line per emission — steps, episodes, success rate and tags — for "
                         "reading the trend rather than one window")
    dg.add_argument("--stored", action="store_true",
                    help="print the diagnostics the RUN emitted instead of re-deriving them from "
                         "the stored statistics. They differ only if the rules moved since.")
    dg.set_defaults(fn=cmd_skill)

    lg = sub.add_parser("lineage", help="check deployed records against the live environment")
    lg.add_argument("--cwd", default="/home/luca/lego-arm")
    lg.set_defaults(fn=cmd_lineage)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
