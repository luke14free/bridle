# The skill document

[← README](../README.md)

> **YAML is a serialization, not the design.** `bridle/skill/` imports no YAML anywhere —
> `parse_spec` takes a plain `dict`. There is a typed Python front-end (`bridle.skill.Skill`,
> `Training`, `terms`, `predicates`) whose builders are generated from the vocabulary, so a new term
> cannot leave it behind; it produces the same document, through the same compiler, to the **same
> fingerprint**, pinned by a test. Write Python for yourself; hand a model YAML when the author is a
> model, because you can safely parse an untrusted document and you cannot safely `exec` untrusted
> code. Everything below describes the shared shape, in its YAML spelling.

A skill is a YAML file declaring a scene, a reward, and a criterion for success. The intended author
is a **local 27–30B model**, which is the constraint that shapes everything here: the vocabulary must
fit in a prompt alongside a task description, and every refusal must be something the model can act
on without reading the source.

```console
bridle skill vocab            # the whole authorable surface — this is the prompt payload
bridle skill check  f.yaml    # schema, then compile. Exit 1 on the first refusal
bridle skill compile f.yaml   # the resolved plan, every default, the fingerprint
```

Neither `check` nor `compile` starts a simulator.

## The document

```yaml
name: descend_to_target
kind: carry                        # chassis: supplies weight defaults AND their rationale
contract: stack
env_id: SO100DescendToTarget-v1    # an EXISTING registered env (see the scope limit below)

scene:
  goal: {type: platform, half: 0.04, top_z: 0.03}
  held: {type: cube, half: [0.014, 0.016]}
init:
  after: move_to_target            # WHOSE handoff this trains from — see below
  snapshot: descend_init_move
  sha256: "917e1025..."            # the CONTENT, so a swapped file is a refusal

params:                            # per-skill physics the Contract has no field for
  hover: {value: 0.015, severity: retrain, doc: "reward attractor height above resting"}

reward_scale: {divisor: 12.0}      # the inherited normalizer, stated rather than assumed
reward:
  - PredicateBonus  {weight: 1.0, predicate: grasped, why: "..."}
  - DistancePull    {weight: 2.5, measure: height_above_seat_live, kernel: one_minus_tanh,
                     k: 6.0, setpoint: params.hover, gate: grasped, why: "..."}
  - HingePenalty    {weight: 3.0, measure: height_above_seat_live, threshold: 0.0,
                     side: below, gate: grasped, why: "..."}
  - SuccessBonus    {value: 12.0, mode: replace, scope: preceding, why: "..."}
  - ActionPenalty   {weight: 0.001, norm: l2, why: "..."}

success: all[grasped, below_resting_height(band=params.low_band), ...]
```

Every entry under `scene:` must declare a `type`; every entry under `params:` must declare a
`severity` (`run` / `adapt` / `retrain`), so a per-skill parameter hashes into the fingerprint on the
same terms as a core field.

## `why:` is mandatory

Not decoration. The tuning rationale for these numbers normally survives only in source comments, and
YAML destroys comments — so the field that carries it is required on every row. It is also staging:
making a model describe the behaviour in prose before emitting the constrained call is a measured
improvement, not a style preference.

## The ordered fold

Rows are **not summed**. They fold in document order, `acc = op(acc)`, because `mode` can be
`add`, `replace` or `floor`, and a `replace` row overwrites everything above it.

That is why `SuccessBonus {mode: replace, scope: preceding}` sitting *before* the action penalty
matters: on the success step the reward is `12.0 - 0.001*||a||`, not a flat `12.0` and not the sum of
everything. A flat sum-of-rows compiler either drops the action penalty or makes the answer depend on
undeclared ordering. **Row order is part of the reward, and it is in the fingerprint.**

## Nine terms

`ActionPenalty` · `SuccessBonus` · `PredicateBonus` · `DistancePull` · `HingePenalty` ·
`VelocityPenalty` · `Ramp` · `ProgressPotential` · `RewardScale`

Nine express all 99 reward rows across a 15-primitive corpus, with none needing arbitrary Python.
Adding a tenth needs a measured justification.

## Measures carry a sign and a frame

19 measures, and both tags are load-bearing rather than metadata:

- **Sign.** `height_above_seat_live` is SIGNED. An unsigned one makes a crush penalty
  `clamp(-dz, min=0)` identically zero — silently deleting the term that exists because pressing to
  `dz = 0` broke 16 grasps out of 16.
- **Frame.** The same physical quantity graded against a live surface and against a goal frozen at
  episode init are two different measures, and one skill in the corpus uses both. So the frame is in
  the name: `height_above_seat_live` and `height_above_seat_static_goal` both exist, and the bare
  name is illegal.

## 17 predicates

`grasped` · `not_grasped` · `above_z` · `below_height` · `within_radius` · `in_cylinder` ·
`at_rest` · `undisturbed` · `height_above_resting_in` · `below_resting_height` · `and_` · `or_` ·
`not_` · `sustained` · `latched` · `forall` · `for_n`

Note the pair `height_above_resting_in(band)` (bounded, `0 ≤ h ≤ band`) and
`below_resting_height(band)` (unbounded below). They differ exactly where it matters — a pressed-down
object is still *low* but is not *inside a band* — and routing one criterion through the wrong one
disagreed on 37 of 64 measured states. Their docs point at each other so the choice is visible rather
than accidental. `forall` / `for_n` are declared but not yet evaluable.

## Six chassis

`approach` · `close_and_hold` · `hold_and_ramp` · `carry` · `carry_with_potential` · `release`

A chassis is a starting point that supplies weight defaults **and the rationale for each one**, in
the text the model reads before deciding whether to change the number. The 15-primitive corpus is
really these six copy-pasted, which is both why the vocabulary is small and why its coverage claim is
weaker than 99/99 suggests.

## Three tiers

| tier | form | when |
|---|---|---|
| 1 | a named term | covers every measured row |
| 2 | `expr:` — a safe arithmetic micro-language over named measures | a shape the vocabulary lacks |
| 3 | `custom: module:function` | last resort; fingerprinted as opaque |

Tier 2 is parsed against an `ast` whitelist and **never `eval`'d** — `Attribute` and `Subscript` are
refused outright, which makes dunder traversal impossible rather than merely unlikely. It evaluates
identically for a CPU float in a unit test and a batched CUDA tensor in training, because it is
restricted to operators that mean the same thing for both.

One asymmetry to know: tier 1 references a declared param as `params.hover`; inside an `expr:` you
write the bare name `hover`. The refusal says so.

## What the compiler refuses

Beyond schema errors, it rejects combinations with a **recorded failure mode**:

- **Reward flooding** — per-step shaping that out-earns completion, where the success bonus
  *replaces* the shaping and the agent must therefore choose. (Where the bonus *adds*, completion
  always pays strictly more and there is nothing to refuse.)
- **An attractor peaking at a contact surface** — a `DistancePull` whose setpoint is the resting
  surface of a signed measure, which is the 16/16 grasp-loss bug.

A model may still choose bad weights. It may not choose a combination that is already known to fail.

## Scope limit

The `scene:` block is parsed, validated and printed, but **not synthesised into a simulator
environment**. `env_id` must name an env that already exists. Authoring a genuinely new scene still
needs a Python env file; generating one is a later phase.


## `init:` — declaring the chain, not just naming a file

A downstream skill trains on the states its predecessors actually leave it. That dependency used to
be an opaque filename, and the failure mode is not theoretical: a snapshot was replaced under a
stable name with a *different predecessor's* handoff, and it fed the wrong initiation set to a skill
for two months. A reward proven byte-identical to its reference still trained to 0.06 instead of
0.94, because the data had changed and the reward fingerprint — which covers ops and scale — could
not see it.

So `init:` declares what the document expects, and three checks enforce it:

| check | needs | refuses when |
|---|---|---|
| `init.after` | nothing — pure data | the snapshot's own recorded provenance names a different predecessor |
| `init.sha256` | nothing — pure data | the file's content is not the content the document declares |
| preflight initiation distribution | a simulator, ~30 s | the episode does not start where this skill's reward was tuned for |

The snapshot files already recorded their provenance (`{"after": "move_to_target", ...}` or
`{"prim": "grab", ...}`); nothing had ever compared it to the document. When a file records no
provenance at all the check reports **NOT CHECKED** — it never passes silently.

The third check is the one the first two cannot replace. The project's own capture rule said to
validate a fresh snapshot by asserting *grasped-majority and plausible z* — and the broken snapshot
**passes both**: the cube is held, and it is airborne at a perfectly plausible height. What it is
not, is anywhere near where this skill is supposed to move it from (29.6 cm versus 8.4). Assert the
distance to the skill's own goal, not just that the object exists and is held.