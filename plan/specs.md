# Joint-Placement Research Brief

Revision 4. Written against commit `12ac666`. Companion background reading
for future `/plan-feature` sessions in this problem area — not itself a
feature spec. Related prior plan: `plan/gradual_sparsity_annealing/`
([spec](gradual_sparsity_annealing/spec.md),
[research](gradual_sparsity_annealing/research.md)).

*Changes from Revision 1:* §1 corrected — the goal is scoped to the case
where the caller supplies pipeline-specific input, not the absence of it (see
below); added problem #6 (a numerical-stability defect, not a placement-
quality gap) and threaded it through §3.2, §7 and §8.

*Changes from Revision 2:* §1 success criterion made quantitative (50%
distance-ratio bar); §3.1 results corrected — annealing's effect on
*locality* (does the joint drive vertices near itself) turns out to be
separate from, and opposite to, its effect on *delta quality* for custom
joints (locality improves, delta worsens) — merged the previously
duplicated/contradictory bullets; §4 vastSolver citation resolved (Maya
plugin, linked) and hierarchy-awareness (problems #2/#4) marked nice-to-have,
not mandatory; §3.3 gained a constraint that solutions should avoid dropping
provided joints — flagged as a question against the existing
threshold-reassignment proposal, which currently *does* drop joints (§7).

*Changes from Revision 3:* resolved the dropping-vs-not tension per the
user — added an "ideal direction" bullet to §3.3 (train-time reallocation
of a joint heading to zero weight, instead of a post-solve drop) and
narrowed §7 #4 accordingly. Split the user's new open question (rest-pose
drift) into a §3.3 candidate proposal (bounded rest-position optimization)
plus a tightened §7 #6 that isolates the actual open call: whether drift
from the caller-authored position is acceptable at all, separate from
whether it would help.

## 1. Goal

Make compskin's rest-pose joints land the proper way — driving the vertex
region actually closest to them, concentrating non-deforming vertices onto
one root/head joint instead of scattering them — **when the caller supplies
pipeline-specific input** (a rest pose, a rig hierarchy, custom joint
positions), without giving up compskin's runtime/quality advantage over Dem
Bones (paper Table 1-3). That input being supplied is the case this brief is
about; the no-input path (fully automatic joint generation) is a separate,
already-acceptable case, and is in scope here only as a **non-regression**
constraint — see the design constraint below §2.

Success is quantitative: joint weight support should concentrate on the
vertices nearest each joint. Distance from the joint to the barycentric
position of the weighted area should be at maximum 50% of the distance
between the joint in question and its nearest joint. Static (non-deforming)
vertices should collapse onto a single joint rather than splitting across
several.

## 2. Problems

Problems 1-5 are gaps between "a sparse LBS decomposition" and "a rig an
animator can use," exposed by adopting compskin into a game pipeline that
supplies rest-pose/rig-hierarchy input (§1). Problem 6 is a numerical-
stability defect surfaced during that same work — a bug, not a
pipeline-knowledge gap, and not conditional on pipeline input being present.
Severity is the user's own judgment call, not a derived score.

| # | Problem                                                                                                                                                                                                                          | Why it matters | Severity  |
|---|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|-----------|
| 1 | No native knowledge of rigging                                                                                                                                                                                                   | Baseline gap; not itself actionable | none      |
| 2 | No knowledge of rig hierarchy, in particular the head joint (which should move all non-deforming vertices at once)                                                                                                               | Without this, the solver has no notion of "the rest of the head" | medium    |
| 3 | Non-moving vertices end up assigned to a variety of joints instead of one base/root joint (usually the head joint)                                                                                                               | Produces a rig topology animators won't accept | very high |
| 4 | No knowledge of deeper hierarchies (e.g. lower-lip/chin joints parented under the jaw joint, itself parented under the head joint)                                                                                               | Limits how well joint placement matches an authored rig | medium    |
| 5 | No real understanding of joint rest pose: ideally each joint should drive the vertices closest to it, so animators can use joints as intuitive tweakers                                                                          | Central to whether the output is usable as a rig at all, not just a compression artifact | critical  |
| 6 | (Probably due to geodesic distance filter) With large iteration numbers some weights collapse to 0 when some parts of the geometry is removed (e.g. mouth bag), causing the whole matrix to collapse to NaN over time (div by 0) | This causes the whole result to be invalid | critical  |

Problems #2 and #4 (hierarchy awareness) are nice-to-have, not mandatory —
see the priority note in §4 — and should not be pursued at the expense of
#3, #5 or #6.

**Design constraint on any fix:** keep the API simple and the code generic
enough for users who *don't* have these problems — e.g. automatic joint
generation without a supplied rest pose must keep working unchanged.

## 3. Prior art tried in this repo

### 3.1 Gradual sparsity annealing (implemented — `plan/gradual_sparsity_annealing/`)

- **Hypothesis:** relaxing the sparsity budget gradually across training
  stages (instead of enforcing the final $K$/$L$ from iteration 1) would let
  joint assignment settle into a better configuration before being locked in.
- **Result:** mixed, and the effect differs by axis — delta quality
  (mean/max error vs. baseline) moves opposite to locality (whether the
  joint drives vertices actually near itself):
  - Randomly generated joints: delta quality improved; locality unaffected
    — already placed at the center of their weighted area and driving the
    correct area by default, annealed or not.
  - Custom (user-supplied) joint positions: delta quality worsened relative
    to the single-stage baseline; locality improved significantly (the
    driven area moved markedly closer to the joint's actual position), but
    is still far from the §1 bar.
- **Status:** implemented and available (staged `iterations` schedules), but
  not a general win — see `plan/gradual_sparsity_annealing/design.md` for the
  mechanism and open questions.

### 3.2 Geodesic distance filter (implemented — see `model_fit.py`, `#3` in git log: `37323ab`)

- **Hypothesis:** restricting each vertex's candidate joints to its
  geodesically-nearest neighbors (`candidate_joints_per_vertex`) would keep
  custom joint positions local to the region they were placed in, improving
  the joint-to-weighted-area association from problem #5.
- **Result:** placement/weighted-area association significantly improved,
  though still imperfect. Interacts with annealing stage count:

  | Configuration | vs. baseline | Result (mean & max delta) |
  |---|---|---|
  | Custom joints + geodesic filter, single stage, 10000 iterations | random joints, single stage, 10000 iterations | ~1.1% improvement |
  | Custom joints + geodesic filter, 3 stages of 5000 iterations | random joints, 3 stages of 5000 iterations | ~1% loss |

  i.e. the filter helps custom joints in the single-stage configuration but
  *hurts* them once combined with the 3-stage anneal from §3.1 — the two
  fixes do not simply compose in our (so far, admittedly very limited) tests.
- **Status:** implemented, on by default when joint matrices are supplied
  (`candidate_joints_per_vertex = min(2K, P-1)`, per `CLAUDE.md`).
- **Known defect (problem #6):** at large iteration counts, if part of the
  geometry is excluded from a vertex's candidate set (e.g. an interior
  region like the mouth bag), that vertex's weights can collapse to zero,
  and the transformation matrix collapses to NaN over time (division by
  zero). Suspected trigger — a vertex whose geodesic candidate region ends
  up with no valid remaining joint — but not confirmed; see §7.

### 3.3 Proposed, not implemented

- **Proximity-based default weight initialization.** For custom joints only,
  initialize weights from joint proximity instead of the random seed
  (`constants.py` seed default `12345`), on the theory that a proximity-aware
  starting point survives the sparsity projections better than noise does.
- **Threshold-based root-joint reassignment** (addresses problem #3
  directly). Per vertex, take the largest displacement across all shapes;
  below a chosen threshold (e.g. 0.01mm) reassign full weight to a
  designated root/head joint and drop joints left with no weight, reporting
  the resulting error. This is ideally a rig-builder-side fix (post-solve), not a
  solver change, so it doesn't interact with the sparsity projections at all.
  (Changing the algorithm is also a valid option if supported by research,
  provided it preserves quality and doesn't change the result for randomly
  generated joints. This is especially true if that helps ensure that all
  provided joints can be used for deformation without dropping any one of
  them — ideally every single joint should move except for the root/head
  joint. **This constraint is currently in tension with the reassignment
  scheme described above, which explicitly drops joints left with no
  weight — see §7.**)
  A cheaper lossless variant exists for the special case of a joint whose
  transform is exactly zero across every shape (merge it into the root
  outright, no threshold or error to report). A solver-side variant — a
  fixed identity influence the optimizer can route weight onto directly —
  is more invasive (touches `model_fit.py`, needs a flag since regression
  tests are pinned to current numerics) and should only be considered if the
  builder-side fix leaves too few joints for the face.
- **Ideal direction on dropping (open, not yet designed):** rather than the
  post-solve reassignment above dropping a zero-weight joint outright
  (wasting its slot in the joint budget), identify joints heading toward
  zero weight *during* (or before) training and reallocate/reuse that slot
  for another region, so the final joint count stays at the number the
  caller provided. Not worked out — see §7.
- **Bounded rest-position optimization (open, acceptability not yet
  decided).** Let the solver nudge each custom joint's rest position within
  a small threshold of what the caller supplied (e.g. clamped or
  regularized to ≤X% of the distance to its nearest joint, matching the §1
  metric), instead of treating the supplied position as fixed, on the
  theory that a small amount of positional freedom could materially improve
  locality. Contingent on §7's open question: is drifting away from the
  caller-authored rest position acceptable at all, given problem #5 is
  specifically about joints staying where an animator/rig expects them?

## 4. External references

- **Dem Bones** [Le and Deng 2012; Electronic Arts, open-sourced 2019] is the
  paper's own baseline throughout (`paper/compressed_skinning_for_facial_blendshapes.md:83,107,461`)
  and is cited there as solving problem #5's joint-rest-pose property.
- **"vastSolver"** is a Maya plugin that can be found here :
  https://marketplace.autodesk.com/apps/c896f394-a280-4580-acf1-1303ea35f12c

The stated goal is to reach Dem Bones'/vastSolver's joint-placement quality
while keeping compskin's runtime and sparsity advantages over Dem Bones
(paper Tables 1-3). Additional improvement over dembones such as hierarchy
awareness (e.g. head/jaw) are also desirable but not mandatory and should
not be considered if they come at the expense of other more important goals.

## 5. Constraints

- **Numbers:** findings here and in any research this brief feeds may be
  mentioned in the resulting implementation plan/research docs, but **raw
  pipeline numbers must not be** — no model-specific dimensions, tuned
  `alpha`/`seed`/`L` values, or throughput figures in the public repo.
  Relative results measured against the `meta-compskin_private_tests` data
  (e.g. the §3.2 percentage deltas) are the one exception and are fine to
  cite, since they don't expose the underlying model. *Note: this narrows an
  earlier, stricter "no numbers in `plan/` at all" policy — flagging in case
  that's not the intended scope.*
- Package defaults (`P=100`, `K=8`, `L=6000`) stay the reference point for
  any example in produced docs; `K` and `P` are typically fixed by the
  target platform, so treat `L`, `seed`, and the staged `iterations`
  schedule as the tunable levers.

## 6. Testing & data

Tests and test data for this problem area live in the
`meta-compskin_private_tests` sibling repo, not here — fixtures under
`tests/test_data/maya/*.obj` (HEAD.obj + ~52 shapes, 5761 vertices) and
`tests/test_data/joint_matrices/matrices.json` (28 joints); see that repo's
`scripts/run_schedule_experiment.py` for the schedule-comparison driver. If
that data proves insufficient for a given experiment, say so to the user
rather than substituting synthetic data silently.

## 7. Open questions

1. **(Problem #6, critical, likely most urgent)** What exactly triggers the
   NaN collapse — is it specifically a vertex whose entire geodesic
   candidate region has been excluded from the mesh (e.g. the mouth bag
   interior), leaving it with an all-zero weight row that then gets divided
   by somewhere downstream? Needs a repro (large iteration count against a
   mesh that excludes an interior region) and a fix — likely either guarding
   the division or guaranteeing every vertex keeps at least one valid
   candidate joint.
2. §3.1 and §3.2 don't compose cleanly (3-stage anneal + geodesic filter
   underperforms single-stage + geodesic filter, for custom joints). Is
   there a configuration search planned, or should staged annealing be
   scoped out of the problem-#5 solution entirely?
3. Which of the two §3.3 proposals (proximity-based init, root-joint
   reassignment) is the next one worth a `/plan-feature` pass? They're
   independent and could be sequenced either way.
4. "Drop joints left with no weight" (the threshold-reassignment proposal,
   §3.3) vs. "don't drop provided joints" (the constraint added to the same
   section) — per the user, this stays open, but the ideal direction is a
   train-time (or pre-training) reallocation: catch a joint heading to zero
   weight early enough to reuse its slot elsewhere, rather than a post-solve
   drop that wastes it. No design for this yet.
5. §1 now has a quantitative locality bar (50% distance ratio); none of the
   §3.1/§3.2 results above were measured against it (they report delta % or
   qualitative locality only). Re-running the existing experiments against
   this bar would be needed before ranking them against each other.
6. Is it acceptable for a caller-supplied joint's *solved* rest position to
   drift from the position they authored it at, within a bounded threshold?
   Problem #5's whole premise is that a joint should stay where an
   animator/rig expects it, so this needs a caller-facing policy decision
   before the §3.3 bounded rest-position proposal is worth building —
   testing whether it *improves* locality is comparatively cheap once that's
   settled.

## 8. Next steps

Problem #6 is a correctness bug (it invalidates output), not a placement-
quality research question — fix it first: reproduce with a large iteration
count against a mesh that excludes part of the geometry (e.g. mouth bag
interior), find where the division by a zero weight sum happens, and guard
it. Only after that, pick one of the placement-quality open items from §7
and run `/plan-feature` on it; this brief is the background reading that
step should start from.

## 9. Deliverables
Produce research aimed a addressing the problems outlined here.
When possible the research should be backed by scientific papers,
and provide a short summary of the scientific paper in question.
The result of the research shall be added to the (for now empty)
document research_output.md (/Users/martinlanton/Dev/meta-compskin/plan/research_output.md).