# Joint-Placement Research Output

Revision 3. Written against commit `12ac666`, as the deliverable requested in
[specs.md](specs.md) §9, with the failure traces from
[additional_context.md](additional_context.md) and the experiment results in
[experiment_results.md](experiment_results.md). Companion reading: the
annealing notes (`gradual_sparsity_annealing/research.md`) and the
geodesic-filter notes (`joints_geodesic_distance_filter/research.md`). Code
lines refer to `src/metacompskin/model_fit.py` at that commit unless another
file is named.

*Changes from Revision 1:* the NaN mechanism is now confirmed by the private
traces (§2) and the fix is folded into the design for problems #3 and #5
rather than treated as a separate guard; §3 corrects an overstatement about
"global deformation modes" (the transform basis spans rotations about any
pivot; what it changes is their sparsity cost); the centre of the document is
now a single coherent design, a *rig-aware mode* built on a zero-transform
root influence and a hierarchical, joint-centred transform basis (§5), which
serves #3, #5 and #4 at once; the zero-code experiments a review found
missing are listed (§6); the metric section gains a calibration step (§7);
the bibliography gains the hierarchy and sparse-simplex literature (§10).
*Changes from Revision 2:* results from the private experiments are folded
in. The NaN reproduces on an uncut head with a long single-stage run (§2.1),
so it is a property of the warm-up plus the top-$K$ cut, not of cut meshes;
a new mechanistic detail explains why annealed schedules survive longer
(their warm-up runs at $K_0 = M$, where the cut is a no-op) and yields a
one-line mitigation, R0 (§2.2, §5.0). Tighter masks fail the same way (E2),
so every locality experiment is blocked until R1 lands. Three seeds spread
the errors by a few percent (E4), which retires the brief's one-percent
conclusions. The sparsity-cost argument for the joint-centred pivot was
tested and not supported (E7); R2's case now rests on inheritance alone, and
an effective-pivot diagnostic (E8) replaces the refuted proxy. Per the
user's note, the hard geodesic filter may be removed if #5 is solved another
way; §5.3 makes it a wide safety bound rather than the locality mechanism.
Per the brief's §5 no model-specific numbers appear here; examples use the
package defaults $P = 100$, $K = 8$, $L = 6000$.

## 1. Findings in one page

| # | Problem (brief §2) | Root cause | Literature | Recommendation |
|---|---|---|---|---|
| 6 | NaN collapse | Confirmed, and reproduced on an uncut head: after a long un-normalised warm-up the weight column of some vertices reaches exactly zero; the first normalised iteration divides by that zero sum (line 1069) and the NaN reaches every parameter through the shared $B_{rt}$. The mask is never empty; the top-$K$ cut zeroes small positive weights every step, the warm-up gives the survivors time to decay, and LBS has no way to express "no influence" without an identity joint (§2). Cut meshes only reach it sooner; tighter masks reach it sooner still (E2). | SSDR/Dem Bones re-initialise idle bones; James & Twigg's static bone; FLAME's neck joint; sparse-simplex projection (Kyrillidis 2013). | Immediately: warm up at $K_0 = M$ on the joint path, where the cut is a no-op (R0, §5.0). Properly: a zero-transform **root influence** with a floor, exempt from the cut (§5.1). Keep an output validation as the safety net. |
| 3 | Static vertices scatter | Same degeneracy as #6: nothing in the loss prefers one joint for a vertex the data term does not care about; the only regulariser acts on $\mathbf{BC}$, not on $W$; phase 2 renormalises whatever tiny survivors phase 1 left (§3.1). | Weight Laplacian (Le & Deng 2014, Dem Bones `weightsSmooth`); identity bone; bone pruning. | The root influence again (§5.1); optional builder-side snap for exactness; optional weight Laplacian (§5.4). |
| 5 | Joints do not drive their neighbourhood | Three causes: the hard geodesic mask is a feasibility set with no preference inside it, too loose for locality on rigs with few joints and too tight for survival on cut rims (E1); the assignment is locked in by noise in the first steps; in a flat model a vertex must weight a *far* joint (the jaw) to receive shared motion, which pulls supports away from the nearest joints (§3.2). The sparsity cost of local rotations is **not** a cause on the tested data (E7). | Rigid bones seeded spatially (SSDR, James & Twigg, Schaefer & Yuksel); weights from joint positions (Pinocchio, BBW, geodesic voxel binding, SkinCells); SPLOCS support maps; Neural Blend Shapes (weights for a given hierarchy). | **Hierarchical transform basis** so shared motion is inherited and local weights suffice (§5.2); a scale-free distance penalty as the locality mechanism, with the hard mask demoted to a wide safety bound or removed (§5.3); proximity initialisation (§5.3). Nothing can be measured before R0/R1 stop the NaN. |
| 2, 4 | No hierarchy, no head joint | Hierarchy has been treated as cosmetic. It is not: it decides whether local weights can carry global motion, and it stores shared motion once, which is a sparsity win on the paper's own metric (§5.2). | Le & Deng 2014 joint constraints; DemBonesExt local transforms; Neural Blend Shapes; MetaHuman practice (face joints under the head bone). | Accept an optional parents array with the joint matrices; head = root. |

The rig-aware mode is only active when joint matrices are supplied, so the
anonymous path and its pinned regression tests are untouched (§5.6).

## 2. Problem #6, confirmed

### 2.1 What the private traces show

Two long runs on a head with the mouth interior removed and caller-supplied
joints diverged to NaN, and a single-stage run at the default iteration
count on the same mesh had diverged before them. In the annealed run the
warm-up completed normally
and the very first progress line of the first normalised phase, at the
loosest influence budget, was already NaN; the same schedule with a warm-up
two orders of magnitude shorter completed. Removing the joints that sat far
from the remaining geometry changed nothing, which is expected: the default
candidate count is $M = \min(2K, P-1)$, so with joint counts well above $2K$
the mask keeps $2K$ candidates per vertex either way and only swaps which
joints fill the slots. Both observations match the prediction made in
Revision 1 (Route A) and rule out the "empty candidate set" hypothesis and
Route B.

The follow-up experiments (`experiment_results.md`) add four facts:

- **It is not a cut-mesh problem.** On the uncut head, a single-stage run
  with a very long warm-up diverged the same way, NaN within the first
  progress lines of the normalised phase, while an annealed run of the same
  total length completed. The cut only lowers the iteration count at which
  the collapse happens.
- **The cut rim is where it starts.** Measured with the solver's own
  distance functions and no training, the handful of worst-served vertices
  (largest distance to the farthest of their $M$ kept candidates) sit on one
  edge loop, the rim left by removing the mouth interior, and their farthest
  kept candidate lies several times the mesh-wide average away. The cut
  removed the surface paths through which the lip and mouth-corner joints
  used to reach them, so the mask hands them candidates that cannot drive
  them.
- **Tighter masks make it worse, not better.** Every value of $M$ tried in
  E2, down to $K + 1$, diverged at the tested budget. Fewer candidates means
  fewer weights that must decay before the column is empty. No locality
  measurement is possible until the collapse is fixed.
- **Annealed schedules survive longer for a structural reason** (§2.2), not
  because annealing helps the fit: with the filter, an annealed schedule's
  warm-up runs at $K_0 = M$, and at that budget the top-$K$ cut keeps every
  candidate.

### 2.2 The mechanism, precisely

`candidate_joint_mask` (lines 142-169) keeps exactly $M$ joints per vertex
and `geodesic_joint_distances` (line 138) makes every distance finite, so no
column of the mask is empty. In the un-normalised warm-up (line 274) the
forward pass uses $W$ directly, and for a vertex whose deltas are zero, or
whose deltas none of its $M$ candidates can reproduce, the loss decreases as
its weights shrink: the data term is $\sum_k \lVert \sum_j w_{ij}
\mathbf{N}_{k,j}\mathbf{v}_{0,i} - \mathbf{a}_{k,i} \rVert^2$ and with
uncorrelated candidates every positive weight adds error. Adam moves each
weight by about the learning rate per step whatever the gradient magnitude,
so the column reaches the clamp (line 1096) after enough steps and stays
there: once it is zero the data term's gradient is zero, and the Laplacian
term's gradient (line 1083) is zero as well whenever the neighbours are also
at zero. Without the mask the column has $P$ entries and the residual is
anti-correlated with *some* joint's motion, which keeps a weight alive; with
the mask the candidates are geodesic neighbours whose motions are
correlated, so nothing survives. The first normalised iteration then
computes $0/0$ (line 1069); the NaN enters $B_{rt}$'s gradient because
$B_{rt}$ is shared by all vertices, and Adam's moments keep it there.

Two operations empty a column, and the results separate them. The **cut**
(`topk`, line 1092) zeroes the $M - K$ smallest candidates every step, small
positive ones included, and a zeroed entry does not re-enter (annealing
notes §3). The **decay** is Adam walking the $K$ survivors to the clamp. On
the joint path a single-stage schedule warms up at $K = 8$ with $M = 16$, so
the cut discards half of every column each step and death needs only eight
survivors to decay. An annealed schedule warms up at $K_0 = M$
(`build_training_schedule`, line 224 onward, with `start_influences`
set): the $(K_0{+}1)$-th largest entry of a column with $M$ candidates is a
masked zero, so the cut keeps every positive candidate and only the clamp
acts; death then needs all $M$ candidates to cross zero on their own. That
is why the annealed runs survived where single-stage runs of equal length
did not, on both meshes, and why the cut mesh, whose rim vertices have no
useful candidate at all, eventually died even at $K_0 = M$. It also means
the single-stage joint-path default is the most dangerous configuration in
the code today, and that annealing's apparent robustness is a side effect
of its warm-up budget, not a quality result.

Whether the anonymous path shares the fragility is untested: it has $P$
candidates instead of $M$, so all $P$ must decay, which was never observed
at the tested budgets, but the tiny synthetic case in the private
improvement plan (item 2.2) shows it is possible.

### 2.3 Why a head/root influence is the right fix, and why a threshold alone is not

The brief's follow-up question, whether forcing non-moving vertices onto the
head joint would solve this altogether, has a two-part answer.

- For vertices that are static across every shape, yes: assigning them to a
  joint with zero transforms in every shape is exact, costs nothing from the
  $L$ budget, and removes the vertex from the degeneracy. A displacement
  threshold at the builder or before the solve does this.
- The traces cannot tell whether the dying vertices are static or merely
  *unexplainable* (a vertex that moves, but whose $M$ candidates cannot
  reproduce that motion). A threshold does not catch the second kind. A root
  influence *inside the solve* catches both: for an unexplainable vertex the
  optimiser can put the weight on the root and accept the residual, which is
  the honest answer, instead of driving the column to zero. It also turns the
  warm-up into a feature: a root weight has exactly zero gradient in the
  un-normalised phase (its transform is zero), so it keeps its initial
  positive value while the useless weights decay, and normalisation then
  yields weight one on the root for exactly the vertices that would have
  died. No NaN is possible as long as that value stays positive, which is
  why the root must be protected from the top-$K$ cut (see R1).

The instrumentation in §6 (E1) tells the two kinds apart after the fact, and
the report of per-vertex residual on root-assigned vertices tells the rigger
where a joint is missing.

### 2.4 What not to do

- An epsilon in the denominator ships an all-zero weight row, which Eq. 2
  skins to the origin at runtime. The solver would not notice, because a zero
  prediction fits a static vertex.
- Rescuing a dead column onto its nearest *moving* joint (Revision 1's P1a)
  makes a static vertex follow that joint, an error the root avoids.
- Removing geometry to hide interior vertices deletes the surface paths the
  geodesic filter and the Laplacian rely on. The paper's own suggested
  extension, per-vertex importance weights in the loss, achieves the intent
  (ignore the mouth interior) without cutting the mesh (E6 in §6).
- Tightening $M$ to force locality: it accelerates the collapse (E2) and,
  on the rim, forces vertices onto candidates that cannot drive them.
- Treating a warm-up at $K_0 = M$ (R0) as the fix: it removes the cut from
  the warm-up but not the decay, and the cut mesh died with it. It is a
  mitigation that costs one line and should land first, nothing more.

## 3. Why the solver has no sense of place

### 3.1 Static vertices (problem #3)

- **Degeneracy.** In the delta formulation a zero delta is fit exactly by
  $w_{i,:} = 0$, by any cancelling combination, and by any joint with zero
  transforms. Noise-level deltas are fit best by whichever joint's motion
  happens to align with the noise, regardless of distance.
- **The regulariser is silent there.** Line 1083 penalises
  $\lVert \mathbf{L}(\mathbf{BC})^{\top}\rVert^2$, the smoothness of the
  *reconstruction*, which is zero on static regions whatever the weights.
  Dem Bones' `weightsSmooth` (Le & Deng 2014) is a Laplacian on the *weight
  map* and is why its static regions come out contiguous.
- **Renormalisation of noise.** Phase 1 drives static weights toward zero;
  whatever tiny arbitrary survivors remain are scaled to sum one at the start
  of phase 2. The scatter the brief describes is the un-normalised warm-up's
  noise, frozen by the lock-in the annealing notes describe.
- **No identity influence.** `maya_rig_builder.py` creates a root only as an
  identity parent without weights; the solver has nothing that means "does
  not move".

### 3.2 Locality (problem #5)

Revision 1 said every joint is a "global deformation mode" because Eq. 8 is
written about the centred rest origin. That is wrong as stated: a linearised
rotation about pivot $\mathbf{p}$ equals a rotation about the origin plus the
translation $\mathbf{r}\times\mathbf{p}$, so the solver can place an
effective pivot anywhere, and the span of the transforms is the same for any
choice of basis origin. What the origin-centred basis changes is the **cost**
of a local transform: a rotation about the joint's own position needs six
non-zero coefficients instead of three, and the translation part grows with
the joint's distance from the head centre. Revision 2 predicted from this
that the top-$L$ projection (lines 1098-1104) would prune far joints'
rotation coefficients before their translation coefficients. **E7 did not
confirm it**: the correlation between a joint's distance from the centre
and its rotation-to-translation energy ratio came out weakly positive, the
wrong sign, and the only strong pattern was that joints near the centre
carry more non-zero coefficients of both kinds, which reads as those joints
doing more work, not as rotation-specific pruning. The argument is
withdrawn as a cause of #5 on this data; what the origin-centred basis does
to a joint's *effective pivot* remains unmeasured (E8). The corrected list
of causes:

1. **Shared motion forces far weights.** In a flat model a lower-lip vertex
   only moves with the jaw if it carries weight on a joint whose transform
   contains the jaw motion. Either the far jaw joint is in its candidate set
   and takes a slot, or every lip joint must duplicate the jaw motion in its
   own coefficients. Both pull supports away from "nearest joint drives
   nearest vertices" and both cost $L$ budget. A hierarchy removes the
   conflict (§5.2).
2. **The mask has no preference.** Inside the $M$ candidates the solver is
   indifferent to distance, and with $M = 2K$ a vertex may settle on its
   ninth-to-sixteenth nearest joints. For rigs with a few tens of joints
   $M = 2K$ covers a large fraction of all joints, so the filter barely
   constrains anything.
3. **Lock-in by noise.** The annealing notes show the top-$K$ survivors are
   chosen by the random $B_{rt}$ in the first steps and rarely change. The
   geodesic notes dismissed distance-shaped initialisation as unnecessary;
   if lock-in is real, initialisation is the strongest cheap lever, and that
   dismissal should be reversed.
4. **Not a cause on the tested data: the sparsity cost of local
   rotations** (E7, above).

## 4. Literature by mechanism

### 4.1 Spatial initialisation and region growing

- **SSDR / Dem Bones.** `init()` clusters the mesh sequence (LBG-VQ) into
  approximately $P$ rigid bones with 0/1 weights; `computeLabel()` assigns
  vertices to their best bone by a Dijkstra-like propagation from seed
  vertices, so labels are spatially contiguous from the first iteration;
  `split()` divides large, high-error clusters and `pruneBones()` removes
  bones with fewer than a handful of vertices (verified in `DemBones.h`).
- **James & Twigg 2005.** Mean-shift clustering of per-triangle rotation
  sequences; each vertex's influence set is restricted to bones of nearby
  triangles; non-rotating triangles form one identity bone.
- **Schaefer & Yuksel 2007, de Aguiar et al. 2008, Hasler et al. 2010.**
  Rigid clustering, then weights, then joints and hierarchy; locality
  enforced in the weights explicitly.

### 4.2 Weights derived from joint positions

- **Pinocchio (Baran & Popović 2007)**: heat equilibrium per bone on the
  surface graph; one sparse solve per bone.
- **Bounded biharmonic weights (Jacobson et al. 2011)**: Laplacian energy
  under bounds; the reference for intuitive handle weights.
- **Geodesic voxel binding (Dionne & de Lasa 2013)**: Maya's default bind,
  hence the rigger's expectation.
- **SkinCells (Larionov et al. 2025)**: Voronoi-cell weight functions with a
  direct per-vertex influence budget, by two of the compskin authors.
- **Neural Blend Shapes (Li et al. 2021)**: given a mesh in rest pose and a
  target skeleton hierarchy, predicts skinning weights and a pose-dependent
  residual with a differentiable envelope, i.e. "weights for a given
  hierarchy" learned under forward kinematics.

### 4.3 Locality and smoothness as regularisation

- **SPLOCS (Neumann et al. 2013)**: per-component support map ramping a
  sparsity penalty with geodesic distance from the component centre; centres
  re-estimated during the alternating solve.
- **Le & Deng 2014 / Dem Bones**: weight Laplacian with deformation-energy
  edge weights, solved implicitly (`computeSmoothSolver`, `compute_ws`).

### 4.4 Hierarchy

- **Le & Deng 2014**: rigid bones with joint constraints (parent and child
  transforms must agree at the joint), hierarchy inferred, redundant bones
  pruned.
- **DemBonesExt**: a supplied `parent` array and bind matrices; `computeRTB`
  converts the solved global transforms to local rotations and translations
  after the fact; `bindUpdate` moves bind translations to support centroids.
- **FLAME (Li et al. 2017)**: neck, jaw and eyeball joints with LBS plus
  blendshapes; the neck carries everything the face joints do not.
- **MetaHuman**: the face skeletal mesh puts several hundred facial joints
  under the head bone; lower LODs prune joints without changing the
  hierarchy (Epic documentation).
- **Moutafidou et al. 2023**: learned vertex-to-bone assignment with
  persistent labels from vertex trajectories, the paper compskin cites for
  recent skeletal fitting.

### 4.5 Constraints inside an Adam loop

- **Kyrillidis et al. 2013** (sparse simplex projection), **Condat 2016**
  and **Duchi et al. 2008** (simplex projection), **sparsemax (Martins &
  Astudillo 2016)** (the simplex projection as a differentiable layer),
  **AdaProx (Melchior et al. 2019)** and **ProxGen (Yun et al. 2021)** (the
  prox in Adam's metric), **PALM (Bolte et al. 2014)** (convergence on
  non-convex sparsity sets).

### 4.6 Reusing a dying joint's slot

- SSDR `split()`/`pruneBones()`, **RigL (Evci et al. 2020)** (regrow by
  gradient magnitude), **ReDo (Sokar et al. 2023)** (recycle dormant units).
  For caller-authored joints, reallocation means moving a joint the rigger
  placed, which conflicts with the brief's premise; see §8 #4.

## 5. The proposed design: a rig-aware mode

Everything below is keyed on `rest_joint_matrices` being supplied (plus an
optional parents array). With identity rest matrices, no parents and no
root, every formula reduces to today's solver exactly.

### 5.0 R0: warm up at $K_0 = M$ on the joint path

One line in `build_training_schedule` (line 224 onward): when
`start_influences` is set, the un-normalised warm-up uses that budget even
for a single-stage schedule, so the cut never discards a candidate during
the warm-up (§2.2). Annealed schedules already behave this way. This changes
the joint path's single-stage numerics, which no public test pins, and
leaves the anonymous path untouched. It is a mitigation, not a fix: the
decay still empties columns on rims and, given a large enough budget,
elsewhere. Land it first because it is free and because it makes the
default configuration the survivable one until R1 exists.

### 5.1 R1: a zero-transform root influence

- One joint, the caller's head joint (or a synthetic one when the caller
  gives none), whose $B_{rt}$ block is held at zero and excluded from the
  top-$L$ projection. It costs nothing from $L$ and nothing at runtime beyond
  one identity transform.
- It is a universal candidate: its mask row is all true, so every vertex has
  $M + 1$ candidates and the root never competes for the $M$ geodesic slots.
- Every vertex starts with a small positive root weight, and the root is
  **exempt from the top-$K$ cut with a floor**: after each projection the
  root entry is reset to at least that floor. Without the exemption the
  root would be pruned within the first steps, because Adam's noise-driven
  steps outgrow any fixed small value in a handful of iterations, and the
  lock-in described in the annealing notes means a pruned root does not
  come back; a vertex whose other weights later decay would then die
  exactly as today. With the floor the column sum is always positive, so
  the warm-up cannot produce a zero column: vertices whose other weights
  decay end up fully on the root at normalisation (§2.3). In normalised
  phases the root weight receives the "dilution" gradient
  $-\frac{1}{s}\sum_k \tilde w_{ik} g_{ik}$, negative (so the weight grows)
  whenever the vertex overshoots, so static vertices keep flowing to the
  root while moving vertices leave it at the floor.
- The $K$ budget is reconciled at save time, not during training: root
  weights below a tolerance are dropped and the column renormalised (a
  relative change of the floor's order), and a vertex that keeps a genuine
  root weight on top of $K$ other influences drops its smallest influence
  and renormalises, with the count and the largest such change reported.
  The alternative, training with $K-1$ non-root influences plus the root,
  respects the budget exactly at a small capacity cost and is the fallback
  if the reconciliation changes too many vertices.
- Effects: #6 cannot occur; #3 is addressed at the source, because "does not
  move" is now representable; the joint count the caller supplied is
  preserved.
- Optional exactness: a builder-side snap that moves any vertex with root
  weight above a threshold fully onto the root, reporting the error, keeps
  memory option B's behaviour for riggers who want static skin exactly on
  the head. With R1 in the solve the snap is a cosmetic step, not the fix.
- Risk to watch: moving vertices using the root as a "partial rigidity" knob
  and spending a $K$ slot on it. This is legitimate LBS, but the coverage
  report should list the fraction of moving vertices carrying root weight.

### 5.2 R2: hierarchical, joint-centred transform basis

**Parametrisation.** For joint $j$ with rest position $\mathbf{p}_j$ and
parent $\pi(j)$ (root for top-level joints), store a *local* delta
$\mathbf{D}_{k,j}$ per shape in the Eq. 8 family, acting about the joint's
own position:

$$
\mathbf{D}_{k,j}\,\mathbf{v} = \mathbf{r}_{k,j}\times(\mathbf{v}-\mathbf{p}_j) + \mathbf{t}_{k,j},
\qquad
\mathbf{N}_{k,j} = \sum_{a \in \mathrm{anc}(j)\cup\{j\}} \mathbf{D}_{k,a}.
$$

The Eq. 8 family is closed under addition, so $\mathbf{N}_{k,j}$ stays in it
and Eq. 7 is unchanged. In the origin basis $\mathbf{D}_{k,j}$ has rotation
$\mathbf{r}_{k,j}$ and translation $\mathbf{t}_{k,j} - \mathbf{r}_{k,j}\times
\mathbf{p}_j$, and the tree sum is a cumulative-sum matrix over ancestors:
the whole thing is one fixed linear map $\mathbf{T}$ from the new
coefficients to today's $B_{rt}$. Because $\mathbf{T}$ is invertible (set
$\mathbf{D}_{k,j} = \mathbf{N}_{k,j} - \mathbf{N}_{k,\pi(j)}$), **the span is
identical to today's** and the optimum of the unconstrained problem does not
move. Only the top-$L$ projection sees a different vector, and that is the
point. No forward-kinematics approximation is involved: the deltas are fit
directly in this basis.

**Why it serves #5.** A lower-lip vertex weighted only on its nearest lip
joint inherits the jaw's $\mathbf{D}_{k,\text{jaw}}$ through the chain. Local
weights can therefore carry global motion, the jaw need not be in any lip
vertex's candidate set, and the mask can be tightened without losing the
jaw. This is the mechanism a rigger's own skinning relies on, and the reason
Neural Blend Shapes and every skeletal-rigging paper fit weights under a
hierarchy rather than against independent bones.

**Why it serves sparsity.** Shared motion is stored once in the parent's
$\mathbf{D}$ instead of once per descendant. With the defaults, a jaw-open
shape that today costs six coefficients in every lower-face joint costs six
in the jaw and only the residual in each child. The projection therefore
buys more accuracy per non-zero, and a wrong hierarchy can only cost
sparsity (a child cancels its parent), never accuracy.

**Why joint-centred, after E7.** Revision 2 also argued that a pivot at
$\mathbf{p}_j$ makes local rotations cheaper under the $L$ budget and would
therefore improve locality. E7 tested the premise and did not support it
(§3.2), so that is no longer a reason to build the pivot change. Two
reasons remain, neither of which E7 touches: the solved
$\mathbf{r}_{k,j}, \mathbf{t}_{k,j}$ are then directly the joint's local
rotation and translation, which is what the rig builder keys and what the
inheritance rule composes (a parent rotates about its own pivot, a child
about its own, the linearised form of forward kinematics); and it is free,
since $\mathbf{T}$ is a fixed linear map. Treat the pivot as part of the
hierarchy's bookkeeping, not as a locality lever, and let E8 (the effective
pivot of the solved transforms relative to the joint and to the head
centre) decide whether it changes anything measurable.

**Units.** The top-$L$ projection compares $\lvert\mathbf{r}\rvert$ in
radians with $\lvert\mathbf{t}\rvert$ in model units, today as well as under
R2. R2 makes local rotations relatively cheaper by removing the
$\mathbf{r}\times\mathbf{p}_j$ term from the translation, but the comparison
still depends on the model's scale; a length scale per joint (its support
radius) applied to $\mathbf{r}$ before the projection would make it
scale-aware, and is worth a flag in the same plan.

**Runtime.** Store the sparse $\mathbf{D}$ and the parents array; per frame,
compute $\mathbf{E}_a = \sum_k c_k \mathbf{D}_{k,a}$ with the existing sparse
product, then accumulate $\mathbf{N}_j = \sum_{\mathrm{anc}} \mathbf{E}_a$
down the tree: $P$ additions of $3\times4$ blocks, negligible against the
sparse product. `AnimationFrameGenerator` gains that accumulation when the
NPZ carries parents; without parents the format and the code path are
today's.

**Rig builder.** Keying the global $\mathbf{N}_{k,j}$ per joint through
`frame_joint_matrices` stays exact. Keying local channels instead, the way
a hand-made rig is animated, reproduces the shapes to first order, the same
order as the paper's linearised rotations; the plan should measure that
discrepancy on a real head before offering local keys.

**Inputs.** `matrices.json` today holds flat $4\times4$ matrices, no names
or parents. The Maya exporter can write a `joint_parents` array (index of
the parent, $-1$ for the head) alongside `rest_joint_matrices`. Without
parents the mode falls back to a flat hierarchy under the root, which is the
joint-centred basis plus R1.

### 5.3 R3: locality on the weight side

With R1 and R2 in place the weight-side levers become cheap to apply and
easy to measure:

- **The hard mask is the wrong instrument, at both ends.** At $M = 2K$ it
  is too loose to enforce locality on rigs with few joints (§3.2) and too
  tight to keep rim vertices alive (§2.1); tightening it fails faster (E2).
  The user's note in `experiment_results.md` allows removing it if #5 is
  solved another way. Keep the geodesic *distances* (the penalty and the
  initialisation need them) and demote the *cut* to a wide safety bound,
  for example every joint within three times the vertex's nearest-joint
  distance, or turn it off; with R1 in place a wide or absent mask cannot
  produce a dead column.
- **Distance penalty, scale-free, as the locality mechanism.** Add
  $\lambda \sum_{i}\sum_{j \ne \text{root}} w_{ij}\,\rho_{ij}$ with
  $\rho_{ij} = \max\!\big(0,\ d_{ij}/d_i^{(1)} - 1\big)^2$, where $d_{ij}$ is
  the geodesic distance and $d_i^{(1)}$ the vertex's distance to its nearest
  joint. The penalty is zero on the nearest joint and on the root, grows
  with relative distance, and pulls each joint's weighted barycentre toward
  the joint, which is the §1 metric itself. A rim vertex whose nearest
  joints are all far pays a cost proportional to relative distance rather
  than being forced onto useless candidates, and with R2 whichever
  near-ish joint it takes inherits the jaw's motion. One elementwise product
  per step; $\lambda$ tuned by the metric, not by error alone. The geodesic
  notes rejected a soft penalty because it cannot *guarantee* locality; the
  wide bound above restores a guarantee where a rigger needs one, and the
  penalty supplies the preference the hard mask never had. SPLOCS is the
  precedent.
- **Proximity initialisation.** Start $W$ from a local map instead of noise:
  the cheapest is a Dem Bones-style label (weight one on the nearest joint,
  a root weight as in R1), the smoother option is Pinocchio heat weights on
  the existing adjacency graph restricted to the mask. Leave $B_{rt}$ at its
  noise initialisation; with meaningful weights its first gradient already
  reflects each joint's region. Do not initialise the transforms by least
  squares: a dense transform init is destroyed by the top-$L$ projection on
  step one.
- **Candidate policy.** $M = \min(2K, P-1)$ was chosen with $P = 100$ in
  mind and E2 shows that sweeping it downward is not an option before R1.
  After R1, repeat E2 with the metric and add the wide-bound and no-mask
  arms; the relative penalty above is what should carry locality in all of
  them.
- **Warm-up.** With a proximity init there is no ownership to discover, and
  the warm-up is what kills columns; with R1 the warm-up is harmless.
  Normalising from step one requires a non-negative initial $W$ (E3). Try
  both and keep whichever the metric prefers.

### 5.4 R4: optional weight Laplacian

$\beta \lVert \mathbf{L} W^{\top}\rVert_F^2$ with $\beta = 0$ by default, the
term the paper omitted and Dem Bones has. It makes supports contiguous and
stops single stray vertices, and it interacts with the mask at region
boundaries (neighbours with different candidate sets), so it comes after R3
and is tuned on the metric.

### 5.5 Idle joints and reallocation

With R1 absorbing static vertices, a joint without support is visible early
(its $\mathbf{D}$ block is pruned to zero). For caller-authored joints the
only legitimate "reallocation" is toward vertices for which the joint is a
candidate and that currently sit on the root with a non-trivial residual:
give such a joint the RigL-style gradient signal at a stage boundary and let
it re-enter there. Anything more (moving the joint) is the rigger's
decision, so the solver reports idle joints and root-assigned residuals
rather than acting. For the anonymous path, SSDR-style split-and-prune
remains the reference and is out of scope here.

### 5.6 Non-regression

- R1 to R4 are active only with joint matrices (or an explicit flag). The
  anonymous path keeps noise init, the un-normalised warm-up, the origin
  basis and no root, so the pinned tests are unchanged.
- R2 with identity rest matrices and no parents is today's basis
  ($\mathbf{T} = \mathbf{I}$).
- The NPZ format gains `joint_parents` only when a hierarchy is used; the
  existing keys keep their meaning.

### 5.7 Expected effects and what could go wrong

| Element | #3 | #5 | #4 | Error | Sparsity | Risk |
|---|---|---|---|---|---|---|
| R0 warm-up at $K_0 = M$ | neutral | neutral | neutral | none | none | mitigation only; rims still die given the budget |
| R1 root | fixes | neutral | needs a head | none for static vertices; honest residual for unexplainable ones | frees $L$ | moving vertices parking weight on the root |
| R2 hierarchy (+ pivot bookkeeping) | neutral | large: local weights carry global motion | fixes | unchanged span | shared motion stored once | wrong parents cost non-zeros; local-key playback is first-order; the pivot's sparsity benefit is not supported (E7) |
| R3 penalty + init + $M$ | helps (contiguous) | direct | neutral | trades error per $\lambda$ | neutral | $\lambda$ tuning per rig |
| R4 weight Laplacian | helps | helps | neutral | small | neutral | boundary effects at mask edges |

## 6. Experiments: results so far and what remains

Results are reported qualitatively or as relative figures; the raw runs are
in `experiment_results.md`, on data outside both repositories.

### 6.1 Run

| Id | Run | Result | Consequence |
|---|---|---|---|
| E1 (mask geometry) | Distance to the nearest and to the farthest kept candidate for every vertex of the cut mesh, using the solver's own functions, no training | The worst-served vertices form one edge loop at the cut rim, with their farthest kept candidate several times the mesh-wide average away | The mask strands rim vertices; cutting geometry removes the paths joints need (§2.1). Motivates E6 and the mask demotion in §5.3. |
| E1 (uncut head) | Very long single-stage run versus an annealed run of the same total length, on the full head | Single-stage diverged to NaN in the normalised phase; annealed completed | Not a cut-mesh problem; the warm-up budget explains the difference (§2.2); R0 and R1 follow. |
| E2 ($M$ sweep) | $M \in \{K+1, K+2, 2K\}$ at the long budget | All diverged | Locality cannot be measured before R0/R1; tighter masks accelerate the collapse. Re-run after R1 with the metric. |
| E4 (seeds) | Three seeds, otherwise identical | Mean error spreads by about two percent, maximum error by about four percent | The brief's one-percent conclusions (§3.2 of the brief) are inside seed noise; differences below roughly five percent need several seeds; the earlier annealing implementation's double-digit gain stands. |
| E7 (coefficient energy) | Correlation between a joint's distance from the head centre and its rotation-to-translation energy ratio | Weakly positive, the wrong sign for the prediction; near-centre joints simply carry more coefficients of both kinds | The sparsity-cost argument for the joint-centred pivot is withdrawn (§3.2, §5.2). |

### 6.2 Blocked or still to run

- **E3. Warm-up on/off**: needs a non-negative initial $W$ (absolute value
  of the noise, or a proximity init) because the current mixed-sign noise
  sums to nearly zero per column. Once R0 exists, E3 doubles as the check
  that the decay alone, without the cut, still empties columns at the long
  budget.
- **E2 again, after R1**, with the wide-bound and no-mask arms and the §7
  metric as the primary readout.
- **E5. Fixed-weights bound**: fix $W$ to a proximity bind and solve only
  $B_{rt}$ (needs a solver mode). Prices the §1 bar in error and gives the
  locality ceiling.
- **E6. Importance weights instead of geometry removal**: per-vertex loss
  weights (needs a loss term). Directly addresses the rim finding of E1 by
  keeping the surface paths.
- **E8. Effective pivot.** For every solved $(k, j)$ with a non-zero
  rotation, the linearised transform fixes a pivot *line* with direction
  $\mathbf{r}$ through $\mathbf{p}^{*} = (\mathbf{r}\times\mathbf{t})/
  \lVert\mathbf{r}\rVert^2$ (the line's closest point to the origin).
  Report, weighted by $\lVert\mathbf{r}\rVert$ over shapes, the distance from
  that line to the joint's rest position against the distance from the line
  to the head centre. This measures what E7 proxied: whether far joints
  rotate about themselves or about the centre. If pivots already sit near
  the joints, the pivot part of R2 is bookkeeping only; if they sit near the
  centre, R2's pivot is worth keeping for rig plausibility even though E7
  found no sparsity effect.
- **E9. Metric on existing outputs.** Compute §7 on the runs that
  completed (the annealed uncut run, the seed runs) and on a reference bind
  from the same joints. No solver change; gives the first reading of the
  bar and its calibration before any of R1 to R4 is built.
- **E10. Anonymous-path fragility.** One long single-stage anonymous run at
  the budget that killed the joint path, to learn whether R1 should be
  offered there behind a flag.

## 7. Measuring the §1 bar

For joint $j$: $\bar{\mathbf{x}}_j = \sum_i w_{ij}\mathbf{v}_{0,i} /
\sum_i w_{ij}$, $d_j = \min_{l \ne j}\lVert\mathbf{p}_l - \mathbf{p}_j\rVert$,
pass if $\lVert\bar{\mathbf{x}}_j - \mathbf{p}_j\rVert \le 0.5\,d_j$. Everything
needed is in the output NPZ, so the metric is an evaluation module, not a
solver change. Four refinements:

1. **Calibrate first.** Compute the metric on a reference bind from the same
   joints (heat weights, or Maya's geodesic voxel bind) before measuring the
   solver. The bar scales with nearest-joint spacing, so it is loose for an
   isolated joint and very strict for lip joints spaced closer than the
   region each must drive; a correct lip support can fail it by
   construction. Joints failing on the reference bind need a per-joint floor.
2. **Curvature.** The Euclidean barycentre of a curved patch lies under the
   surface; snap the joint to its nearest vertex as the filter does and
   compare along the surface, or use a weighted geodesic centroid.
3. **Symmetric supports.** A ring passes the barycentre test with the wrong
   support; add the weight-weighted 90th-percentile geodesic distance from
   the joint, relative to $d_j$.
4. **Static concentration.** Count vertices whose maximum displacement over
   all shapes is below a threshold and report how many joints carry their
   weight; with R1 the target is exactly one, the root.

## 8. Answers to the brief's open questions

1. **NaN trigger.** Confirmed as the all-zero column at the first
   normalised step after a long warm-up, on cut and uncut meshes (§2), with
   the cut and the decay separated (§2.2). Mitigate with R0 now; fix with
   R1, plus output validation. The follow-up question in
   `additional_context.md` is answered in §2.3: a head influence solves it,
   and it should live in the solve, not only in a threshold.
2. **Annealing and the filter.** The brief's composition result is inside
   the seed spread measured in E4 and is withdrawn until re-run over several
   seeds. Moreover the annealed schedule's apparent robustness against the
   NaN is its warm-up budget, not a fit property (§2.2). Mechanistically,
   annealing inside the mask only re-ranks the survivors of a noise
   assignment; after R3's initialisation and R2's inheritance there is much
   less for it to fix. Re-test after those, with seeds.
3. **Which §3.3 proposal next.** Neither as written. The root influence
   (R1) is the smallest change with the largest effect and it is the
   precondition for measuring anything else without NaN; then R2, because
   it changes what "local weights" can express; then the weight levers in
   R3, which the metric can tune. The brief's proximity init is inside R3;
   its threshold reassignment becomes R1's optional snap.
4. **Dropping versus reallocating.** With R1 the caller's joint count is
   preserved by construction; idle joints are reported, and the only
   reallocation compatible with fixed positions is residual-driven re-entry
   among a joint's own candidates (§5.5).
5. **Measuring against the bar.** §7, with calibration first.
6. **Drift.** Automatic methods move joints to their support (Dem Bones
   `bindUpdate`, SPLOCS centres, Le & Deng 2014) because the support is
   their truth; for authored tweakers the joint is the truth. Keep positions
   fixed, make the support respect them (R2, R3), and report the desired
   drift, which the §7 metric already is. Build bounded drift only if the
   metric stays out of reach after R1 to R3.

## 9. Suggested sequence for `/plan-feature`

1. R0 (one line) and the output validation; E9 on the completed runs.
2. R1 root influence, coverage report extension; then E2 again and E3.
3. §7 metric module, calibrated (E5 gives the bounds; E8 for pivots).
4. R2 parents array through exporter, solver basis, NPZ, runtime
   accumulation, builder. The pivot part is bookkeeping unless E8 says
   otherwise.
5. R3 penalty, init, mask policy (wide bound or off), warm-up choice,
   tuned by the metric over several seeds.
6. R4 if supports are still fragmented.
7. Annealing re-test with seeds; E10 to decide whether the anonymous path
   gets R1 behind a flag.

## 10. Annotated bibliography

Each entry: what the paper does, then what this repo takes from it.

1. **B. H. Le, Z. Deng. *Smooth Skinning Decomposition with Rigid Bones.*
   ACM TOG 31(6), SIGGRAPH Asia 2012.** https://doi.org/10.1145/2366145.2366218
   LBS extraction from example poses as constrained least squares over
   rigid bone transforms and a sparse convex weight map, by block coordinate
   descent, initialised by clustering vertices on their motion. *Take:*
   spatial seeding; idle bones are split and pruned, not left idle.
2. **B. H. Le, Z. Deng. *Robust and Accurate Skeletal Rigging from Mesh
   Sequences.* ACM TOG 33(4), SIGGRAPH 2014.** https://doi.org/10.1145/2601097.2601161
   Skeleton, joint positions, weights and transforms as one optimisation
   with joint constraints and weight smoothness, alternating updates,
   pruning redundant bones. *Take:* the weight Laplacian (R4); joint
   constraints as the rigid counterpart of R2; pruning as consolidation.
3. **Electronic Arts. *Dem Bones* (2019).** https://github.com/electronicarts/dem-bones
   Header-only implementation of 1 and 2. `init()` LBG-VQ clustering to 0/1
   weights; `computeLabel()` Dijkstra-like propagation from seeds;
   `split()`, `pruneBones()`; `weightsSmooth`, `weightsSmoothStep`,
   `weightEps`, `nnz`, `lockW`, `lockM`; `DemBonesExt` with `parent`, bind
   matrices, `computeRTB`, `bindUpdate`. *Take:* the baseline's exact knobs;
   region growing as initialisation; hierarchy handled after the solve,
   which R2 improves on by putting it inside.
4. **D. L. James, C. D. Twigg. *Skinning Mesh Animations.* ACM TOG 24(3),
   SIGGRAPH 2005.** https://doi.org/10.1145/1073204.1073206
   Mean-shift clustering of triangle rotation sequences; influence sets from
   nearby triangles; NNLS weights with $K$ influences; static triangles form
   an identity bone. *Take:* the identity bone (R1) and the locality mask.
5. **L. Kavan, P.-P. Sloan, C. O'Sullivan. *Fast and Efficient Skinning of
   Animated Meshes.* CGF 29(2), Eurographics 2010.**
   https://doi.org/10.1111/j.1467-8659.2009.01602.x
   Coordinate-descent skinning decomposition with unmodified LBS. *Take:*
   reference alternating solver from one of the compskin authors.
6. **I. Baran, J. Popović. *Automatic Rigging and Animation of 3D
   Characters.* ACM TOG 26(3), SIGGRAPH 2007.** https://doi.org/10.1145/1276377.1276467
   Pinocchio: skeleton embedding plus heat-equilibrium weights on the
   surface with a nearest-bone heat term. *Take:* the cheapest proximity
   init (R3), on the graph the Laplacian already uses.
7. **A. Jacobson, I. Baran, J. Popović, O. Sorkine. *Bounded Biharmonic
   Weights for Real-Time Deformation.* ACM TOG 30(4), SIGGRAPH 2011.**
   https://doi.org/10.1145/2010324.1964973
   Laplacian-energy-minimising weights under bounds. *Take:* reference
   "intuitive" weights; a costlier init and a calibration bind (§7).
8. **O. Dionne, M. de Lasa. *Geodesic Voxel Binding for Production
   Character Meshes.* SCA 2013.** https://doi.org/10.1145/2485895.2485919
   Voxel geodesic distances to bones, smooth weights at interactive rates,
   robust to production meshes; Maya's default bind. *Take:* the rigger's
   expectation and the calibration bind (§7).
9. **E. Larionov, I. Santesteban, H. Chen, G. Lin, P. Herholz, R. Goldade,
   L. Kavan, D. Roble, T. Stuyck. *SkinCells: Sparse Skinning using Voronoi
   Cells.* arXiv 2506.14714, 2025.** https://arxiv.org/abs/2506.14714
   Voronoi-cell weight functions with a per-vertex influence budget, from a
   mesh and skeleton in rest pose, optimised in the volume. *Take:*
   sparsity-aware locality by construction; a candidate initialiser.
10. **P. Li, K. Aberman, R. Hanocka, L. Liu, O. Sorkine-Hornung, B. Chen.
    *Learning Skeletal Articulations with Neural Blend Shapes.* ACM TOG
    40(4), SIGGRAPH 2021.** https://doi.org/10.1145/3450626.3459852
    Given a rest mesh and joint rotations on a target hierarchy, an envelope
    branch predicts weights and rigging parameters under a differentiable
    envelope, and a residual branch predicts blend shapes and coefficients
    from the joint rotations. *Take:* weights are fit *under the hierarchy*,
    with pose-dependent residuals as blendshapes; the closest modern
    analogue of R2 plus compskin's delta shapes.
11. **T. Neumann, K. Varanasi, S. Wenger, M. Wacker, M. Magnor,
    C. Theobalt. *Sparse Localized Deformation Components.* ACM TOG 32(6),
    SIGGRAPH Asia 2013.** https://doi.org/10.1145/2508363.2508417
    Spatially varying L1 with geodesic support maps and centre updates.
    *Take:* the soft locality penalty (R3) and the literature's form of
    joint drift (§8 #6).
12. **T. Li, T. Bolkart, M. J. Black, H. Li, J. Romero. *Learning a model of
    facial shape and expression from 4D scans.* ACM TOG 36(6), SIGGRAPH Asia
    2017.** https://doi.org/10.1145/3130800.3130813
    FLAME: shape space with articulated neck, jaw, eyeballs, pose
    correctives and expression blendshapes. *Take:* the head/neck joint as
    the "everything else" influence (R1).
13. **S. Schaefer, C. Yuksel. *Example-Based Skeleton Extraction.* SGP
    2007.** https://diglib.eg.org/handle/10.2312/SGP.SGP07.153-162
    Rigid face clustering, weights with enforced locality, joints and
    hierarchy from the weights. *Take:* the "cluster, weights, joints" order
    and explicit locality in the weights.
14. **E. de Aguiar, C. Theobalt, S. Thrun, H.-P. Seidel. *Automatic
    Conversion of Mesh Animations into Skeleton-based Animations.* CGF
    27(2), Eurographics 2008.** https://doi.org/10.1111/j.1467-8659.2008.01136.x
    Kinematic skeleton, motion parameters and skinning weights from
    arbitrary mesh animations. *Take:* a second reference for hierarchy
    inference from motion.
15. **N. Hasler, T. Thormählen, B. Rosenhahn, H.-P. Seidel. *Learning
    skeletons for shape and pose.* I3D 2010.** https://doi.org/10.1145/1730804.1730809
    Rigid skeleton, connectivity, joint positions and weights from sparse
    example poses, across subjects. *Take:* skeleton fitting stays within
    standard skinned rigid skeletons for pipeline compatibility, the same
    constraint R2 respects.
16. **A. Moutafidou, V. Toulatzis, I. Fudos. *Deep fusible skinning of
    animation sequences.* The Visual Computer 40, 2024 (online 2023).**
    https://doi.org/10.1007/s00371-023-03130-3
    Learned, persistent vertex-to-bone labels from vertex trajectories for
    LBS compression. *Take:* the citation the compskin paper uses for recent
    skeletal fitting; data-driven assignment, not pursued here.
17. **Epic Games. *MetaHuman DNA, Rig Definition, and Rig Operation.***
    https://dev.epicgames.com/documentation/metahuman/metahuman-dna-rig-definition-and-rig-operation
    Not a paper. The face skeletal mesh carries several hundred facial joints
    under the head bone; LODs prune joints without changing the hierarchy.
    *Take:* industry practice for what a game face rig expects of a
    hierarchy and of the head bone.
18. **A. Kyrillidis, S. Becker, V. Cevher, C. Koch. *Sparse projections onto
    the simplex.* ICML 2013.** https://proceedings.mlr.press/v28/kyrillidis13.html
    Exact projection onto the simplex intersected with a sparsity
    constraint. *Take:* the exact projection for $W$'s columns if the
    clamp-then-normalise scheme is ever replaced.
19. **L. Condat. *Fast projection onto the simplex and the $\ell_1$ ball.*
    Mathematical Programming 158, 2016.** https://doi.org/10.1007/s10107-015-0946-6
    Exact finite-time simplex projection. (J. Duchi, S. Shalev-Shwartz,
    Y. Singer, T. Chandra, ICML 2008, is the sorting-based original.)
20. **A. F. T. Martins, R. F. Astudillo. *From Softmax to Sparsemax: A Sparse
    Model of Attention and Multi-Label Classification.* ICML 2016.**
    https://proceedings.mlr.press/v48/martins16.html
    The Euclidean projection onto the simplex as a differentiable layer with
    an efficient Jacobian. *Take:* a differentiable, never-empty
    parametrisation of a vertex's weights, an alternative to R1 for the
    NaN alone that would change every iteration's numerics.
21. **P. Melchior, R. Joseph, F. Moolekamp. *Proximal Adam: Robust Adaptive
    Update Scheme for Constrained Optimization.* arXiv 1910.10094, 2019.**
    https://arxiv.org/abs/1910.10094
    The paper compskin cites for Adam plus proximal operators: a Euclidean
    prox after an adaptive step is inconsistent with Adam's scaled metric;
    take the prox in that metric. *Take:* background for any future
    projection change; not needed for R1.
22. **J. Yun, A. C. Lozano, E. Yang. *Adaptive Proximal Gradient Methods for
    Structured Neural Networks.* NeurIPS 2021.**
    https://proceedings.neurips.cc/paper/2021/hash/cc3f5463bc4d26bc38eadc8bcffbc654-Abstract.html
    Convergence for preconditioned proximal methods with non-convex
    regularisers, including a proximal Adam. *Take:* as 21.
23. **J. Bolte, S. Sabach, M. Teboulle. *Proximal alternating linearized
    minimization for nonconvex and nonsmooth problems.* Mathematical
    Programming 146, 2014.** https://doi.org/10.1007/s10107-013-0701-9
    Convergence of proximal alternating schemes on semi-algebraic non-convex
    sets. *Take:* the bilinear sparse problem is in scope.
24. **U. Evci, T. Gale, J. Menick, P. S. Castro, E. Elsen. *Rigging the
    Lottery: Making All Tickets Winners.* ICML 2020.** https://arxiv.org/abs/1911.11134
    Drop by magnitude, grow by gradient magnitude at constant sparsity.
    *Take:* the re-entry signal for idle joints (§5.5).
25. **G. Sokar, R. Agarwal, P. S. Castro, U. Evci. *The Dormant Neuron
    Phenomenon in Deep Reinforcement Learning.* ICML 2023.**
    https://proceedings.mlr.press/v202/sokar23a.html
    Detect and recycle dormant units. *Take:* detect-and-recycle as the
    template for idle joints in the anonymous path.
26. **B. H. Le, J. K. Hodgins. *Real-time Skeletal Skinning with Optimized
    Centers of Rotation.* ACM TOG 35(4), SIGGRAPH 2016.**
    https://doi.org/10.1145/2897824.2925959
    Per-vertex centres of rotation from rest pose and weights. *Take:*
    weighted-centre arithmetic for the §7 metric.
27. **L. Kavan, J. Doublestein, M. Prazak, M. Cioffi, D. Roble. *Compressed
    Skinning for Facial Blendshapes.* SIGGRAPH 2024.** Local copy
    `paper/compressed_skinning_for_facial_blendshapes.md`. §1 (noise init
    rationale, anonymous path only), §3.1 (Eq. 8, closure under addition,
    which R2 relies on), §4 (projections), §6 (importance weighting of
    vertices, E6).
28. **vastSolver 2 (vidiaApps, Maya plugin).**
    https://marketplace.autodesk.com/apps/c896f394-a280-4580-acf1-1303ea35f12c
    Not a paper. Solves joints and weights, weights only for an existing
    skeleton, or joint animation only for existing weights, with Laplacian
    weight smoothing. *Take:* the "existing weights" mode is E5; the "any
    joint hierarchy" support is what R2 adds.

## 11. Verification notes

- The Dem Bones function-level claims (label propagation, split, prune,
  implicit weight smoothing) were checked against the inline implementation
  in `DemBones.h`; the SSDR and Le & Deng 2014 PDFs themselves could not be
  fetched (size limit) and are summarised from their ACM abstracts and the
  Dem Bones code.
- The SPLOCS project page failed on a certificate error; its description
  follows the ACM abstract and the reference implementation.
- The private traces in `additional_context.md` were produced on data not
  in either repository; this document reports them only qualitatively.
- The experiments in §6.1 were run by the user on proprietary data on
  another machine; this document reports them qualitatively or as relative
  figures only. E7's negative result is taken at face value; its proxy
  (energy ratio) is coarser than the pivot geometry E8 measures, so the
  pivot question is open, not closed.
- §5's remaining expected effects (R1 to R4) are still predictions; §6.2
  lists the runs that test them.
