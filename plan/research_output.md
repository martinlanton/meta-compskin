# Joint-Placement Research Output

Revision 4. Sections 1 to 11 are Revision 3, written against commit `12ac666`
and kept as written; sections 12 to 15 were added against release 0.5.1,
and §13.12 to §13.17 with entries 74 to 98 on 14 November 2026.

*Changes from Revision 3 (this revision):* §1 to §11 are unchanged and remain
the record of what was predicted before the designs were built and measured.
**§12 says which of those predictions held and which were overturned; read it
before acting on anything in §5 to §9.** §13 adds the literature for the
directions the later work opened: choosing the non-zero coefficients, the
family of joint transforms, measuring the error the eye sees, fitting for how
the shapes are used, redundancy between shapes, constraints on the weights,
joint placement, objectives for the worst error, and hybrid representations.
§14 extends the bibliography (entries 29 onward) and §15 the verification
notes. The measurements behind §13, and the plan built on them, are in
[roadmap.md](roadmap.md).

Revision 3 was the deliverable requested in
[specs.md](specs.md) §9, with the failure traces from
[additional_context.md](additional_context.md) and the experiment results in
[experiment_results.md](experiment_results.md). Companion reading: the
annealing notes (`gradual_sparsity_annealing/research.md`) and the
geodesic-filter notes (`joints_geodesic_distance_filter/research.md`). Code
lines refer to `src/metacompskin/model_fit.py` at that commit unless another
file is named.

*Changes from Revision 1:* the NaN mechanism is now confirmed by the traces
in `additional_context.md` (§2) and the fix is folded into the design for problems #3 and #5
rather than treated as a separate guard; §3 corrects an overstatement about
"global deformation modes" (the transform basis spans rotations about any
pivot; what it changes is their sparsity cost); the centre of the document is
now a single coherent design, a *rig-aware mode* built on a zero-transform
root influence and a hierarchical, joint-centred transform basis (§5), which
serves #3, #5 and #4 at once; the zero-code experiments a review found
missing are listed (§6); the metric section gains a calibration step (§7);
the bibliography gains the hierarchy and sparse-simplex literature (§10).
*Changes from Revision 2:* experimental results are folded in. The NaN reproduces on an uncut head with a long single-stage run (§2.1),
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

### 2.1 What the traces show

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
at the tested budgets, but a tiny synthetic case shows it is possible.

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
- The traces in `additional_context.md` are reported only qualitatively.
- The experiments in §6.1 are reported qualitatively or as relative figures
  only. E7's negative result is taken at face value; its proxy
  (energy ratio) is coarser than the pivot geometry E8 measures, so the
  pivot question is open, not closed.
- §5's remaining expected effects (R1 to R4) are still predictions; §6.2
  lists the runs that test them.

## 12. What the experiments settled (Revision 4)

Sections 5 to 9 were predictions. Most of the design was then built (releases
0.2 to 0.5) and measured over several seeds and budgets, on the four public
sample heads with hand-authored joint sets of a few hundred joints. Results are given
qualitatively or as relative figures, as in §6.

| Revision 3 item | Outcome | Standing |
|---|---|---|
| R0, warm-up at $K_0 = M$ (§5.0) | Built. No collapse since, at any budget tried. Turning the warm-up off altogether cost a large share of the locality for a couple of percent of average error. | Kept. |
| R1, root influence (§5.1) | Built as specified: synthetic or designated joint, floor, exempt from the cut. No collapse and no accuracy cost (average error marginally better). The risk §5.1 named, moving vertices parking weight on the root, is real: by default about a third of the vertices carry some root weight. That is harmless there (fast-moving skin carries almost none) but it is the channel through which the first distance penalty failed (below). A designated joint used as the root must be allowed on every vertex, as the synthetic root is. | Kept; default with joint matrices. |
| R2, hierarchy and joint-centred pivot (§5.2) | **Overturned.** Solving through the hierarchy gave no locality gain over three budgets and cost from one to a dozen percent of average error. Joint-centred pivots cost accuracy at a fixed delta budget. | Opt-in only (`joint_parents`, `deltas_about_joints`). Do not re-propose as a default. E8 was never run. |
| R3, mask policy (§5.3: demote the mask or remove it) | **Overturned.** Without the mask no joint passes the §7 bar. A wide mask ($2K$) gave roughly a third of the pass rate of $M = K+2$ at equal error. $M = K+1$ gained a little locality but stranded more vertices on the root and cost accuracy. With a distance penalty on, the wide mask matches the tight one. | Default $M = \min(K+2, P-1)$. The hard mask is the locality mechanism; a penalty is an optional addition to it. |
| R3, proximity initialisation (§5.3) | **Confirmed, and the largest single lever.** The distance start alone doubled the pass rate, at the best average error of any arm. Alternative starts (sharper falloff, nearest joint plus a share on the rest, heat weights, a broader smooth start) all landed in the same band. | Default with joint matrices. Starting the transforms at zero instead of noise gives the same result and makes the solve independent of the seed. |
| R3, distance penalty (§5.3) | **The form written in §5.3 is degenerate once a root exists.** $\rho$ is zero on the root, so shrinking every joint weight while the transforms grow lowers the penalty at no cost in fit. Weight at the joints collapsed, joints travelled several times farther than the skin they drive, and the §7 pass rate, which leaves the root out, rose all the same. The corrected form penalises each vertex's *share* of joint weight on far joints; the last phase refits without the penalty inside the support it left; a guard returns moving skin stranded on the root to its joints, with a floor on the nearest one. | Released, off by default. Its strength is still being chosen. |
| R3, warm-up choice (E3) | Run. The warm-up stays (see R0). | Closed. |
| R4, weight Laplacian (§5.4) | Not built. | Open; see roadmap. |
| §5.5, idle joints | Not built. Not reported as a problem since the distance start. | Parked. |
| E5, fixed-weights bound | Run. A fixed smooth distance bind passes about four joints in ten at roughly twice the solver's average error; a fixed nearest-joint bind about six in ten at well over three times. **The reading rule was wrong:** the nearest-joint bind was taken as the locality ceiling, and a later penalty run passed more joints than it. With a root, a joint passes the §7 bar by holding only a little weight, close to itself. | The pass rate alone is not a sufficient measure (see below). |
| E6, importance weights | Not built. | Open; see roadmap. |
| E9, metric on existing outputs | Done, and now part of every run. | In use. |
| E10, anonymous-path fragility | The path without joint matrices did not collapse at the budgets tested. | Closed unless a failure appears. |
| §7, the bar | In use. Refinement 1 (calibrate each joint against a reference bind) is still open. | See below for the measures added beside it. |
| §8 #2, annealing with the mask | Not a general win: it improves locality for supplied joints and costs accuracy. | Available, not a default. |
| Residual-target Laplacian (a follow-up in the annealing notes, not in Revision 3) | Built. Smoothing the *error* instead of the result cut the worst error by half or more on the public heads, with the average error unchanged or better. The gain follows how sharp the target is (lip and eyelid edges), not how far it moves. | Default with joint matrices. |
| Pinning static skin | Built. Solve time falls with the share pinned; results do not change. | Default with joint matrices and a root. |
| Long budgets | The same locality at a hundred times the steps; a few percent of average error. | Closed. |
| Seeds | With the distance start, seeds differ by under one percent in average error. E4's caution applies to the noise start. | Closed for the joint path. |
| Cutting geometry (§2.4) | Confirmed: removing the mouth interior makes the rim markedly worse. | Keep the full head. |

Three lessons carry forward:

1. **A penalty or measure on the weights must survive the trade between weight
   and transform magnitude.** Weights and transforms enter the fit only as a
   product, so anything that looks at the weights alone can be satisfied by
   shrinking weights and growing transforms, with the root absorbing the
   difference. The same holds for measures: count the root in every share.
2. **The §7 pass rate is necessary, not sufficient.** The measures now read
   beside it are the weight a joint holds at its own position (mean weight on
   its nearest vertices, and the share of joints above 0.4, 0.5 and 0.6), how
   far a joint travels per unit of motion of the skin it drives, the root's
   share on moving and on fast-moving skin, the moving skin left entirely on
   the root, and the size of the joints' rotations.
3. **Structural predictions need a measurement before a build.** The
   hierarchy and the mask demotion both read well on paper and both failed.
   The cheapest check that would have caught them, an exact solve of one
   factor with the other fixed on a finished result, needs no training; the
   roadmap makes that a standard first step.

## 13. Literature for the next directions (Revision 4)

Each subsection gives what the literature offers, then what this repository
takes from it. Numbers in brackets refer to the bibliography (§10 for 1 to
28, §14 from 29). Codes such as Q1 or L2 are items of [roadmap.md](roadmap.md),
and "finding F3" is a measurement listed in its §3. The SIGGRAPH course
notes on skinning [72] are the general background for §13.1, §13.2 and §13.6.

### 13.1 The solve as sparse coding: exact refits and choosing the support

With the weights fixed, the joint deltas are an ordinary linear least-squares
problem. It separates per shape, every shape shares one dictionary (the
displacement field each coefficient of each joint produces), and the
Laplacian term only changes the metric. With the deltas fixed, the weights are
a convex quadratic programme over a simplex. The solver never uses either
fact: it takes Adam steps and projects.

- **Iterative hard thresholding and its refinements.** The projection onto
  the $L$ largest coefficients is iterative hard thresholding (Blumensath &
  Davies, reference 8 of the annealing notes). **Hard thresholding pursuit**
  [29] adds an exact least-squares fit on the chosen support at every
  iteration, which is what makes the support and the values consistent.
  Jain, Tewari and Kar [30] analyse the family, including "fully corrective"
  variants that refit after each selection, and show that projecting onto a
  sparsity level *looser* than the target gives guarantees the plain method
  lacks, which is the theory behind annealing a budget.
- **Choosing what to drop.** Magnitude is the crudest saliency. Optimal
  Brain Damage [31] ranks a coefficient by its squared value times the
  curvature of the loss along it; Optimal Brain Surgeon [32] uses the
  inverse Hessian, which gives the exact rise in a quadratic loss when one
  coefficient is removed *and the others are refitted*. Our loss is quadratic
  in the deltas, so OBS is exact here and costs one small inverse per shape.
  It matters because the coefficients of one joint are strongly correlated (a
  rotation about the mesh centre is almost a translation for a joint far
  from the centre), and neither magnitude nor curvature alone sees that.
- **Dictionary learning.** K-SVD [33] alternates sparse coding with
  dictionary updates; Batch-OMP [34] makes the coding step cheap when many
  signals share one dictionary by precomputing its Gram matrix, which is
  exactly our situation: one Gram matrix serves every shape.
- **Letting pruned coefficients come back.** Top-KAST [35] keeps the forward
  pass sparse but updates a larger set of coefficients in the backward pass,
  so entries just below the cut can grow and re-enter; movement pruning [36]
  ranks coefficients by how training moves them instead of by size. Both
  address the lock-in described in the annealing notes §3.

*Take.* Measured on finished results (findings F1 and F2): refitting both
factors exactly on the solver's own supports gains almost nothing, and
re-choosing the deltas' support by OBS, alternated with exact weights, gains
about a tenth of the loss and a twentieth of either error. **The optimiser is
not the bottleneck**, which closes "a better optimiser" as a quality lever
and leaves the paper's claim standing. What survives is small and cheap: an
OBS re-selection with an exact refit as a post-process (Q5), and the exact
solves as a measuring instrument (M4), for instance to price a constraint
before building it into the solver.

### 13.2 The family of joint transforms

The paper restricts each joint's delta to a linearised rotation plus a
translation (six coefficients) because that family is closed under blending,
and notes two things in passing: the general choice is a full $3\times4$
matrix, and the skinning shader must already accept arbitrary matrices,
because blended linearised rotations are not rotations (paper §1 and §3.1).

- James & Twigg [4] fit unconstrained affine proxy bones, which can squash
  and stretch. Kavan et al. [37] fit skinning to deformations that are not
  quasi-articulated (cloth, soft bodies; by extension, skin sliding over a
  skull) and get considerably higher accuracy there than methods that look
  for rigid parts.
- Le & Deng [1] chose rigid bones for compactness and for editing, not for
  accuracy. That argument matters less here: the joints' rest positions are
  authored, and the per-shape transforms are already non-rigid.
- Mukai's helper-bone work [38, 39, 40] fits extra bones to example shapes to
  correct a skeletal rig; the 2018 paper converts rigs by sampling them and
  fits *non-rigid* helper bones.
- Polar decomposition [41] splits any such matrix into a rotation and a
  stretch, which is how a rig builder would key rotation, scale and shear
  channels and how the size of the non-rigid part can be reported.
- **The runtime decides what the family is worth.** The paper's runtime
  blends the $3\times4$ deltas linearly and hands matrices to the shader.
  Engines' own animation systems usually store a joint as a translation, a
  rotation and a scale along its axes, and layer poses channel by channel. A
  linearised rotation of more than a few degrees is not representable there:
  it carries a stretch across its axis, and layering rotations is not adding
  matrices. Finding F7 measures what a finished result loses when stored
  that way.

*Take.* Adding the six symmetric coefficients (three scales, three shears) to
the basis keeps the family closed under blending and costs nothing at runtime
for the same number of non-zeros; the budget projection then chooses which
kind of coefficient each joint and shape deserves. With exact solves it is
the largest gain in accuracy found so far (finding F3), on two conditions.
The delta budget has to grow with it: at today's budget twelve kinds of
coefficient compete for the same non-zeros and the gain mostly vanishes; at
about twice the budget most of it is there. And the runtime has to take
matrices. Where it does not, the opposite direction applies: keep the
rotations small enough to survive the engine's channels (L3). Which of the
two holds is the first decision the roadmap asks for (M1, Q1).

### 13.3 Measuring the error the eye sees

MAE and MXE are position errors. Three strands of work say position error is
a poor proxy for what a viewer notices on a shaded surface.

- **How smooth the error is matters more than how large it is.** Karni &
  Gotsman [42] judge compressed geometry by combining the position error with
  the error of the geometric Laplacian. Sorkine et al. [43] show that
  low-frequency error is far less visible than high-frequency error, which
  disturbs normals and shading, and design a quantiser around that. Váša &
  Skala's STED [44], built from user studies on *animated* meshes, measures
  local change of edge lengths in space and time and correlates with
  perception far better than vertex distances. Sobolev training [45] is the
  general form of the idea: fit derivatives as well as values.
- **Our residual Laplacian is this idea.** `smoothing="residual"` minimises
  $\lVert E\rVert^2 + \alpha\lVert \mathbf{L}E\rVert^2$, a metric of the
  Karni-Gotsman kind on the error $E$. That explains why it won better than
  "it stops flattening creases" does, and it says what to tune $\alpha$
  against: a shading measure, not MAE. The paper's $\alpha$ values were
  chosen for the other form of the term and have not been revisited.
- **Normals of a skinned surface are not the skinned normals.** Tarini et
  al. [46] show that pushing the rest normal through the blended bone
  matrices is only accurate where the weights are constant around the
  vertex, and give a cheap correction that accounts for how the weights vary
  over the surface. A compressed-skinning face is the worst case for the
  standard formula: hundreds of joints that mostly translate, with the
  surface's change of orientation carried almost entirely by how the weights
  vary. Finding F6 measures it: engine-style skinned normals recover almost
  none of the change of normal.
- **What engines do.** Standard GPU skinning transforms the stored normal
  and tangent with the blended matrices. Unreal Engine's skin cache does so
  in a compute shader and can instead recompute tangents for chosen mesh
  sections [73], an option used for meshes whose shape changes beyond what
  the bones describe, such as morph targets. On any engine the alternatives
  are the correction of [46] in the vertex shader, or normals derived per
  pixel from the skinned positions.
- **Normals inside the decomposition.** Zhao et al. [47, 48] build facial
  rigs by skinning decomposition from 4D captures, with a soft constraint on
  joint positions and a fixed weight sparsity, and then add normal
  constraints to remove wave-like surface artefacts that a position-only fit
  leaves.
- **Where the eye looks.** Mesh saliency [49] derives per-vertex importance
  from curvature at several scales; the paper's own conclusion proposes
  importance weights so that the mouth interior counts less and the
  nasolabial fold more.

*Take.* Report normal error in degrees beside MAE and MXE, both for normals
recomputed from the skinned positions and for engine-style skinned normals
(M2), and find out which of the two the target engine uses. A normal term in
the fit of the deltas was priced with exact solves (finding F6): it removes
about a third of the skinned-normal error for a large loss in position
accuracy, because joint rotations blended by smooth weights cannot follow
fine changes of orientation. The fix belongs in the engine: recompute the
normals, or apply the correction of [46] (Q4). Re-tune $\alpha$ under the
residual form against the normal measure (Q3); add importance weights (Q7).

### 13.4 Fitting for how the shapes are used

The solver minimises the error of every shape taken alone at full strength.
A face is never shown that way: the rig turns a few dozen controls into
activations of several shapes at once, correctives fire with their parents,
and the errors of active shapes add (finding F4).

- Lewis et al. [50] is the reference on how correctives, in-betweens and
  combination shapes are built and driven.
- Le et al. [51] compress a delta-mush deformer into two skinning layers by
  minimising the error over a *continuous set of example poses*, and show
  that this reduces to an ordinary linear matrix factorisation. The same
  algebra applies here with the blendshape activations as the pose: the
  expected squared error of a frame is $\operatorname{tr}(E^{\top}\Sigma E)$,
  with $\Sigma$ the second-moment matrix of the activations.
- Weighted low-rank approximation [52] is the general problem and has no
  closed form, which is not an obstacle for a gradient solver.
- Where no animation library exists, the activations can be sampled from the
  rig itself, as rig-conversion and rig-approximation work does to build its
  example sets [39, 53, 54, 55].

*Take.* Measure the error on frames of real or sampled animation (M3); then
weight the loss by $\Sigma$ (Q2). The cheaper idea, spending the delta
budget by how much each shape is used while leaving the loss alone, was
priced with exact solves and gains little: the frames improve by a few
percent and the worst single shape gets much worse.

### 13.5 Redundancy between shapes

The delta budget is spent per shape, as if shapes were unrelated. They are
not (finding F5): a few dozen directions carry almost all of the energy of a
few hundred shapes, and many correctives are close to a signed multiple of
another shape.

- Seo et al. [56] compress the blendshape matrix itself with hierarchically
  semi-separable blocks after reordering, exploiting this kind of structure.
- Two-layer skinning [57], and its delta-mush successor [51], add a small
  layer of virtual bones so that the expensive factor is shared.
- SPLOCS [11] re-expresses a shape set in a basis in which it is sparse.

*Take.* A sparse change of basis in shape space, $\mathbf{B} = \mathbf{T}
\tilde{\mathbf{B}}$ with $\mathbf{T}$ holding one or two entries per shape
(a shape stored as a multiple of its nearest relative plus a remainder),
lets one set of non-zeros serve several shapes. At runtime it is one extra
sparse product on the activations. Because finding F2 shows that the delta
budget is not what limits accuracy at the budgets in use, this is first a
route to a smaller budget at equal error, and a quality lever only at tight
budgets (Q6).

### 13.6 Constraints on the weights: anchors, monotone falloff, symmetry

- Bounded biharmonic weights [7] obtain intuitive handle weights from three
  ingredients: each weight is *pinned to one at its own handle*, bounded, and
  smooth elsewhere. Jacobson et al. [58] add what smoothness alone does not
  give: control over where the local maxima and minima are, so that a
  function has no extrema other than the ones asked for. Wang et al. [59]
  obtain deformation subspaces of the same family from sparse linear solves.
- The facial-rig literature that keeps joints where a rigger expects them
  does so with explicit priors rather than penalties on the fit: a soft
  constraint on joint positions with fixed sparsity [47], a hand-designed
  joint rig with a learned skinning-weight generator [60], weights
  supervised by a segmentation of the face into regions [61].
- Monotone constraints have an exact, fast projection (pool adjacent
  violators, [62, 63]). A weight vector that is non-increasing in the
  joints' distance order is also a linear re-parametrisation of a point on a
  simplex, so the constraint needs no new machinery.

*Take.* The solver's locality tools are a hard mask (which joints may be
used) and a soft penalty (a preference). Neither says how much weight a joint
must hold at its own position, which is the criterion riggers actually state.
A *floor* would: every vertex keeps a guaranteed share on its nearest joint,
largest at the joint and falling to zero where the next joint is as close
(L1; the stranding guard already implements such a floor for a few
vertices). Priced with exact solves on a finished result, it is expensive
(finding F12): the vertices a floor binds are the ones whose nearest joint
moves differently from them, and the worst error triples for a small gain in
weight at the joint. The pinning in bounded biharmonic weights works because
nothing else is being fitted; here the fit pulls the other way. What the same
exact solves show instead is that the *start* decides how local the result
is (finding F13, L2). Monotone falloff (L6), mirrored weights for a symmetric
layout (L4) and the weight Laplacian of §5.4 (L5) are the remaining members
of the family, and each can be priced the same way before it is built. Any
constraint on the weights alone falls under lesson 1 of §12: read it
together with joint travel and the root's share.

### 13.7 Where the joints should sit

Joint positions do not enter the fit (the transforms are free about any
pivot); they enter through the mask, the start and the penalties. So "the
best position for a joint" means the position that makes its nearest skin the
skin it should drive.

- Dem Bones' `bindUpdate` [3], SPLOCS centre updates [11] and Le & Deng
  [2] move each bone to the centre of what it ended up driving. Iterating
  "assign skin to the nearest joint, move the joint to the centre of its
  skin" is Lloyd's algorithm, whose fixed points are centroidal Voronoi
  tessellations [64].
- Zhao et al. [47] keep supplied joints near a template with a soft
  constraint while letting them adapt to a subject, which is the bounded
  drift of §8 #6.

*Take.* First a report with no solver change: for each joint, the vector
from the joint to the centre of the skin it drives, and a map of moving skin
that is far from every joint (L8). The scalp gap found on one of the public
heads would have been flagged by such a map before any solve. A
bounded-drift loop is the same report applied and re-solved, and stays a
policy question.

### 13.8 Objectives for the worst error

- The paper's own route is an $L^p$ norm with large $p$ (§4.1), which it
  found to need more capacity and many more iterations; the annealing notes
  (§10) add that raising $p$ silently weakens the Laplacian term.
- Iteratively reweighted least squares [65] is the standard way to reach
  other norms by re-weighting an $L^2$ problem, which keeps every step well
  conditioned.
- Tail objectives such as conditional value-at-risk [66] minimise the mean
  of the worst fraction of the errors: every vertex in the tail gets a
  gradient, unlike the maximum, and the rest of the mesh is left alone,
  unlike a high power.

*Take.* The worst error barely moved under any relaxation of the budgets
(finding F2), so it is not bought with capacity; it has to be asked for in
the objective, or removed by other means (Q3, Q9).

### 13.9 Hybrid representations

EigenSkin [67] adds a low-rank displacement correction on top of skinning,
and rig-approximation networks [53, 68] split a linear part from a learned
non-linear remainder. Measured on our residual (finding F8), a low-rank
correction does not pay: the remaining error is spread over many directions.
A *sparse* correction does: storing the largest few hundred to few thousand
per-vertex, per-shape errors exactly removes about half of the worst error.
It leaves pure skinning, so it is an option for pipelines that can afford a
small morph-target pass, not a default (Q9).

### 13.10 Distances on the surface

The mask, the start and the penalty all use Dijkstra distances along mesh
edges, which over-estimate on coarse or anisotropic meshes. The heat method
[69] gives smooth geodesic distances from a pair of prefactored sparse
linear systems and is the standard replacement. Low priority: no result so
far points at the distance estimate.

### 13.11 Learned facial rigging, for context

Neural Face Rigging [70], Neural Face Skinning [61] and RigAnyFace [71]
predict rigs or deformations for unseen meshes from data. They solve a
different problem (no shapes are given) and are not candidates here, but
they confirm two choices made above: weights tied to regions of the face are
treated as a prior worth supervising, and FACS-style controls are kept as
the interface.

### 13.12 Where the error comes from: three gaps

A finished solve misses the shapes for three separable reasons, and every
method below acts on one of them only.

1. **The representation gap**: what the joints can do at all. It is set by
   the family of transforms (six coefficients today), by which joints a
   vertex may use (the mask, $K$) and by the basin the weights end in. With
   no budget at all, today's family on today's weight pattern still leaves
   MAE about 0.007 mm on one public head; that floor does not move with $L$.
2. **The budget gap**: what the cut to $L$ non-zeros costs against no cut.
   With six coefficients at today's $L$ it is about a fifth of the loss
   (roadmap F2); with twelve coefficients at today's $L$ it is most of the
   loss (F3).
3. **The optimiser gap**: what Adam and the projection leave against the
   best fit on the same supports. Under 1% of the loss (F1).

So a better optimiser, penalty or ramp can only recover the budget gap and
the part of the representation gap that is a matter of basin (the weights'
support, F13). It cannot reach accuracy the unconstrained family lacks. This
is why "freeing budget" helps much less than intuition suggests when the
budget is not what binds, and a great deal when it is (§13.13).

### 13.13 The pivot, and counting the budget in blocks

A joint's delta is a $3\times4$ block $[\mathbf{M}\,|\,\mathbf{t}]$ written
about the head centre. Writing it about the joint at $\mathbf{p}$ keeps
$\mathbf{M}$ and replaces the translation by
$\mathbf{t} + \mathbf{M}\mathbf{p}$: the same motion, the same span, a
different set of coordinates. Sparsity is not invariant under a change of
coordinates, so the pivot changes what a budget of $L$ coefficients can
buy. In the language of rigid motion [76] a linearised rigid delta is a
twist, its frame change is the adjoint map, and the pivot that makes a twist
sparsest lies on its screw axis, which is what E8 proposed to measure.

**Measured (one head, CPU, weights fixed from today's default result, each
delta budget chosen by Optimal Brain Surgeon saliency with exact refits):**

| Family, budget | About the centre: loss, MAE, MXE | About each joint: loss, MAE, MXE |
|---|---|---|
| 6, no budget | identical | identical |
| 6, half of today's $L$ | 3.27e-5, 0.0116 mm, 1.94 mm | 2.73e-5 (−17%), 0.0106 mm (−8%), 0.84 mm (−57%) |
| 6, today's $L$ | 2.13e-5, 0.0082 mm, 0.79 mm | 2.04e-5 (−4%), 0.0079 mm (−4%), 0.76 mm (−3%) |
| 6, twice $L$ | 1.92e-5, 0.0070 mm, 0.76 mm | the same within 0.5% |
| 12, today's $L$ | 1.85e-5, 0.0082 mm, 2.09 mm | 1.51e-5 (−18%), 0.0073 mm (−10%), 0.64 mm (−69%) |
| 12, twice $L$ | 1.15e-5, 0.0055 mm, 0.63 mm | 1.11e-5 (−4%), 0.0053 mm (−4%), 0.63 mm |

About the joint, fewer rotation coefficients are kept (43% of the budget
against 49% at today's $L$; 32% against 49% at half): rotations
about the centre were partly paying for translations. With six
coefficients the pivot recovers about 40% of the budget gap at both budgets
measured. That is large where the budget binds (half $L$; twelve
coefficients at today's $L$, whose per-shape MXE falls from 2.7 times
today's to below it) and small where it
does not (six coefficients at today's $L$, whose whole budget gap is a tenth
of the loss). It does not free half the budget: a joint at the head centre
loses nothing, many deltas are translations that cost the same in either
frame, and the remaining error is mostly the representation gap. Limits: one
head, one seed, weights from a centre-pivot solve (a bias against the joint
pivot), OBS selection rather than the solver's magnitude cut. A solve with
joint pivots is the next step (roadmap Q10).

- **Group sparsity.** Counting non-zero *blocks* (one per joint and shape)
  instead of coefficients makes the budget independent of the pivot, since
  a block is zero in every frame or in none. Block-sparse recovery has its
  own greedy method, block OMP [74], with guarantees from a block coherence;
  the convex route is the mixed $\ell_2/\ell_1$ norm (group lasso), surveyed
  with its proximal operators and working-set solvers in [75]. *Take:* what
  the budget should count is a question about the runtime (roadmap D10): a
  runtime that stores and blends whole $3\times4$ blocks pays per block, and
  then twelve coefficients cost no more storage than six per kept block.
  The block saliency for OBS is the same formula with a small inverse per
  block, so exact solves can price it before the solver changes.

### 13.14 Choosing the support: beyond hard thresholding

The solver's delta budget is iterative hard thresholding by magnitude
(§13.1). Three families do better on the $L_0$ problem in the literature,
and all three fit the solver's structure.

- **Local combinatorial search.** Hazimeh and Mazumder [77] solve
  $L_0$-regularised least squares by coordinate descent and then by local
  swaps (drop one coefficient, add another, keep the swap if the loss falls),
  and report that the swaps give clearly better supports than coordinate
  descent or thresholding alone. *Take:* the natural extension of Q5's
  finishing step: our loss is quadratic in the deltas, so the exact change of
  a swap is a rank-two update of one shape's small system.
- **Exact and nonconvex penalties.** The trimmed lasso [78] penalises the sum
  of all but the $L$ largest magnitudes: it is zero exactly when the budget
  is met, so it turns the hard cut into a penalty that gradient steps can
  follow, and its strength can be ramped. The minimax concave penalty [79]
  and reweighted $\ell_1$ [80] are the standard nonconvex penalties that
  shrink small coefficients without biasing large ones, which plain $\ell_1$
  does. *Take:* a smooth route to the same budget, worth trying only where
  the budget binds (twelve coefficients at today's $L$, or a smaller $L$).
- **Smoothed $L_0$ with a ramp.** SL0 [81] replaces the count of non-zeros by
  a sum of Gaussian bumps of width $\sigma$ and lowers $\sigma$ step by step,
  from a smooth, nearly convex problem to the count itself. It is graduated
  non-convexity [82] applied to sparsity. *Take:* the annealing notes ramp
  the *budget* (how many survive); these ramp the *shape of the penalty*
  (how sharply survival is decided), which the solver has never tried.

### 13.15 Solving the two factors together: variable projection

The fit $\mathbf{A}\approx\mathbf{B}\mathbf{C}$ is a separable least-squares
problem: for fixed weights the deltas are linear. Variable projection
eliminates them exactly and optimises the weights on the reduced objective,
usually by Gauss-Newton or Levenberg-Marquardt. On matrix factorisation it
converges from far more starting points than joint optimisation or plain
alternation [83], and [84] explains why: the reduced problem has a wider
basin. iPALM [85] is the principled form of what the solver does today
(alternating proximal steps on two blocks, here with momentum), with
convergence to a critical point for nonconvex, nonsmooth problems such as
sparse factorisation.

*Take.* F1 says the optimiser finishes its own basin; F13 says the basin is
chosen early and decides locality. Variable projection is the one method in
this list that changes *which* basin is reached rather than how well it is
finished, so it belongs with the locality work (L2) as much as with quality.
The exact delta solve it needs is the one used for the findings. Price it on the
CPU first: variable projection on the weights from the distance start,
against the solver's result, on the loss, MAE, MXE and the locality
measures. The sparsity constraints make the reduced objective piecewise
smooth, so expect to fix the delta support between Gauss-Newton steps.

### 13.16 Choosing alpha from the head: ideas from inverse problems and geometry

`constants.py` keeps one alpha per head (10 for the two coarse heads, 50 for
the two dense ones) with a note asking whether it could be derived from mesh
density and sharpness. Three fields answer parts of that question.

- **Why alpha depends on the mesh at all.** The solver's Laplacian is the
  uniform one: each vertex minus the mean of its neighbours. On a smooth
  field it returns about $h^2/4$ times the continuous Laplacian, with $h$
  the edge length, so the smoothness term on smooth motion shrinks like
  $h^4$ against the data term when the mesh is refined. On per-vertex noise
  it does not shrink at all. One alpha therefore means different things on
  a coarse and a dense head, and different things for smooth and sharp
  errors on the same head. This is this document's reading of the
  operator, consistent with the table (the dense heads need the larger
  alpha, although the note in the file says the opposite).
- **A resolution-independent operator.** The cotangent Laplacian with
  Voronoi areas [88] converges to the Laplace-Beltrami operator as the mesh
  is refined. With it, $\alpha\,\lVert \mathbf{M}^{-1}\mathbf{L}\,\mathbf{r}\rVert^2_{\mathbf{M}}$
  has units: alpha is a length to the fourth power, and
  $\ell = \alpha^{1/4}$ is the wavelength at which smoothing and data
  weigh the same. The manifold-harmonics view [89] makes that literal: the
  term is a low-pass filter on the residual with cut-off near $\ell$.
- **What the table encodes (measured on the four public heads).** Edge
  length $h$ is about 3.6 mm on the coarse heads and 1.7 mm on the dense
  ones; joints sit about 10 mm apart on all four. The curvature the shapes
  add to the skin (the Laplacian of each shape's deltas, cotangent form,
  worst shape per vertex) has its sharpest tenth below a radius of 3.3 mm
  (`jupiter`), 4.4 mm (`aura`), 1.9 mm (`bowen`) and 2.6 mm (`proteus`):
  the sharpest creases are one or two edges wide. On each head's actual
  residual, the cotangent smoothness term with
  $\ell = k \times$ that radius equals today's term at $k \approx 0.40$
  (`aura`, alpha 10) and $k \approx 0.48$ (`proteus`, alpha 50). One $k$
  therefore reproduces both table values within about 20%: the table is,
  in effect, a rule in mesh density and shape sharpness. Today's alpha
  divided by three, which lowered MAE by about 5% on all four heads,
  corresponds to $k \approx 0.3$ to $0.35$. A sweep of $k$ with the
  cotangent operator is running on the GPU.
- **Not the joint spacing.** A cut-off at the joint spacing (about 10 mm)
  would be a smoothing length three to five times today's, an alpha
  hundreds of times larger; a much larger alpha was tried and closed
  (§12). The smoothing acts at the scale of the sharpest shapes, not of the
  joints.
- **Choosing alpha from the data.** Regularised least squares has standard
  rules for its parameter: generalised cross-validation [86] picks the value
  that best predicts left-out data, and the L-curve [87] picks the corner of
  the curve of residual size against regularisation size. Both are cheap
  with the exact delta solve (weights fixed), so they can
  calibrate the joint-spacing rule above on the four heads.
- **Sharpness, and its direction.** The worst errors sit where the target
  is sharp (F11). Three refinements, cheapest first. (i) Per-vertex alpha,
  lower where the shapes are sharp, from the same curvature measure: one
  product in the smoothness term. (ii) Principal directions: Taubin's
  curvature tensor [96] gives, per vertex, the principal curvatures and
  their directions as the eigenvalues and eigenvectors of a small symmetric
  matrix, in linear time. The line-drawing work [97] follows the first
  principal direction because it carries the shape; a crease (a lip line,
  an eyelid fold, a wrinkle) is sharp across and smooth along. An
  anisotropic Laplacian that smooths the residual along the second
  principal direction and little across the first keeps the crease, as
  anisotropic mesh filtering does [98]. Applied to the curvature each shape
  adds, this is the eigenanalysis route from the line-drawing papers to
  alpha. (iii) Edge-preserving regularisation (total variation [90])
  penalises the magnitude of variation, not its square, and keeps sharp
  features without choosing directions, at the cost of a non-smooth term.

### 13.17 Moving the joints before the solve

The owner allows a first pass that moves the supplied joints within a bound
(5 mm by default), symmetrically, before the final solve (roadmap L9).

- **For tweakers, the centre of the skin they drive.** A tweaker's skin
  mostly slides, so the natural target is the motion-weighted centre of
  that skin, kept at the joint's depth under the surface. Repeating
  solve-and-move is Lloyd's iteration for a centroidal Voronoi tessellation
  (entry 64), the standard way to spread sites evenly over what they cover.
  The bound is best relative: a fraction of the distance to the nearest
  other joint lets sparse joints (the neck) move further than dense ones
  (the lips), and below one half no two joints can meet. A prototype is
  measured in the roadmap (L9).
- **For pivot joints, the centre of rotation.** The jaw and the eyelids
  rotate, and the centre of their skin is the wrong target (the prototype
  wants to move the jaw by 34 to 39 mm). Biomechanics estimates joint
  centres from the motion of skin markers: SCoRE [91] fits a centre fixed
  in both segments, and the least-squares sphere fit of Gamage and Lasenby
  [92] is closed-form. Applied to a joint's skin over the shapes that rotate
  it, either gives a pivot to move towards; both are known to be noise
  sensitive for small ranges of motion, so only the large rotators qualify.
- **Symmetry.** Joints are paired by their L_ and R_ prefixes, which rigging
  naming conventions provide; symmetry detection [93] is not
  needed.
- **Where to add a joint, and how many.** Choosing joint positions is a
  sensor-placement problem: choose locations that best explain a field
  sampled elsewhere. Greedy selection by mutual information has a
  constant-factor guarantee [94], and graph signal processing gives
  sampling sets for band-limited signals on a mesh graph [95]. *Take:* the
  basis for a separate tool that suggests where a joint is missing (the
  coverage gap of §2 of the roadmap), not for moving existing ones.

## 14. Annotated bibliography, continued (Revision 4)

Same form as §10: what the work does, then what this repository takes from
it. Numbering continues from §10.

29. **S. Foucart. *Hard Thresholding Pursuit: An Algorithm for Compressive
    Sensing.* SIAM Journal on Numerical Analysis 49(6), 2011.**
    https://doi.org/10.1137/100806278
    Iterative hard thresholding with an exact least-squares fit on the
    selected support at every iteration. *Take:* support and values should be
    made consistent by a refit; the template for Q5.
30. **P. Jain, A. Tewari, P. Kar. *On Iterative Hard Thresholding Methods for
    High-dimensional M-Estimation.* NeurIPS 2014.**
    https://arxiv.org/abs/1410.5137
    Analysis of hard-thresholding methods under realistic conditions,
    covering HTP, CoSaMP, subspace pursuit and fully corrective variants.
    *Take:* the theory behind refitting after selection and behind working at
    a looser sparsity than the target.
31. **Y. Le Cun, J. S. Denker, S. A. Solla. *Optimal Brain Damage.* NIPS
    1989.** https://papers.nips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html
    Prune by a saliency built from second derivatives instead of by
    magnitude. *Take:* squared value times curvature as the cheap saliency.
32. **B. Hassibi, D. G. Stork. *Second order derivatives for network
    pruning: Optimal Brain Surgeon.* NIPS 1992.**
    https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html
    Saliency from the inverse Hessian, with the remaining weights adjusted
    when one is removed; outperforms magnitude pruning and Optimal Brain
    Damage. *Take:* exact for our quadratic delta problem; the selection rule
    of Q5 and of the measuring instrument M4.
33. **M. Aharon, M. Elad, A. Bruckstein. *K-SVD: An Algorithm for Designing
    Overcomplete Dictionaries for Sparse Representation.* IEEE Transactions
    on Signal Processing 54(11), 2006.** Cited by the compskin paper.
    Alternates sparse coding with dictionary updates. *Take:* the frame for
    reading the solve as dictionary learning with a structured dictionary
    (the weights) and globally sparse codes (the deltas).
34. **R. Rubinstein, M. Zibulevsky, M. Elad. *Efficient Implementation of the
    K-SVD Algorithm using Batch Orthogonal Matching Pursuit.* Technion
    technical report CS-2008-08, 2008.**
    https://csaws.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf
    Sparse coding of many signals over one dictionary through its
    precomputed Gram matrix. *Take:* all shapes share one Gram matrix, so
    exact per-shape solves cost almost nothing.
35. **S. Jayakumar, R. Pascanu, J. Rae, S. Osindero, E. Elsen. *Top-KAST:
    Top-K Always Sparse Training.* NeurIPS 2020.**
    https://proceedings.neurips.cc/paper/2020/hash/ee76626ee11ada502d5dbf1fb5aae4d2-Abstract.html
    Sparse forward pass, a somewhat larger set updated in the backward pass,
    so the mask can change during training. *Take:* the mechanism for
    re-entry if lock-in is ever shown to cost accuracy; not needed now (F1).
36. **V. Sanh, T. Wolf, A. Rush. *Movement Pruning: Adaptive Sparsity by
    Fine-Tuning.* NeurIPS 2020.** https://arxiv.org/abs/2005.07683
    Rank weights by how training moves them, not by their size. *Take:* as
    35.
37. **L. Kavan, R. McDonnell, S. Dobbyn, J. Žára, C. O'Sullivan. *Skinning
    Arbitrary Deformations.* I3D 2007.**
    https://dcgi.fel.cvut.cz/publications/2007/kavan-i3d-sad/
    Skinning approximations of deformations that are not quasi-articulated,
    with transformations fitted in the least-squares sense. *Take:* skin is
    such a deformation; evidence for non-rigid joint transforms (Q1).
38. **T. Mukai. *Building Helper Bone Rigs from Examples.* I3D 2015.**
    Helper bones and their weights fitted to example shapes to correct a
    skeletal rig. *Take:* supplied joints plus fitted correction, the same
    setting as ours.
39. **T. Mukai. *Sampling-based Rig Conversion into Non-rigid Helper Bones.*
    Proc. ACM Comput. Graph. Interact. Tech. 1(1), 2018.**
    https://dblp.org/rec/journals/pacmcgit/Mukai18
    Converts a rig into a helper-bone rig from sampled poses, with non-rigid
    helper bones. *Take:* scale and shear on fitted bones (Q1); sampling the
    rig as the source of training poses (Q2).
40. **T. Mukai, S. Kuriyama. *Efficient Dynamic Skinning with Low-Rank Helper
    Bone Controllers.* ACM TOG 35(4), SIGGRAPH 2016.**
    Helper-bone rigs driven by low-rank controllers. *Take:* background for
    38 and 39.
41. **K. Shoemake, T. Duff. *Matrix Animation and Polar Decomposition.*
    Graphics Interface 1992.**
    https://graphicsinterface.org/wp-content/uploads/gi1992-30.pdf
    Polar decomposition as the way to extract the rotation of a matrix.
    *Take:* how to report and key the rotation and the stretch of an affine
    joint delta (Q1, L3).
42. **Z. Karni, C. Gotsman. *Spectral Compression of Mesh Geometry.*
    SIGGRAPH 2000.** https://dblp.org/rec/conf/siggraph/KarniG00
    Geometry compression in the eigenbasis of the mesh Laplacian, judged
    with a visual metric rather than position error alone. *Take:* the
    residual Laplacian term is a metric of this kind (M2, Q3).
43. **O. Sorkine, D. Cohen-Or, S. Toledo. *High-Pass Quantization for Mesh
    Encoding.* SGP 2003.** https://doi.org/10.2312/SGP/SGP03/042-051
    Quantising Laplacian coordinates moves the error to low frequencies,
    which are much less noticeable than the high-frequency error that
    disturbs normals. *Take:* why smooth error is the right kind of error.
44. **L. Váša, V. Skala. *A Perception Correlated Comparison Method for
    Dynamic Meshes.* IEEE TVCG 17(2), 2011.**
    https://doi.org/10.1109/TVCG.2010.38
    STED: local change of edge lengths in space and time, derived from
    subjective experiments, correlating with opinion far better than
    position metrics. *Take:* a candidate measure for animated results (M2,
    M3).
45. **W. M. Czarnecki, S. Osindero, M. Jaderberg, G. Swirszcz, R. Pascanu.
    *Sobolev Training for Neural Networks.* NeurIPS 2017.**
    https://arxiv.org/abs/1706.04859
    Fit target derivatives as well as target values. *Take:* the general
    statement of fitting positions and their differences together.
46. **M. Tarini, D. Panozzo, O. Sorkine-Hornung. *Accurate and Efficient
    Lighting for Skinned Models.* Computer Graphics Forum 33(2),
    Eurographics 2014.** https://doi.org/10.1111/cgf.12330
    The usual skinned normal is accurate only where the weights are constant
    around a vertex; a GPU method that accounts for the variation of the
    weights over the surface, for linear blend and dual quaternion skinning,
    with open-source shaders. *Take:* the engine-side fix for F6 (Q4).
47. **Z. Zhao, D. Weng, H. Guo, J. Hou, J. Zhou. *Facial Auto Rigging from
    4D Expressions via Skinning Decomposition.* ACM Multimedia 2023.**
    https://doi.org/10.1145/3581783.3612934
    Facial rigs from a rigging template and a 4D expression sequence, with a
    soft constraint on joint positions and a fixed sparsity on the weights.
    *Take:* the closest published work to this repository's joint path;
    bounded joint drift as a soft constraint (L8).
48. **Z. Zhao, Y. Cheng, D. Weng, Z. Tian. *Auto Facial Rigging of Digital
    Humans Based on Normal Constrained Skinning Decomposition.* IEEE ISMAR
    Adjunct 2024.**
    https://pure.bit.edu.cn/en/publications/auto-facial-rigging-of-digital-humans-based-on-normal-constrained/
    Adds normal constraints to the decomposition of 47 to prevent wave-like
    surface artefacts. *Take:* precedent for a normal term in the loss (Q4).
49. **C. H. Lee, A. Varshney, D. W. Jacobs. *Mesh Saliency.* ACM TOG 24(3),
    SIGGRAPH 2005.** https://www.cs.umd.edu/~varshney/papers/mesh_saliency_sig05.pdf
    Regional importance from centre-surround differences of mean curvature.
    *Take:* an automatic default for importance weights (Q7).
50. **J. P. Lewis, K. Anjyo, T. Rhee, M. Zhang, F. Pighin, Z. Deng. *Practice
    and Theory of Blendshape Facial Models.* Eurographics State of the Art
    Reports 2014.** https://doi.org/10.2312/egst.20141042
    The survey of blendshape practice, including correctives and
    in-betweens. *Take:* how shapes combine at runtime, which is what M3 and
    Q2 measure and fit.
51. **B. H. Le, K. Villeneuve, C. Gonzalez-Ochoa. *Direct Delta Mush Skinning
    Compression with Continuous Examples.* ACM TOG 40(4), SIGGRAPH 2021.**
    https://doi.org/10.1145/3450626.3459779
    Compresses a delta-mush model into a small one driving virtual bones plus
    a skinning layer, minimising the error over an uncountable set of example
    poses, which reduces to linear matrix factorisation. *Take:* the pose
    distribution enters as a quadratic form (Q2); two-layer structure (Q6).
52. **N. Srebro, T. Jaakkola. *Weighted Low-Rank Approximations.* ICML
    2003.** https://people.csail.mit.edu/tommi/papers/SreJaa-icml03.pdf
    Factorisation under a weighted error has no closed form and needs
    iteration. *Take:* background for Q2 and Q7.
53. **S. W. Bailey, D. Omens, P. DiLorenzo, J. F. O'Brien. *Fast and Deep
    Facial Deformations.* ACM TOG 39(4), SIGGRAPH 2020.**
    https://doi.org/10.1145/3386569.3392397
    A learned approximation of a film facial rig, with an extension for
    high-frequency detail such as wrinkles; cited by the compskin paper as
    future work. *Take:* rig sampling as data; the separate treatment of
    fine detail (Q9).
54. **D. Holden, J. Saito, T. Komura. *Learning an Inverse Rig Mapping for
    Character Animation.* SCA 2015.** https://doi.org/10.1145/2786784.2786788
    Regression between rig controls and deformation learned from example
    animation. *Take:* animator-made examples as the pose distribution (Q2).
55. **S. Radzihovsky, F. de Goes, M. Meyer. *FaceBaker: Baking Character
    Facial Rigs with Machine Learning.* SIGGRAPH 2020 Talks.**
    https://graphics.pixar.com/library/FaceBaker/paper.pdf
    Approximates facial rig deformations to cut rig computation; cited by
    the compskin paper. *Take:* as 53.
56. **J. Seo, G. Irving, J. P. Lewis, J. Noh. *Compression and Direct
    Manipulation of Complex Blendshape Models.* ACM TOG 30(6), SIGGRAPH Asia
    2011.** https://doi.org/10.1145/2070781.2024198
    Compresses the blendshape matrix with a hierarchically semi-separable
    representation after block reordering. *Take:* shape sets are highly
    structured; evidence for Q6.
57. **B. H. Le, Z. Deng. *Two-layer Sparse Compression of Dense-Weight Blend
    Skinning.* ACM TOG 32(4), SIGGRAPH 2013.**
    https://doi.org/10.1145/2461912.2461949
    Dense skinning weights compressed through a layer of virtual bones,
    described elsewhere as dictionary learning on the weights. *Take:* the
    two-layer pattern applied to the other factor (Q6).
58. **A. Jacobson, T. Weinkauf, O. Sorkine. *Smooth Shape-Aware Functions
    with Controlled Extrema.* Computer Graphics Forum 31(5), SGP 2012.**
    https://doi.org/10.1111/j.1467-8659.2012.03163.x
    Smoothness energies with the locations of maxima and minima prescribed
    and all other local extrema prevented. *Take:* weights that only fall
    away from their joint (L6).
59. **Y. Wang, A. Jacobson, J. Barbič, L. Kavan. *Linear Subspace Design for
    Real-Time Shape Deformation.* ACM TOG 34(4), SIGGRAPH 2015.**
    https://users.cs.utah.edu/~ladislav/wang15linear/wang15linear.html
    Deformation subspaces unifying skinning and generalised barycentric
    coordinates from sparse linear solves on a Laplacian energy. *Take:* a
    source of reference weights for a prior or a start (L7).
60. **N. Vesdapunt, M. Rundle, H. Wu, B. Wang. *JNR: Joint-based Neural Rig
    Representation for Compact 3D Face Modeling.* ECCV 2020.**
    https://arxiv.org/abs/2007.06755
    A face model built on a hand-designed joint rig with a learned
    skinning-weight generator, far smaller than blendshape models of similar
    capacity. *Take:* an authored joint rig as the prior; context for L1.
61. **S. Cha, S. Yoon, K. Seo, J. Noh. *Neural Face Skinning for
    Mesh-agnostic Facial Expression Cloning.* Computer Graphics Forum,
    2025.** https://arxiv.org/abs/2505.22416
    Predicts per-vertex skinning weights that localise a global expression
    code on an unseen mesh. *Take:* region-tied weights as a learned prior;
    context only.
62. **R. E. Barlow, D. J. Bartholomew, J. M. Bremner, H. D. Brunk.
    *Statistical Inference Under Order Restrictions.* Wiley, 1972.**
    The standard reference for isotonic regression and the pool adjacent
    violators algorithm. *Take:* the exact projection for monotone weights
    (L6).
63. **M. J. Best, N. Chakravarti. *Active set algorithms for isotonic
    regression; a unifying framework.* Mathematical Programming 47, 1990.**
    https://doi.org/10.1007/BF01580873
    Pool adjacent violators as a dual active-set method. *Take:* as 62.
64. **Q. Du, V. Faber, M. Gunzburger. *Centroidal Voronoi Tessellations:
    Applications and Algorithms.* SIAM Review 41(4), 1999.**
    https://doi.org/10.1137/S0036144599352836
    Tessellations whose generators are the centroids of their own cells, and
    the algorithms that reach them. *Take:* what a joint-drift loop converges
    to, and the name of the report in L8.
65. **I. Daubechies, R. DeVore, M. Fornasier, C. S. Güntürk. *Iteratively
    Reweighted Least Squares Minimization for Sparse Recovery.*
    Communications on Pure and Applied Mathematics 63(1), 2010.**
    https://doi.org/10.1002/cpa.20303
    Non-quadratic norms minimised through a sequence of weighted
    least-squares problems, with convergence results. *Take:* re-weighting as
    the route to other error norms (Q3).
66. **R. T. Rockafellar, S. Uryasev. *Optimization of Conditional
    Value-at-Risk.* Journal of Risk 2(3), 2000.**
    https://www.risk.net/journal-risk/2161159/optimization-conditional-value-risk
    Minimising the mean of the worst fraction of outcomes, by a formulation
    that is easy to optimise and applies to percentiles outside finance.
    *Take:* a tail objective for the worst error (Q3).
67. **P. G. Kry, D. L. James, D. K. Pai. *EigenSkin: Real Time Large
    Deformation Character Skinning in Hardware.* SCA 2002.**
    https://www.cs.mcgill.ca/~kry/pubs/eigenskin/index.html
    Skinning plus corrections in data-dependent eigenbases evaluated in the
    vertex program. *Take:* the low-rank correction that F8 shows does not
    suit our residual.
68. **S. W. Bailey, D. Otte, P. DiLorenzo, J. F. O'Brien. *Fast and Deep
    Deformation Approximations.* ACM TOG 37(4), SIGGRAPH 2018.**
    Splits a rig's deformation into a linear part and a learned non-linear
    remainder. *Take:* the hybrid pattern of Q9.
69. **K. Crane, C. Weischedel, M. Wardetzky. *Geodesics in Heat: A New
    Approach to Computing Distance Based on Heat Flow.* ACM TOG 32(5),
    2013.** https://doi.org/10.1145/2516971.2516977
    Geodesic distance from a pair of standard sparse linear problems that
    can be prefactored. *Take:* a better distance for the mask, the start and
    the penalty, if the distance estimate is ever implicated.
70. **D. Qin, J. Saito, N. Aigerman, T. Groueix, T. Komura. *Neural Face
    Rigging for Animating and Retargeting Facial Meshes in the Wild.*
    SIGGRAPH 2023.** https://doi.org/10.1145/3588432.3591556
    A learned rig for arbitrary face meshes with interpretable controls.
    *Take:* context only.
71. **W. Ma, D. Kneubuehler, M. Chu, I. Sachs, H. Jiang, S. X. Huang.
    *RigAnyFace: Scaling Neural Facial Mesh Auto-Rigging with Unlabeled
    Data.* NeurIPS 2025.** https://arxiv.org/abs/2511.18601
    Deforms a neutral face mesh into FACS poses to form a blendshape rig.
    *Take:* context only; a possible source of shapes upstream of this tool.
72. **A. Jacobson, Z. Deng, L. Kavan, J. P. Lewis. *Skinning: Real-time
    Shape Deformation.* ACM SIGGRAPH 2014 Courses.** https://skinning.org/
    Course notes on direct, automatic, example-based and decomposition
    methods. *Take:* the survey to read first.
73. **Epic Games. *Skeletal Mesh Rendering Paths in Unreal Engine.***
    https://dev.epicgames.com/documentation/unreal-engine/skeletal-mesh-rendering-paths-in-unreal-engine
    Not a paper. The skin cache skins positions and normals or tangents in a
    compute shader. *Take:* what a production engine does with normals, and
    where recomputing them is offered (Q4).

74. **Y. C. Eldar, P. Kuppinger, H. Bölcskei. *Block-Sparse Signals:
    Uncertainty Relations and Efficient Recovery.* IEEE Transactions on
    Signal Processing 58(6), 3042-3054, 2010.** Block coherence, block OMP and
    the mixed $\ell_2/\ell_1$ programme for signals whose non-zeros come in
    blocks. *Take:* a budget counted per joint and shape (§13.13, Q11).
75. **F. Bach, R. Jenatton, J. Mairal, G. Obozinski. *Optimization with
    Sparsity-Inducing Penalties.* Foundations and Trends in Machine Learning
    4(1), 1-106, 2012.** Proximal, block-coordinate, reweighted, working-set
    and homotopy methods for sparse and group-sparse problems. *Take:* the
    reference for implementing group penalties and their proximal steps.
76. **R. M. Murray, Z. Li, S. S. Sastry. *A Mathematical Introduction to
    Robotic Manipulation.* CRC Press, 1994.** Twists, screw axes and the
    adjoint map between frames. *Take:* the pivot of a linearised joint delta
    as a frame choice for a twist (§13.13).
77. **H. Hazimeh, R. Mazumder. *Fast Best Subset Selection: Coordinate
    Descent and Local Combinatorial Optimization Algorithms.* Operations
    Research 68(5), 1517-1537, 2020.** $L_0$ least squares by coordinate
    descent plus local swaps; the swaps give better supports. *Take:* swaps
    as a finishing step for the delta budget (Q5, Q13).
78. **D. Bertsimas, M. S. Copenhaver, R. Mazumder. *The Trimmed Lasso:
    Sparsity and Robustness.* arXiv:1708.04527, 2017.** A penalty on all but
    the largest coefficients, exact for a sparsity level. *Take:* a budget
    that gradient steps can follow and that can be ramped (Q13).
79. **C.-H. Zhang. *Nearly Unbiased Variable Selection under Minimax Concave
    Penalty.* Annals of Statistics 38(2), 894-942, 2010.** The MCP penalty
    and its solution path. *Take:* a nonconvex penalty that does not shrink
    large deltas (Q13).
80. **E. J. Candès, M. B. Wakin, S. P. Boyd. *Enhancing Sparsity by
    Reweighted $\ell_1$ Minimization.* Journal of Fourier Analysis and
    Applications 14, 877-905, 2008.** A sequence of weighted $\ell_1$
    problems approaching a log penalty. *Take:* as entry 79.
81. **H. Mohimani, M. Babaie-Zadeh, C. Jutten. *A Fast Approach for
    Overcomplete Sparse Decomposition Based on Smoothed $\ell_0$ Norm.* IEEE
    Transactions on Signal Processing 57(1), 289-301, 2009.** SL0: a smooth
    surrogate of the count of non-zeros, sharpened step by step. *Take:* a
    ramp on the shape of the sparsity penalty (Q13).
82. **A. Blake, A. Zisserman. *Visual Reconstruction.* MIT Press, 1987.**
    Graduated non-convexity: solve a convex relaxation, then deform it
    towards the true nonconvex objective. *Take:* the principle behind every
    ramp in §13.14.
83. **J. H. Hong, A. Fitzgibbon. *Secrets of Matrix Factorization:
    Approximations, Numerics, Manifold Optimization and Random Restarts.*
    ICCV 2015, 4130-4138.** A unified view of variable-projection methods for
    factorisation, with high success rates from random starts. *Take:*
    variable projection for the weights (Q12).
84. **J. H. Hong, C. Zach, A. Fitzgibbon. *Revisiting the Variable
    Projection Method for Separable Nonlinear Least Squares Problems.* CVPR
    2017.** Why variable projection has a wider basin than joint
    optimisation. *Take:* as entry 83.
85. **T. Pock, S. Sabach. *Inertial Proximal Alternating Linearized
    Minimization (iPALM) for Nonconvex and Nonsmooth Problems.* SIAM Journal
    on Imaging Sciences 9(4), 1756-1787, 2016.** Alternating proximal steps
    with momentum, convergent for sparse factorisation and dictionary
    learning. *Take:* the principled form of the solver's alternation.

86. **G. H. Golub, M. Heath, G. Wahba. *Generalized Cross-Validation as a
    Method for Choosing a Good Ridge Parameter.* Technometrics 21(2),
    215-223, 1979.** The regularisation parameter that best predicts
    left-out data, without refitting per left-out point. *Take:* calibrating
    alpha per head (§13.16, Q14).
87. **P. C. Hansen. *Analysis of Discrete Ill-Posed Problems by Means of the
    L-Curve.* SIAM Review 34(4), 561-580, 1992.** The corner of the curve of
    residual against solution size as the parameter choice. *Take:* as
    entry 86.
88. **M. Meyer, M. Desbrun, P. Schröder, A. H. Barr. *Discrete
    Differential-Geometry Operators for Triangulated 2-Manifolds.*
    Visualization and Mathematics III, 2003.** Cotangent Laplacian, mixed
    Voronoi areas and curvature estimates on meshes. *Take:* a smoothness
    term whose alpha means the same at any resolution, and a measure of
    sharpness (§13.16).
89. **B. Vallet, B. Lévy. *Spectral Geometry Processing with Manifold
    Harmonics.* Computer Graphics Forum 27(2), 251-260, 2008.** A Fourier
    basis on meshes and filtering in it. *Take:* reading alpha as a cut-off
    wavelength.
90. **L. I. Rudin, S. Osher, E. Fatemi. *Nonlinear Total Variation Based
    Noise Removal Algorithms.* Physica D 60(1-4), 259-268, 1992.** Total
    variation regularisation, which keeps sharp edges. *Take:* regularising
    the residual without blurring sharp targets.
91. **R. M. Ehrig, W. R. Taylor, G. N. Duda, M. O. Heller. *A Survey of
    Formal Methods for Determining the Centre of Rotation of Ball Joints.*
    Journal of Biomechanics 39(15), 2798-2809, 2006.** Compares joint-centre
    estimators from marker motion and introduces SCoRE. *Take:* a target
    for rotating joints in the relocation pass (L9).
92. **S. S. H. U. Gamage, J. Lasenby. *New Least Squares Solutions for
    Estimating the Average Centre of Rotation and the Axis of Rotation.*
    Journal of Biomechanics 35(1), 87-93, 2002.** Closed-form centre and
    axis of rotation from all marker motion. *Take:* as entry 91.
93. **N. J. Mitra, L. J. Guibas, M. Pauly. *Partial and Approximate Symmetry
    Detection for 3D Geometry.* ACM TOG 25(3), 560-568, 2006 (SIGGRAPH).**
    Symmetries found by voting in transformation space. *Take:* a mirror
    map for joint layouts without left and right names.
94. **A. Krause, A. Singh, C. Guestrin. *Near-Optimal Sensor Placements in
    Gaussian Processes: Theory, Efficient Algorithms and Empirical
    Studies.* JMLR 9, 235-284, 2008.** Greedy placement by mutual
    information with a $(1-1/e)$ guarantee. *Take:* advising where a joint
    is missing.
95. **A. Anis, A. Gadde, A. Ortega. *Efficient Sampling Set Selection for
    Bandlimited Graph Signals Using Graph Spectral Proxies.* IEEE
    Transactions on Signal Processing 64(14), 3775-3789, 2016.** Choosing
    which graph nodes to sample so a band-limited signal can be rebuilt.
    *Take:* as entry 94, on the mesh graph itself.

96. **G. Taubin. *Estimating the Tensor of Curvature of a Surface from a
    Polyhedral Approximation.* ICCV 1995, 902-907.** Principal curvatures
    and directions per vertex as eigenvalues and eigenvectors of a symmetric
    matrix, in linear time. *Take:* the per-vertex sharpness and crease
    direction for alpha (§13.16).
97. **A. Girshick, V. Interrante, S. Haker, T. Lemoine. *Line Direction
    Matters: An Argument for the Use of Principal Directions in 3D Line
    Drawings.* NPAR 2000.** Following an earlier SIGGRAPH 99 sketch (A.
    Girshick, V. Interrante, *Real-time Principal Direction Line Drawings of
    Arbitrary 3D Surfaces*, Conference Abstracts and Applications, 271).
    Lines along the first principal direction convey shape best; principal
    directions are estimated on meshes with Hamann's and Taubin's methods.
    *Take:* which direction across a crease must not be smoothed (§13.16).
98. **K. Hildebrandt, K. Polthier. *Anisotropic Filtering of Non-Linear
    Surface Features.* Computer Graphics Forum 23(3), 391-400, 2004.** A
    discrete shape operator and an anisotropic mean curvature flow that keep
    and sharpen feature lines. *Take:* the anisotropic form of the
    smoothness term (§13.16).

## 15. Verification notes, continued (Revision 4)

- Entries 74 to 98 were checked the same way on 14 November 2026; entry 97
  was also read in full. The
  page range of entry 84 differs between indexes and is left out; entry
  78 is cited as its preprint because no journal version was confirmed.
  The measurements in §13.13 are this repository's own (one public head,
  exact solves on the CPU), as are the length scales and the calibration
  of $k$ in §13.16. The scaling argument for the uniform Laplacian and the
  anisotropic use of principal directions are this document's reasoning,
  not claims of the cited works.
- Entries 29 to 73 were checked by web search for authors, title, venue and
  year, and described from their abstracts or publisher records. None of the
  full texts was read for this revision. Claims in §13 are limited to what
  those records state; anything stronger is this document's reading and is
  marked as a "take".
- Entry 33 is taken from the compskin paper's own reference list. Entries 38
  and 40 were confirmed by title, author and venue only.
- Two descriptions go beyond the records found: the form of the visual metric
  in entry 42 (position error combined with the error of the geometric
  Laplacian) is given from memory of the paper, and the statement in §13.6
  that entry 61 supervises its weights with a segmentation of the face comes
  from a secondary summary. Check both before relying on the detail.
- The statement in §13.3 that Unreal Engine can recompute tangents for chosen
  mesh sections rests on the engine's settings as described by its users and
  on entry 73 for the skin cache itself; check it against the engine version
  in use.
- §12 reports results qualitatively or as relative figures.
- The findings cited in §13 as F1 to F14 come from exact solves and
  measurements on finished results of the public sample heads. They are
  described in [roadmap.md](roadmap.md) §3 with their limits: one or two
  heads, one seed, weights or deltas held fixed where stated.
