# Geodesic candidate-joint filter for user-supplied joints

Research notes behind `candidate_joints_per_vertex` in `SkinCompressor`
(`src/metacompskin/model_fit.py`). Written 2026-09-10, implemented in the same
session.

## 1. Problem

`rest_joint_matrices` lets a rigger hand the compressor the rest matrices of
their own facial joints. Before this work those matrices only set the joint
count $P$ and were echoed into the output as `restXform`. The solve itself
never read them: the weights `W` start as $10^{-8}$ Gaussian noise, the
top-$K$ projection keeps whichever joints the gradient reached first, and
nothing ties joint $j$ to a location on the face. A joint the rigger placed
on the left cheek could end up driving the right eyebrow.

That defeats the reason riggers pass their own joints. In the vast majority of
cases those joints, and the controls above them, double as **tweakers**: the
solved motion is applied to the tweaker control's parent, so a generic control
that activates a shape moves the parent, which moves the tweaker, which moves
the joint; the animator can then move the tweaker inside its parent to adjust
a pose on top of the shape. That only works if the joint drives the skin
around it.

## 2. Requirements

1. **Locality.** Every non-zero weight of a vertex must sit on a joint that is
   close to that vertex, so moving a tweaker moves the skin under it and
   nothing far away. This has to be a guarantee, not a tendency.
2. **Pivots are the rig's job, not the solver's.** The solved deltas are
   affine transforms about the centred rest origin. `frame_joint_matrices`
   (`maya_rig_builder.py`) conjugates them by the rest-to-scene matrix and the
   bind matrix, so the joint's world matrix is correct wherever the joint
   sits, and a tweak added in the joint's parent space rotates the skin about
   the joint's own pivot. No solver change is needed for this.
3. **Off by default without joints.** The anonymous 100-bone path, and the
   regression tests pinned to its numerics, must not change.
4. **Composable with influence annealing** (`build_training_schedule`), which
   was added just before this work.

## 3. Approaches considered

| Approach | Guarantees locality | Cost | Verdict |
|----------|--------------------|------|---------|
| **Hard candidate mask** in the proximal step: each vertex may only take weight from its $M$ nearest joints | yes | one multiply per iteration, one distance computation up front | **chosen** |
| Soft distance penalty $\lambda \sum_{ij} w_{ij} d_{ij}^2$ in the loss | no, a strong shape can still pull a far joint | new hyper-parameter, changes numerics | rejected |
| Distance-shaped initialisation only | no, the solver drifts | none | rejected alone; not needed with the mask since the mask already confines the random init |
| Joint-centred transformation basis (conjugate `TR` by each rest matrix; item 2.3a of the private improvement plan) | no effect on which joint drives which vertex | changes what the sparse coefficients mean; may change sparsity | separate experiment, out of scope |

The mask fits the existing method naturally: the solver is already
"unconstrained optimiser plus projections", and locality is one more
projection onto a feasible set, applied before the existing top-$K$ prune.
Because it runs from the first iteration, the solver adapts to it rather than
being pruned afterwards, the same argument the paper makes for delta sparsity.

Within the candidate set the solver keeps full freedom to pick *which* $K$ of
the $M$ joints matter and with what weights, so the fit is still optimised;
only the search space is restricted to what a rigger would accept.

## 4. Distance: geodesic along the mesh

### 4.1 Why not Euclidean

Straight-line distance fails exactly where facial joints are densest. An
upper-lip vertex is closer in space to a lower-lip joint than to the lip
corner, and eyelid vertices are close to the joints of the opposite lid. A
Euclidean filter would let a lower-lip tweaker drag the upper lip, which is
the artefact riggers most want to avoid.

### 4.2 Definition used

For joint $j$ with rest position $\mathbf{p}_j$ and vertex $i$:

$$
d_{ij} = \lVert \mathbf{p}_j - \mathbf{v}_{s(j)} \rVert + \operatorname{geo}(s(j), i)
$$

where $s(j)$ is the vertex nearest to the joint in space and $\operatorname{geo}$ is
the shortest path along mesh edges. Adding the snap gap keeps distances
comparable between joints that sit on the surface and joints that float a
little above or below it (typical for joints placed by hand or at a bone's
pivot).

### 4.3 Algorithm

Implemented as `geodesic_joint_distances(rest_verts, rest_faces, joint_positions)`:

1. Graph: every pair of vertices that shares a face is an edge, weighted by
   its Euclidean length. This is the same connectivity as the Laplacian's
   adjacency (`_build_adjacency_matrix`), so quad diagonals are edges too.
   Diagonals are a legitimate shortcut across a quad and make the path metric
   closer to true surface distance on quad meshes than edge-only paths would be.
   Duplicate pairs (an edge shared by two faces) are removed before building
   the sparse matrix, otherwise SciPy would sum their lengths.
2. Sources: `cdist` from the $P$ joints to the $N$ vertices, `argmin` per joint.
3. `scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=sources)`
   runs one Dijkstra per joint and returns a $(P, N)$ matrix in one call.
4. Add the snap gap per row.
5. Unreachable pairs (a vertex on a shell with no joint, e.g. eyeballs or
   teeth exported with the head) come back as `inf`. They are replaced by the
   Euclidean distance plus the largest finite geodesic distance. Joints on the
   vertex's own shell therefore always rank first, and the rest fall back to
   spatial order instead of an arbitrary tie.

Cost: one `cdist` of $P \times N$ and $P$ Dijkstra runs on a graph with about
$4N$ edges. For $P = 100$, $N = 7000$ this is well under a second and runs
once before training.

### 4.4 Alternatives not taken

- Exact or heat-method geodesics (Crane et al. 2013) are more accurate on
  coarse meshes but need a discretised Laplace-Beltrami operator and a linear
  solve per joint, or an extra dependency. Only the *ranking* of joints per
  vertex matters here, and edge-path distances rank correctly at facial joint
  spacing. Can be revisited if a real head shows ranking errors.
- A radius instead of a count. A count is scale-free (model units differ
  between exporters) and maps directly onto the annealing budget (Section 5).

## 5. Mask: the $M$ nearest joints per vertex

`candidate_joint_mask(distances, M)` marks the $M$ smallest distances per
column with `argpartition`, giving a boolean $(P, N)$ mask. With $M \ge P$
the mask is all true.

In `SkinCompressor.run` the mask is built once, applied to the random initial
`W`, and stored on `self.candidate_mask`. In `train` it is applied inside the
existing `no_grad` block, before the top-$K$ cut:

```
W *= mask            # locality (new)
top-K per vertex     # sparsity
clamp >= 0           # sign
```

Adam still accumulates moments for masked entries, but they are zeroed after
every step, so they never contribute to the forward pass.

### 5.1 Interaction with influence annealing

`build_training_schedule` anneals $K$ over staged iterations so that many
joints can compete before the final cut. Three observations settled how the
two features combine:

1. **$M$ equal to $K$ collapses the solver.** With exactly $K$ candidates the
   top-$K$ prune has nothing to choose; the weights become a nearest-neighbour
   assignment with fitted magnitudes. $M$ must stay strictly wider than the
   final $K$.
2. **Locality is a hard requirement, not a budget.** Widening the candidate
   set in early stages would let weight build on far joints that must be
   discarded later, wasting iterations. So $M$ is constant over every phase,
   the warm-up included; it does not anneal.
3. **Annealing still helps inside the mask.** With $M = 16$ and $K = 8$ from
   step one, the first prune keeps the eight neighbours the random init
   happened to favour, the same lottery as before, only local. Letting all
   sixteen compete for a stage before pruning is exactly what annealing
   offers, and the range is now short, so one extra stage covers it.

Hence: **$M$ is the ceiling of the annealing range.** `build_training_schedule`
gained a `start_influences` argument and now interpolates the budget
geometrically from the ceiling down to $K$:

$$
K_i = \operatorname{round}\!\left(K \cdot (M / K)^{\frac{N-1-i}{N-1}}\right), \quad i = 0 \ldots N-1
$$

Without a ceiling it defaults to $K \cdot 2^{N-1}$, which reproduces the
previous halving rule exactly. A single stage keeps $K$ throughout, so the
default run is unchanged with or without joints. Examples with $K = 8$,
$M = 16$: two stages give 16, 8; three stages give 16, 11, 8.

The earlier alternative, clamping each phase's $K$ to $M$, was dropped: it
silently made stages above $M$ identical and needed an explanation in the
log. Making annealing and the filter mutually exclusive was also dropped,
for reason 3 above.

## 6. Defaults and validation

- `candidate_joints_per_vertex=None` (default): with joint matrices,
  $M = \min(2K, P - 1)$, so 16 at the defaults. The user's premise is that
  joints are passed precisely to get this behaviour, so it is on by default.
  Without joint matrices there is no filter. If even $P - 1 < K$ the filter is
  left off and `run` raises its existing "K must be smaller than P" error.
- `0` turns the filter off explicitly (joints then only set $P$ and
  `restXform`, as before).
- Any other value must satisfy $K \le M < P$, otherwise `ValueError`.
- A value without joint matrices raises `ValueError`: the filter has nothing
  to rank by.
- The same value is exposed as `--candidate-joints-per-vertex` on the CLI and
  as `candidate_joints_per_vertex` on `CompressionSettings` /
  `compress_and_build_rig` in the Maya pipeline.

## 7. Diagnostics

After a run with joint matrices, `run` prints the minimum, median and maximum
number of vertices driven per joint, and lists joints that drive no vertex at
all. An idle joint means it is never among any vertex's $M$ nearest (placed
too far from the mesh, or redundant with a neighbour) or the solver found no
use for it. Either way the rigger should know before building a tweaker on it.

## 8. Expected effects and open experiments

Restricting the search space can only raise the reconstruction error relative
to the unconstrained solve with the same $P$; the question is by how much, and
whether annealing inside the mask recovers it. Experiments belong in the
private tests repository (`../meta-compskin_private_tests`), on a real head
with its facial joints:

1. Error (MAE, MXE) with the filter at the default $M$ versus `0`.
2. One stage of 10 000 versus two stages 5 000 + 10 000 at the same $M$, to
   decide whether the joint path should default to two stages.
3. A sweep of $M$ in {K+1, 2K, 3K, P-1}.
4. Visual check of the lips and eyelids: move each lip tweaker and confirm
   only its own lip follows.

## 9. References

- Compressed Skinning for Facial Blendshapes (SIGGRAPH 2024), Section 4:
  the proximal projections this filter joins.
- Le & Deng 2012, *Smooth Skinning Decomposition with Rigid Bones*, and
  Le & Deng 2014, *Robust and Accurate Skeletal Rigging from Mesh Sequences*
  (Dem Bones): spatial initialisation of bones by clustering, the closest
  prior art for tying bones to regions.
- Dijkstra 1959, shortest paths; `scipy.sparse.csgraph.dijkstra`.
- Crane, Weischedel & Wardetzky 2013, *Geodesics in Heat*: the alternative
  distance not taken.
- Zhu & Gupta 2017, *To prune, or not to prune*: the gradual sparsity
  schedule that $M$ now caps.
