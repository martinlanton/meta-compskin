# Research notes — why a root influence, and why this shape of it

Revision 1, against commit `12ac666`. Companion to [spec.md](spec.md),
[design.md](design.md), [tasks.md](tasks.md). The full literature survey
lives in `plan/research_output.md`; this note keeps only what the design
rests on.

## 1. The defect

`plan/research_output.md` §2 and `plan/experiment_results.md` establish, on
cut and uncut heads, that long un-normalised warm-ups empty some weight
columns and the first normalised step divides by zero. Two operations
empty a column: the top-$K$ cut, which discards small positive candidates
every step and never lets them back, and Adam's decay of the survivors of
a vertex whose candidates cannot explain its motion. Annealed schedules
survived longer only because their warm-up runs at $K_0 = M$, where the
cut is a no-op.

The deeper reason is representational. In the delta formulation
(paper Eq. 5) a vertex that does not move is fit exactly by zero weights,
and nothing in the loss prefers any joint for it. Linear blend skinning
(paper Eq. 2) cannot ship a zero row: such a vertex would collapse to the
origin at runtime. The only legal expression of "does not move" is weight
one on a joint whose transform is the identity in every frame. The solver
has no such joint; James & Twigg 2005 get one from clustering (static
triangles form an identity bone), FLAME carries it as the neck joint, and
game face rigs put the skull on the head bone.

## 2. Why the root works, mechanically

Let $r$ be the root, with $\mathbf{N}_{k,r} = 0$ for all $k$, and
$w_{ir} \ge \varepsilon_r$ enforced after every projection.

- **Un-normalised warm-up.** The root contributes nothing to the forward
  pass, so $\partial \mathcal{L} / \partial w_{ir} = 0$: it stays at the
  floor while the other weights of a static or unexplainable vertex decay
  to zero. The column sum is $\ge \varepsilon_r$ throughout.
- **First normalised step.** $w_{i,:} / \sum_j w_{ij}$ is finite for every
  column. A vertex whose other weights all reached zero gets $w_{ir} = 1$.
- **Normalised phases.** With $s_i = \sum_j w_{ij}$ and $g_{ij} =
  \partial\mathcal{L}/\partial \tilde w_{ij}$ the gradient on the raw root
  weight is $-\tfrac{1}{s_i}\sum_{k \ne r} \tilde w_{ik}\, g_{ik}$. It is
  negative (the weight grows) when the vertex overshoots along its joints'
  motion, positive otherwise; the clamp and floor bound it below. A root
  at the floor cannot displace a strong joint: one Adam step moves it by
  about the learning rate, far below a full column's $(K{+}1)$-th cutoff
  (`plan/gradual_sparsity_annealing/research.md` §3). It re-enters only
  where the other joints are weak, which is the intended semantics.
- **Output.** `design.md` §4 shows every saved column has at most $K$
  non-zeros and none is empty.

## 3. Alternatives considered and rejected

| Alternative | Why not |
|---|---|
| Epsilon in the denominator | Saves an all-zero row; the vertex collapses to the origin at runtime and the solver never notices because a zero prediction fits a static vertex. |
| Displacement threshold before the solve (memory option B, done early) | Catches static vertices only; the traces cannot tell static from unexplainable vertices, and a threshold misses the latter. Remains a sensible builder-side snap later. |
| Rescue a dead column onto its nearest moving joint | A single positive entry normalises to one; a static vertex would follow that joint. |
| Exact projection onto the $K$-sparse simplex (Kyrillidis et al. 2013; Condat 2016) | Guarantees a non-empty column, but replaces the paper's clamp-and-normalise on every iteration and removes the un-normalised phase; every pinned output changes. Worth an experiment later, not a fix now. |
| Sparsemax parametrisation (Martins & Astudillo 2016) | Same objection: a different solver. |
| Top-$K$ cut on non-root rows only, root always kept | Columns can end with $K$ real joints plus a genuine root weight; shipping them needs a lossy drop. "Root competes in the cut, floor restores it" gives exact $K$ with no loss. |
| $K - 1$ real joints plus the root | Exact budget at a capacity cost on every vertex. Kept as the fallback if partial-root vertices turn out to matter. |
| Root exempt from the cut without a floor (initial value only) | Adam's noise-driven steps outgrow any fixed small value within a few iterations, the cut drops the root, and lock-in keeps it out; the collapse returns. |

## 4. Why a floor of $10^{-6}$

Weights that are not exactly zero are either above roughly one Adam step
($10^{-3}$) or on their way to the clamp within a step, so any floor well
below $10^{-3}$ gives a clean hand-over: the root's normalised share is
negligible while any other entry is alive and one when none is. The value
must also survive `float32` arithmetic next to sums of order one, which
rules out $10^{-8}$ (below `float32` resolution at one). $10^{-6}$ meets
both; it is a module constant, not a public argument, until data says it
matters.

## 5. Why R0 is included

R0 removes the cut from the warm-up on the joint path, which the
experiments show is the difference between single-stage and annealed
schedules at equal length. Alone it is a mitigation: the cut mesh died at
$K_0 = M$ given enough steps. With R1 it is still worth having, because a
warm-up that discards half of every column each step is the wrong regime
for deciding ownership, which is what the warm-up is for. It costs one
line and matches what annealed schedules already do.

## 6. References

- Compressed Skinning for Facial Blendshapes, SIGGRAPH 2024, §4
  (projections) and §2.2 (Eq. 2, partition of unity). Local copy
  `paper/compressed_skinning_for_facial_blendshapes.md`.
- D. L. James, C. D. Twigg. *Skinning Mesh Animations.* SIGGRAPH 2005.
  https://doi.org/10.1145/1073204.1073206
- T. Li et al. *Learning a model of facial shape and expression from 4D
  scans* (FLAME). SIGGRAPH Asia 2017. https://doi.org/10.1145/3130800.3130813
- A. Kyrillidis et al. *Sparse projections onto the simplex.* ICML 2013.
  https://proceedings.mlr.press/v28/kyrillidis13.html
- A. F. T. Martins, R. F. Astudillo. *From Softmax to Sparsemax.* ICML 2016.
  https://proceedings.mlr.press/v48/martins16.html
- `plan/research_output.md` (Revision 3) §2, §4.5–4.6, §5.0–5.1;
  `plan/experiment_results.md`; memory note "root-joint-skinning-options"
  (options A/B/C, of which this plan is C).
