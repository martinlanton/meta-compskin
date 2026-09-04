# How the solver works

`SkinCompressor` finds the skin weights and the per-shape joint deltas at the
same time, starting from random values. This page explains the optimisation so
that the settings in [Compressing](../user_guide/compressing.md) make sense, and
so that you can read the progress log. It follows Section 4 of the paper.

## What is being minimised (Equation 9)

For every vertex $i$ and every shape $k$, the error is the distance between the
skinned reconstruction and the original delta:

$$
E_{i,k} = \Big| \hat{\mathbf{v}}_{k,i} - \hat{\mathbf{v}}_{0,i} - \sum_j w_{i,j}\,\mathbf{N}_{k,j}\,\mathbf{v}_{0,i} \Big|
$$

The solver minimises the sum of $E_{i,k}^{\,p}$ over all vertices and shapes,
with $p = 2$ by default (ordinary least squares), subject to the skinning
constraints:

| Constraint | On | Why |
|------------|----|-----|
| non-negative | weights | GPU skinning assumes it; negative weights produce artefacts |
| sum to one per vertex | weights | keeps the neutral pose fixed (see Equation 4) |
| at most $K$ per vertex | weights | vertex shader budget; $K = 8$ by default |
| at most $L$ non-zero coefficients overall | deltas | **the compression**: $L = 6000$ by default |

A **Laplacian smoothness term** is added to the loss, weighted by `alpha`. It
penalises reconstructions where a vertex moves very differently from its
neighbours, which discourages the solver from explaining a shape with jagged
weight maps. This is the same regularisation used by Dem Bones.

In code (`SkinCompressor.train`):

```python
loss = (B_X - A).pow(power).mean().pow(2 / power)      # data term, p = power
loss += alpha * (L @ B_X.T).pow(2).mean()              # Laplacian term
```

where `B_X` is the current reconstruction $\mathbf{B}\mathbf{C}$, `A` the
target deltas, and `L` the mesh Laplacian built from `rest_faces`.

## Unconstrained optimiser plus projections

The parameters are the six coefficients per (shape, joint) in `B_rt` and the
raw weights `W`. Both are ordinary PyTorch tensors optimised with Adam
(learning rate $10^{-3}$, $\beta_1 = \beta_2 = 0.9$). Adam knows nothing about
the constraints, so after **every** step the parameters are projected back
onto the feasible set. This is the "proximal" part of the method:

1. **Weights, sparsity and sign.** For each vertex keep only the $K$ largest
   weights and set the rest to zero, then clamp negatives to zero. One
   `torch.topk` call per iteration.
2. **Weights, partition of unity.** Divide each vertex's weights by their sum.
   In phase 1 this is skipped (see below); in phase 2 it is applied inside the
   forward pass.
3. **Deltas, global sparsity.** Across the whole `B_rt` tensor keep the $L$
   coefficients with the largest absolute value and zero every other one.
   Unlike the weights, the budget is global: the solver decides freely which
   shapes and joints deserve non-zeros.

Projection 3 is what distinguishes this method from earlier skinning
decompositions. Because it runs from the first iteration, the solver adapts
the weights to a sparse set of deltas rather than being handed a dense
solution that is then pruned. The paper shows that pruning a dense Dem Bones
solution to the same sparsity gives 1.5 to 3 times larger errors.

Since $\mathbf{B}$ is a linear combination of the six basis matrices with
coefficients `B_rt`, sparsity in `B_rt` is sparsity in $\mathbf{B}$: 6000
coefficients means at most 1000 non-zero (shape, joint) blocks, out of
$S \times P$ (about 10 000 for the sample heads). That is the "about 90%
zeros" figure.

## Two phases

`run` calls `train` twice with the same tensors:

| Phase | Weight normalisation | Purpose |
|-------|----------------------|---------|
| 1 | off | Weights grow freely. This makes it easier to discover *which* joints should own which region without the coupling that normalisation introduces. |
| 2 | on | Weights are normalised per vertex inside the forward pass, so the solution satisfies partition of unity. Refines phase 1. |

Each phase runs `iterations` steps, so the total is twice the number you pass.
The paper used 20 000 total; the default `iterations=10000` matches it.

## Initialisation and reproducibility

- `B_rt` starts as small Gaussian noise (scale `init_weight`, $10^{-3}$).
- `W` starts as very small Gaussian noise ($10^{-8}$), so the first
  projection picks an essentially random $K$ joints per vertex; the optimiser
  sorts it out within the first few hundred iterations (Figure 1 of the paper
  shows this convergence).
- The torch seed is fixed to 12345 in the constructor. Runs on the same
  hardware and library versions are bit-for-bit repeatable; the regression
  tests rely on this. Different GPUs, CPU versus GPU, or different torch
  versions produce equally valid but not identical solutions.

## Reading the log

Every 200 iterations `train` prints one line:

```
01200(0.987) 1.23456e-02 4.56789e-01 5820 47552
  │      │        │           │        │     └─ non-zero weights (≈ N × K once converged)
  │      │        │           │        └─────── non-zero delta coefficients (≤ L)
  │      │        │           └──────────────── largest single error so far (model units)
  │      │        └──────────────────────────── loss (data + Laplacian term)
  │      └───────────────────────────────────── seconds since the previous line
  └──────────────────────────────────────────── iteration within the current phase
```

The loss should fall steadily and the largest error should drop by an order
of magnitude within the first couple of thousand iterations, then creep down.
The two non-zero counts settle quickly at their budgets.

At the end, `run` prints the two headline metrics on the final normalised
solution (values shown are typical for the Aura sample at default settings):

```
maxDelta 0.58        # MXE: worst vertex on the worst shape
meanDelta 0.0038     # MAE: average over all vertices and shapes
```

Both are in the model's units (centimetres for the sample heads; the paper's
tables are in millimetres). How to interpret them is in
[Evaluating results](../user_guide/evaluating_results.md).

## The settings and what they trade

| Setting | Default | Increasing it | Decreasing it |
|---------|---------|---------------|---------------|
| `iterations` | 10 000 per phase | lower error, longer solve; returns diminish past ~20 000 total | faster, rougher; 600 is enough to check a pipeline |
| `number_of_bones` ($P$) | 100 | more capacity, higher runtime cost per frame | cheaper runtime, error rises quickly below ~20 |
| `max_influences` ($K$) | 8 | smoother weight maps, more shader cost | must stay ≥ 4 or so; must be smaller than $P$ |
| `total_nnz_B_rt` ($L$) | 6000 | more non-zero deltas, better fit, less compression | more compression, more error; cannot exceed $6 S P$ |
| `alpha` | per model, 10 or 50 | smoother, blurrier result | sharper, risk of noisy weights; lower for dense meshes |
| `power` ($p$) | 2 | high values (12) minimise the *worst* error at the cost of smoothness; needs more capacity and many more iterations | |

The paper's "HD" experiment (200 joints, $K = 32$, no delta sparsity,
$p = 12$, 500 000 iterations) shows the far end of this space: 50 times lower
maximum error than Dem Bones, at 2.5 hours of GPU time.

## Where the time goes

Each iteration is a dense matrix product $\mathbf{B}\mathbf{C}$ of size
$3S \times N$, its gradient, and a few `topk` calls. On an A6000 an iteration
takes a few milliseconds; on a CPU, tens to hundreds. The Laplacian is built
once with SciPy. Memory scales with $S \times N$; the largest sample head
(23 735 vertices, 287 shapes) fits comfortably on an 8 GB GPU.

Next: [Data formats](data_formats.md).
