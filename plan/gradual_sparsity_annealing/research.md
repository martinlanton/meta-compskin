# Research notes — why anneal the sparsity budgets, and how to tell if it worked

Companion to [spec.md](spec.md), [design.md](design.md), [tasks.md](tasks.md). Revision 3.
Line numbers into the paper refer to
`paper/compressed_skinning_for_facial_blendshapes.md`; code lines to commit
`87dcd6d`.

## 1. The problem being solved

The compressor factorises the blendshape delta matrix
$\mathbf{A} \in \mathbb{R}^{3S \times N}$ as $\mathbf{A} \approx \mathbf{B}\mathbf{C}$
(paper Eq. 3, lines 311-332) with

- $\mathbf{B} \in \mathbb{R}^{3S \times 4P}$: per (shape $k$, joint $j$) a
  $3 \times 4$ block $\hat{N}_{k,j} = I + N_{k,j}$, where $N_{k,j}$ is a
  linear combination of six fixed basis matrices (Eq. 8) with coefficients
  `B_rt[:, k, j]`. Free parameters: $6SP$.
- $\mathbf{C} \in \mathbb{R}^{4P \times N}$: column $i$ holds
  $w_{i,j}\,\mathbf{v}_{0,i}$ for every joint $j$. Free parameters: the
  weights $W \in \mathbb{R}^{P \times N}$.

The objective (Eq. 9, lines 440-446), with $p = 2$:

$$
\min_{W,\,B_{rt}} \Big(\tfrac{1}{3SN}\sum_{k,i,\text{axis}} (\mathbf{B}\mathbf{C} - \mathbf{A})^{p}\Big)^{2/p} + \alpha \,\big\| \mathbf{L}\,(\mathbf{B}\mathbf{C})^{\top} \big\|_F^2 / (3SN)
$$

subject to $w_{i,j} \ge 0$, $\sum_j w_{i,j} = 1$, $\|w_{i,:}\|_0 \le K$, and
$\|B_{rt}\|_0 \le L$. Code: `model_fit.py:608` (data term), `:615`
(Laplacian term; $\mathbf{L}$ is the rigidity Laplacian of
`model_fit.py:450-511`).

Notation used below: $S$ shapes, $N$ vertices, $P$ joints, $K$ influences per
vertex, $L$ non-zero delta coefficients.

## 2. How the solver enforces the constraints

Unconstrained Adam (`lr=1e-3`, $\beta_1=\beta_2=0.9$, `model_fit.py:597`)
followed, after **every** step, by projections (`model_fit.py:621-634`), the
"proximal" pattern of Parikh & Boyd [5] as described in paper lines 462-497:

| Projection | Operator | Code |
|---|---|---|
| $\Pi_K$ on $W$ | per column keep the $K$ largest entries, zero the rest, clamp at 0 | `topk(W, K + 1, dim=0)` → mask → `clamp_(min=0)` |
| $\Pi_L$ on $B_{rt}$ | keep the $L$ largest $\lvert B_{rt} \rvert$ globally, zero the rest | `topk(B_rt.abs().flatten(), L)` → mask |
| partition of unity | $W_n = W / \sum_j W$ inside the forward pass | `model_fit.py:601`, phase 2 only |

Important mechanical facts:

- Both projections run from **iteration 1** of phase 1.
- The mask is recomputed from the *current* values each step, so a zeroed
  entry is not permanently frozen — it is re-evaluated every step.
- `train()` creates a fresh Adam each call (`model_fit.py:597`), so the two
  phases today already reset Adam's moments.

## 3. Why the first few steps decide the joint assignment

Initialisation (`model_fit.py:333-362`): $B_{rt} \sim 10^{-3}\,\mathcal{N}(0,1)$,
$W \sim 10^{-8}\,\mathcal{N}(0,1)$. At step 0 the joints are noise and no
vertex is attached to anything. The first gradient on $W$ is determined by
the random $B_{rt}$; after one Adam step each vertex's column has $K$
survivors chosen essentially by that noise.

Why a zeroed weight almost never re-enters. After the projection an entry is
exactly 0. On the next step Adam moves it by roughly
$\eta\,\hat m/\sqrt{\hat v} \approx \eta = 10^{-3}$ (Adam normalises the
step per parameter, so a consistently signed gradient gives a step of about
$\eta$ regardless of its magnitude [6]). It survives $\Pi_K$ only if this
single step exceeds the $(K{+}1)$-th largest weight in its column
(`W_cutoff < W`, `model_fit.py:622-623`), because the projection runs again
before a second step can accumulate. Once the surviving weights are of order
$1/K \approx 0.1$ (a few hundred iterations in), the threshold is two orders
of magnitude above what one step can deliver. Re-entry is therefore only
possible while all the weights in a column are still tiny — i.e. during the
noise-driven first steps. The assignment made then is the assignment shipped.

The same holds for $\Pi_L$, more softly: a zeroed coefficient re-enters only
if one $\eta$-sized step beats the global $L$-th largest $\lvert B_{rt} \rvert$,
which is easy while $B_{rt} \sim 10^{-3}$ and hard once it has grown.

Consequences:

1. The seed is not "jitter around one answer"; it picks a **basin** through
   a discrete assignment that is locked in early. Different seeds give
   structurally different joint layouts (paper lines 139-148 lean on random
   init deliberately and warn about symmetric local minima).
2. Any single-seed comparison of hyper-parameters (the L sweep, the alpha
   sweep) is confounded with this basin choice.
3. If the commitment could be delayed until the joints carry information,
   the final $K$ survivors would be chosen on merit. That is what the
   schedule does.

If a capacity sweep over $L$ shows quality *degrading* past some point
rather than plateauing, that is consistent with 1–2: more capacity does not
reduce the attainable error, so the loss of quality must come from the
optimiser settling in a worse basin, which is exactly a basin/initialisation
effect — not evidence of a capacity limit.

## 4. Gradual magnitude pruning (what the schedule is)

Neural-network compression established that pruning to a target sparsity in
stages during training beats pruning to it at once:

- Han, Pool, Tran & Dally 2015 [1]: iterative prune → retrain cycles reach
  ~9–13× fewer weights at no accuracy loss where one-shot pruning cannot.
- Zhu & Gupta 2017 [2]: a smooth sparsity schedule
  $s_t = s_f + (s_i - s_f)\,(1 - \tfrac{t - t_0}{n\Delta t})^3$ applied during
  training ("gradual magnitude pruning") matches large-sparse models to
  small-dense ones across tasks; the key argument is that early pruning
  removes weights before the network has decided which matter.
- Gale, Elsen & Hooker 2019 [3]: across a large sweep, gradual magnitude
  pruning is at least as good as more elaborate methods — a strong, simple
  baseline.
- Frankle & Carbin 2019 [4]: iterative magnitude pruning finds sparse
  sub-networks that one-shot pruning misses ("lottery tickets"), again
  attributing this to the order of commitment.
- Evci et al. 2020 [7] (RigL): regrowing pruned connections by gradient
  magnitude helps further — relevant background for a follow-up, out of
  scope here (spec §4).

The mapping to this solver is direct: $\Pi_K$ and $\Pi_L$ are magnitude
pruning applied every iteration; today's behaviour is "prune to the final
sparsity from step 0"; the proposed `iterations=(T, T, T)` (K stages 32, 16, 8) is a
three-step staircase approximation of [2]. $K = 32$ as the loose end is the
value the paper's own high-detail experiment used (lines 626-633), so the
warm-up runs in a regime the paper has shown to be well-behaved.

Related signal-processing view: $\Pi_L$ makes the $B_{rt}$ update an
iterative hard-thresholding step (Blumensath & Davies [8]); *continuation*
— starting with a loose sparsity level and tightening it — is a standard
device for getting IHT-type methods out of poor local minima.

## 5. Why the staircase is shaped as it is (design choices)

The public knob is `iterations` as a sequence $(T_0, \dots, T_{N-1})$: N is
the number of stages, stage i runs $T_i$ steps at
$K_i = K \cdot 2^{N-1-i}$, i.e. $K \cdot 2^{N-1}, \dots, 2K, K$.

- **Geometric halving.** Each cut removes half of every vertex's candidate
  joints, so the *relative* pressure is the same at each stage — the
  natural discrete analogue of [2]'s smooth schedule. The last cut
  ($2K \to K$) is the harshest in absolute terms and is followed by a full
  phase to recover; if results show that phase struggling, a finer tail
  (e.g. 32, 16, 12, 8) is the first thing to try, by assigning an explicit
  schedule.
- **Warm-up at the loosest stage, unnormalised.** Keeps phase 1's documented
  purpose (weights grow freely to discover ownership) and applies it where
  it matters most — while ownership is being decided. This is also what
  makes N = 1 today's run without a special case, hence N stages → N + 1
  phases.
- **Warm-up at $K \cdot 2^{N-1}$, not unconstrained.** The docs argue that
  sparsity from the first iteration lets the solver adapt to a sparse
  solution rather than prune a dense one; $K = 32$ (N = 3) is the regime the
  paper's own HD experiment validated (lines 626-633).
- **All later stages normalised.** After each cut the survivors' sum drops
  below one; `Wn = W / W.sum()` rescales them immediately, so the cut is
  absorbed by renormalisation rather than by a burst of gradient.
- **Fresh Adam per phase.** Already the case; here it also discards stale
  momentum on entries that were just pruned.
- **Size the final stage generously.** It follows the harshest cut
  ($2K \to K$); the sequence form lets the owner give it more steps than the
  earlier stages, and the docs recommend that.
- **L is not annealed by the knob.** §3's lock-in argument is strongest for
  $W$; L annealing is an arm of the experiment, reachable through an explicit
  schedule, and becomes a public option only if the data supports it.

## 6. Seeds, non-convexity, reproducibility

- The objective is non-convex (bilinear in $W$ and $B_{rt}$ plus hard
  sparsity constraints); different starts reach different local minima of
  similar loss (paper lines 122-128, 139-148).
- All randomness is in the two `torch.randn` calls; both are drawn on the CPU
  and then moved, so a given seed yields the same starting point on CPU,
  CUDA or MPS. The trajectories still differ across devices because of
  floating-point differences accumulated over thousands of Adam steps
  (`tests/test_data/macos/SETUP.md` notes; PyTorch reproducibility notes [9]).
- On CUDA, `torch.manual_seed` does not by itself guarantee bit-identical
  runs (some kernels use non-deterministic atomics). Same-seed
  reproducibility across runs is therefore expected on CPU and *likely* on
  CUDA; the experiment relies on seeds being *different*, not on bitwise
  repeats.

## 7. Designing a fair comparison

Confounds to control, and how the script controls them:

| Confound | Control |
|---|---|
| Annealed runs have N+1 phases, baseline 2 → more steps | equal **total** steps: `iterations_per_phase = total // phases` |
| N+1 phases also mean N+1 Adam resets and N renormalisation kicks | `baseline_control`: N+1 phases all at (K, L) — same phase structure, no annealing |
| Seed luck | 5 seeds per variant, same seed set for every variant; report mean, std, min, max |
| Model-specific alpha / L | fixed at the model's own tuned values for every variant |
| `power` | fixed at 2 (see §10) |
| Warm-up K must be $< P$; L stages must be $\le 6SP$ | $K \cdot 2^{N-1} < P$ (e.g. 32 < 100 at the defaults, N = 3); $\min(6SP, L \cdot 2^{N-1-i})$ cap |

With $n = 5$ per group, treat the result as indicative. Sample standard
deviation (`ddof=1`) is reported; a formal test (Welch's $t$) is optional and
easy to add on the CSV.

## 8. Reading the outcome — decision rule

Let $\mu, \sigma$ be mean and std of `max_abs` (MXE) per variant, and
$\Delta = \mu_{\text{baseline}} - \mu_{\text{variant}}$.

| Observation | Interpretation | Action |
|---|---|---|
| $\sigma_{\text{baseline}}$ is comparable to the gaps seen in the earlier L / alpha sweeps | those sweeps were partly measuring seed luck | re-run the decisive settings over seeds before trusting them |
| `baseline_control` ≈ `baseline` | phase count / Adam resets are neutral | annealing effects can be read directly against either |
| `baseline_control` ≠ `baseline` | the phase structure matters on its own | compare annealed variants to `baseline_control` only |
| `anneal_k` improves $\mu$ by $> 2\sigma_{\text{pooled}}$ **and** lowers $\sigma$ | delayed K commitment helps and stabilises | adopt a three-stage `iterations` sequence as the default setting for this model |
| `anneal_kl` beats `anneal_k` | L commitment also matters | promote L annealing to a public option (follow-up) |
| improvement $< \sigma$ | annealing is within noise | keep baseline; use best-of-N seeds if $\sigma$ is large; consider the §11 follow-ups |
| annealed $\mu$ better but MAE worse or weights visibly lumpy | capacity re-distributed to outliers | inspect visually; try a longer final phase before rejecting |

Always look at MAE next to MXE, and at the weight maps for the best and
worst seed of the chosen variant (docs `evaluating_results.md`).

## 9. Checking the cost claim on your own hardware

The claim that annealing costs nothing extra per iteration rests on how
`topk` behaves: `torch.topk(x, k)` scans the full tensor `x` regardless of
`k`, so both projections ($\Pi_K$ on `W`, $\Pi_L$ on `B_rt`) cost the same
whatever K and L are — the per-iteration cost is flat in the sparsity
budget. This is worth confirming on the target device and dimensions before
relying on it, since kernel behaviour can differ by backend (CPU, CUDA,
MPS): time `compBX` plus the two projections at a few K and L values with
S, P, N held fixed. If the timings come out flat, the only cost an
annealing schedule adds is the extra phase count × `iterations`, which the
experiment script's `--total-iterations` equalises across variants; total
wall time then scales with steps/second on that hardware and the total step
count chosen, both of which are worth measuring once on the actual target
rather than assumed.

## 10. Why `power` and `alpha` are held fixed in this experiment

The data term's magnitude depends on $p$ *and* on how heavy-tailed the
residuals are, while the Laplacian term is always $L^2$. Measured on
synthetic residuals:

| stage of training | data term, $p=2$ | data term, $p=4$ | ratio |
|---|---|---|---|
| init (large, homogeneous) | 1.00e-02 | 1.74e-02 | 1.7× |
| mid (mild tail) | 1.13e-04 | 3.59e-04 | 3.2× |
| late (heavy tail, small) | 3.00e-06 | 1.50e-04 | 50× |

Raising $p$ therefore silently weakens the regulariser by up to ~50× as the
fit converges — a separate effect that would confound an annealing
experiment. Also, odd $p$ is invalid in this code (`pow` on the signed
residual, `model_fit.py:608`, gives a negative mean and NaN after
`.pow(2/p)`), so any later `power` experiment must use even values. These
are follow-ups (spec §4).

## 11. Follow-ups this plan deliberately leaves out

- Public L annealing (`anneal_nnz`) and a tunable halving factor.
- A longer final phase / finer tail — already possible via an explicit schedule.
- Continuous (linear or cubic [2]) ramps of K and L instead of a staircase.
- Regrowth (RigL [7]) to allow re-entry after a cut.
- Residual-target Laplacian $\alpha \lVert \mathbf{L}(\mathbf{B}\mathbf{C} - \mathbf{A})^{\top} \rVert^2$,
  which stops penalising genuine detail present in $\mathbf{A}$.
- `power=4` on the winning schedule, with alpha re-tuned upward.

## 12. References

1. S. Han, J. Pool, J. Tran, W. J. Dally. *Learning both Weights and
   Connections for Efficient Neural Networks.* NeurIPS 2015.
   https://arxiv.org/abs/1506.02626
2. M. Zhu, S. Gupta. *To prune, or not to prune: exploring the efficacy of
   pruning for model compression.* ICLR 2018 workshop.
   https://arxiv.org/abs/1710.01878
3. T. Gale, E. Elsen, S. Hooker. *The State of Sparsity in Deep Neural
   Networks.* 2019. https://arxiv.org/abs/1902.09574
4. J. Frankle, M. Carbin. *The Lottery Ticket Hypothesis: Finding Sparse,
   Trainable Neural Networks.* ICLR 2019. https://arxiv.org/abs/1803.03635
5. N. Parikh, S. Boyd. *Proximal Algorithms.* Foundations and Trends in
   Optimization 1(3), 2014. https://web.stanford.edu/~boyd/papers/prox_algs.html
6. D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.*
   ICLR 2015. https://arxiv.org/abs/1412.6980
7. U. Evci, T. Gale, J. Menick, P. S. Castro, E. Elsen. *Rigging the Lottery:
   Making All Tickets Winners.* ICML 2020. https://arxiv.org/abs/1911.11134
8. T. Blumensath, M. E. Davies. *Iterative hard thresholding for compressed
   sensing.* Applied and Computational Harmonic Analysis 27(3), 2009.
   https://doi.org/10.1016/j.acha.2009.04.002
9. PyTorch, *Reproducibility.* https://pytorch.org/docs/stable/notes/randomness.html
10. L. Kavan, J. Doublestein, M. Prazak, M. Cioffi, D. Roble. *Compressed
    Skinning for Facial Blendshapes.* SIGGRAPH 2024.
    https://arxiv.org/abs/2406.11597 — local copy
    `paper/compressed_skinning_for_facial_blendshapes.md`.
11. B. H. Le, Z. Deng. *Smooth Skinning Decomposition with Rigid Bones.*
    ACM TOG 31(6), SIGGRAPH Asia 2012. https://doi.org/10.1145/2366145.2366218
12. B. H. Le, Z. Deng. *Robust and Accurate Skeletal Rigging from Mesh
    Sequences.* ACM TOG 33(4), SIGGRAPH 2014. https://doi.org/10.1145/2601097.2601161
13. D. L. James, C. D. Twigg. *Skinning Mesh Animations.* ACM TOG 24(3),
    SIGGRAPH 2005. https://doi.org/10.1145/1073204.1073206
14. Electronic Arts, *Dem Bones.* https://github.com/electronicarts/dem-bones

## 13. Symbols

| Symbol | Meaning | Code name |
|---|---|---|
| $S$ | blendshapes | `model_data.n_blendshapes` |
| $N$ | vertices | `model_data.n_vertices` |
| $P$ | proxy joints | `number_of_bones` |
| $K$ | influences per vertex | `max_influences` |
| $L$ | non-zero delta coefficients | `total_nnz_B_rt` |
| $p$ | error-norm exponent | `power` |
| $\alpha$ | Laplacian weight | `alpha` |
| $\eta$ | Adam learning rate ($10^{-3}$) | `lr` in `train()` |
| $\mathbf{L}$ | rigidity Laplacian, $N \times N$ | `self.L` |
