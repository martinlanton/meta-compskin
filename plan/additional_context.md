The experiments mentioned here (and with the failing NaN values) were run on proprietary meshes not present in the current repo. The meshes have roughly 130 shapes, 12000 vtcs.


# Summary from another AI session on those results:

<test07 context>
Root cause: why the divergence isn't a schedule-shape problem

test07 - no mouth geometry, custom joints (P=215), alpha=7, power=2, L=40000, max_influences=8

## What was run:

| # | Run | iterations | status | note |
|---|---|---|---|---|
| 1 | flat 1000000 | 1000000 | skipped | pre empted by test06 evidence
| 2 | annealed 500000 x3 | (500000, 500000, 500000) | failed (NaN) | 5.39h, diverged in phase 2/4

flat 1000000 was skipped before running based on previous test's finding that the same flat k=8 from the start shape diverged reproducibly at only 10000 iterations - no reason to expect 100x more iterations would fix the structural failure.
annealed 500k x3 was run to completion anyway, since the equivalent small scale annealed schedule had succeeded. it ran the full 5.39h and also diverged to NaN - this report is about why, traced through the actual solver code.

## Where it failed
phase 1/4 : iterations=500000 K=16 L=40000 normalize_weights=False (warm up, unnormalized)
... converges normally over 500k steps ...
phase 2/4: iterations=500000 K=16 L=40000normalize_weights=True (first normalized stage)
00000(0.016) nan nan 0 182602
00200(1.949) nan nan 0 0

Note this is at K=16, the loosest annealed budget, not at =8 stage. In the previous test (5000 iterations stages x3) the same phase completed fine. The only difference is the preceding warm up stage is 100x longer here (500k vs 5k)


## The mechanism trace through the code
1. geodesic mask always includes something. Geodesic_joint_distances() computes per vertex the shortest path along mesh edges to every joint (dijkstra), snapping each joint to its nearest vertex first. Even a vertex whose "true" nearby joints are now unreachable through the mesh (because the mouth bag cut removed the surface path) still gets assigned some M nearest joints by candidate_joint_mask() - the mask is never empty, so the failure is not "zero candidates".
2. the per iteration projection can zero out weights within that mask. every training step keeps only the k largest weights per vtx and clamps the rest - including any negative values - to zero. if none of a vertex's masked candidates actually explains its motion well (plausible for a vertex near the cut, whose real driving deformation used to route through now missing geometry) the optimizer has no good reason tyo keep any of them non zero: pushing all of that vertex's weights toward zero can genuinely reduce the loss, since there's no useful signal to preserve.
3. more unnormalized steps means more time to decay to exact 0.0. during the unnormalized warm up, weights are floating point values with no constraint pulling them back up. a 5k warm up isn't long enough to fully underflow a struggling vertex's weights to bit exact 0; a 500k step warm up is 100x the opportunity for that decay to complete.
4. the instant normalization turns on, it's a division by 0. (line 1069, model_fit.py).for the one vertex whose column of W is now all exactly 0, W.sum(dim=0) is 0 for that column, and the div produces NaN. That single nan propagates through the shared matrix multiplication in the very next forward pass and corrupts every other vertex's weight on the next backward step - which is why the whole run shows nan nan 0 0 within 200 iterations, not just on bad column
</test07 context>

<test08 context>
Test 07 was identical to test 07, but with removing the joints that were not near the geometry (tongue joints, etc).
P=208, iterations=100k, failed to NaN.

## Why removing joints wouldn't be expected to help, on reflection
the candidate count per vertex, M= min(2K, P-1) is 16 either way = at P=215 or P=208, 2K=16 is still the binding term since P-1 is far larger in both cases. removing 7 joints from the pool doesn't reduce how many candidates a vertex is assigned; it only changes which 16 joints fill those slots. If the rue proble; is that a vertex near the cut has no genuinely useful joint within geodesic reach at all - not that a couple of specific joints were wrongly assigned to it - then swapping which 16 candidates it gerts doesn't change anything: whichever it's given, none of them explain its motion, and the same zero-weight decay plays out on the same timescale.
</test08 context>


Based on those results, one more question arises : wouldn't forcing assignment of non moving vertices to the head joint solve this problem altogether? This could be the result of non-moving vertices inside the mouth near the cut that have no joint to get assigned to because all nearby joints are moving based on moving nearby vertices.