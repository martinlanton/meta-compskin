# Summary of experiment results

## E1
Using the solver's own public functions directly on the already exported no-mouth model and its 215 joints - no training required. For every vertex, computed the distance to its nearest candidate and to the farthest of its M=16 kept candidates.

Clean result : the 14 worst served vertices cluster almost perfectly at one location: x spread from -8.6 to +7.2, Y~-4, Z -14.2 to -14.5. Tight horizontal band, not scattered noise, signature of an edge loop. most likely boundary rim left behind where the mouth bag was cut away. Their farthest-of-16 candidate distance (27-28 model units) is roughly 4/5 times the mesh wide average.

Directly confirms geometric mechanism suspected in prior report: cutting the mouth bag removed the surface connections several joints used to reach the lip/mouth corner region, and the remaining boundary vertices are geodesically stranded from any joint that could possibly drive them.

## E1 additional experiment
NaN risk appears to not be specific to cut mesh.

parameters: full head (no cut), 215 joints, alpha 7, power 2, L 40000, K 8
run 1 : non annealed, flat 1M iterations (+1M warm up iterations, based on code)
run 2 : annealed, 500k iterations x3 (+500k warm up iterations, based on code)

results:

| run | iterations | mean delta | max delta | time | status |
|---|---|---|---|---|---|
| flat 1M | 1M | NaN | NaN | 5.26h | Failed |
| annealed 500k x3 | (500000, 500000, 500000) | 0.002599 | 0.5307 | 5.38h | ok |

Verdict:
non mesh specific : flat 1M diverged to NaN on the full head. Exact same failure as every prior occurrence. Normalized phase collapses to `nan nan 0 0` within 200 iterations of starting. Mesh has no cut boundary (beyond the neck) and the exact same process ran fine in 10000 iterations in previous tests.
Given enough unnormalized steps some vertices' weights still decayed to exact 0.

Given the different tests, flat iteration number appears to be the common denominator, that said the mouth cut appears to change the number at which the divergence happens.

This appears to be stronger evidence for R1 (zero transform root influence) than previous tests suggested. Not a special case that only shows on damaged/cut geometry - general property of combining the geodesic candidate filter with a long un-normalized warm up on any mesh. The cut mesh simply has at least one vertex bad enough to hit it almost immediately, which is what made it easy to find. Teh full head (and potentially any other mesh) has the same fragility, just at a much higher (but not infinite) iteration budget.

<user_note>
Geodesic filter distance was an attempt to fix problem #5. Removing the filter is totally fine as long as this is replaced with a proper solution to #5.
</user_note>

## E2 - does a tighter candidate mask M help?

| M | mean delta | max delta | status |
|---|---|---|---|
| 9 | nan | nan | failed |
| 10 | nan | nan | failed |
| 16 | nan | nan | failed |

## E3 - warm up on/off
Need unimplemented parameter.

## E4 - how much does seed alone move the numbers

| seed | mean delta | max delta |
|---|---|---|
| 12345 | 0.002512 | 0.2784 |
| 1 | 0.002562 | 0.2818 |
| 2 | 0.002543 | 0.2718 |

Important calibration: 3 seeds of the same otherwise-identical run spread mean delta by about 2% and max delta by about 3.6%. Previous conclusions in tests (1 to 2.1%) appear to be within noise, not confirmed evidence, until re tested across seeds. Differences of 7+% (such as annealing implementation's +18%) remain well outside the noise band.

## E5
Needs unimplemented solver mode.

## E6
Needs unimplemented loss term.

## E7
Histogram not available.

Summary :
Prediction not confirmed. Correlation between a joint's distance from centre and its own rotation/translation energy ratio was +0.26 - the wrong sign. $3.2 predicted a negative correlation (rotation coefficients pruned relatively harder for distant joints, since rotation about a far pivot costs more non-zero coefficients). In absolute terms, near-centre joints do carry far more non-zero coefficients overall (rotation AND translation) than far ones - but that looks like near-centre joints simply doing more overall work, not a rotation-specific pruning effect. This particular argument for R2's joint-centered pivot change is not supported by this dataset; R2 may still be worth building for the locality/hierarchy benefits in $5.2, but not on this specific justification.
