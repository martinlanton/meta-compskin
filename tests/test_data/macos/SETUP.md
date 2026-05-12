# macOS Test Suite — Environment Documentation

All expected data files in this directory were generated on the following setup.
Tests using this data are skipped automatically on other platforms.

## Hardware

| Property | Value |
|----------|-------|
| Platform | macOS (darwin) |
| OS version | macOS 26.4.1 |
| Architecture | arm64 (Apple Silicon) |
| Processor | ARM (Apple M-series) |

## Software

| Package | Version |
|---------|---------|
| Python | 3.13.13 |
| PyTorch | 2.11.0 |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |

## Compute

| Property | Value |
|----------|-------|
| CUDA | Not available (CPU-only) |
| Device | CPU |
| Random seed | 12345 (set via `torch.manual_seed` in `SkinCompressor.__init__`) |

## Files

| File | Iterations | Content |
|------|-----------|---------|
| `expected_aura_600_iter.npz` | 600 | Compressed skinning weights and transforms (matrix results) |
| `expected_aura_10000_iter.npz` | 10 000 | Compressed skinning weights and transforms (matrix results) |
| `expected_aura_600_iter_vertices.npz` | 600 | Animated vertex positions for 30 farthest-point-sampled frames |
| `expected_aura_10000_iter_vertices.npz` | 10 000 | Animated vertex positions for 30 farthest-point-sampled frames |

## Notes

- Matrix results (`weights`, `shapeXform`, etc.) are non-transferable across platforms: CPU and
  CUDA use different floating-point paths and the Adam optimizer accumulates per-step differences
  over thousands of iterations, converging to a different-but-equivalent local minimum on each
  platform. The Windows/CUDA expected data lives in `tests/test_data/windows/`.
- Vertex positions are the higher-level correctness check: two runs that diverge in internal
  weights/transforms can still produce identical vertex positions if both are valid decompositions.
- The 30 test frames were selected via greedy farthest-point sampling in the 267-dimensional
  blendshape weight space to maximise deformation diversity across the test animation.