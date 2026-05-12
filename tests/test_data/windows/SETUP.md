# Windows Test Suite — Environment Documentation

All expected data files in this directory were generated on a Windows machine with CUDA.
Tests using this data are skipped automatically on other platforms.

## Hardware

| Property | Value |
|----------|-------|
| Platform | Windows (win32) |
| GPU | (to be documented) |

## Software

| Package | Version |
|---------|---------|
| Python | (to be documented) |
| PyTorch | (to be documented) |
| CUDA | (to be documented) |
| NumPy | (to be documented) |
| SciPy | (to be documented) |

## Compute

| Property | Value |
|----------|-------|
| Device | CUDA (GPU) |
| Random seed | 12345 (set via `torch.manual_seed` in `SkinCompressor.__init__`) |

## Files

| File | Iterations | Content |
|------|-----------|---------|
| `expected_aura_600_iter.npz` | 600 | Compressed skinning weights and transforms (matrix results) |
| `expected_aura_10000_iter.npz` | 10 000 | Compressed skinning weights and transforms (matrix results) |

## Notes

- To regenerate these files on Windows, run `SkinCompressor` with the appropriate iteration
  count and save the output NPZ. Ensure PyTorch and CUDA versions match those documented above.
- See `tests/test_data/macos/SETUP.md` for the macOS equivalent and an explanation of why
  expected data is platform-specific.