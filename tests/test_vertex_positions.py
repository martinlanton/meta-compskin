import sys
import unittest
from pathlib import Path

import numpy as np
import pytest
import torch

import metacompskin.rig.riglogic as rl
from metacompskin import model_fit
from metacompskin.animation_generator import AnimationFrameGenerator
from metacompskin.model_data import BlendshapeModelData

_MAX_CONTROL_WEIGHTS = 72

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="macOS/CPU test suite: expected data generated on macOS (Apple Silicon, CPU-only)",
)


class TestVertexPositions(unittest.TestCase):
    """Regression tests for animated vertex positions produced by the compressed skinning pipeline.

    Checks that the full pipeline — compression followed by skinning evaluation — produces
    vertex positions that match pre-computed expected values. This is a higher-level complement
    to the weight/transform tests in test_default_output.py: the same internal parameters could
    produce correct weights but broken skinning, which these tests catch.

    The 30 test frames are selected via farthest-point sampling in blendshape weight space,
    maximising deformation diversity across the test animation.
    """

    @classmethod
    def setUpClass(cls) -> None:
        location = Path(__file__).parent
        test_data_location = location / "test_data"

        model_file = test_data_location / "source_models" / "aura.npz"
        cls.model_data = BlendshapeModelData.from_npz(str(model_file))

        expected = np.load(str(test_data_location / "macos" / "expected_aura_600_iter_vertices.npz"))
        cls.expected_vertices = expected["vertices"]          # (30, N, 3)
        cls.frame_indices = expected["frame_indices"].tolist() # [int, ...]

        # Pre-compute blendshape weights for all test frames once.
        anim_data = np.load(test_data_location / "source_models" / "test_anim.npz")
        all_control_weights = anim_data["weights"][:, :_MAX_CONTROL_WEIGHTS].astype(np.float32)
        selected_control_weights = all_control_weights[cls.frame_indices]
        cls.blendshape_weights = rl.compute_rig_logic(
            torch.from_numpy(selected_control_weights).float(),
            cls.model_data.inbetween_info,
            cls.model_data.combination_info,
        ).numpy()

    def setUp(self) -> None:
        self.result_file = Path(__file__).parent / "test_result" / "test_result_vertices.npz"

    def tearDown(self) -> None:
        if self.result_file.exists():
            self.result_file.unlink()

    def test_vertex_positions_short_iter(self):
        compressor = model_fit.SkinCompressor(
            model_data=self.model_data, iterations=600
        )
        compressor.run(output_location=self.result_file)

        generator = AnimationFrameGenerator(
            compressed_data_path=self.result_file,
            model_data=self.model_data,
        )

        for i, anim_frame in enumerate(self.frame_indices):
            vertices = generator.compute_frame_vertices(self.blendshape_weights[i])
            with self.subTest(frame=anim_frame):
                np.testing.assert_array_equal(
                    self.expected_vertices[i], vertices
                )

    def test_vertex_positions_full_iter(self):
        location = Path(__file__).parent
        expected = np.load(
            str(location / "test_data" / "macos" / "expected_aura_10000_iter_vertices.npz")
        )
        expected_vertices = expected["vertices"]  # (30, N, 3)

        compressor = model_fit.SkinCompressor(
            model_data=self.model_data, iterations=10000
        )
        compressor.run(output_location=self.result_file)

        generator = AnimationFrameGenerator(
            compressed_data_path=self.result_file,
            model_data=self.model_data,
        )

        for i, anim_frame in enumerate(self.frame_indices):
            vertices = generator.compute_frame_vertices(self.blendshape_weights[i])
            with self.subTest(frame=anim_frame):
                np.testing.assert_array_equal(expected_vertices[i], vertices)


if __name__ == "__main__":
    unittest.main()