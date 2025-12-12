import unittest
from pathlib import Path

import numpy as np

from metacompskin import model_fit
from metacompskin.model_data import BlendshapeModelData


class TestCompskin(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        location = Path(__file__).parent
        test_data_location = location / "test_data"
        cls.result_data_location = location / "test_result"

        expected_file = test_data_location / "expected_aura.npz"
        cls.expected_data = np.load(str(expected_file))
        expected_file_short = test_data_location / "expected_aura_600_iter.npz"
        cls.expected_data_short = np.load(str(expected_file_short))

        # Load the model data once for all tests
        model_file = Path(__file__).parent / "test_data" / "source_models" / "aura.npz"
        cls.model_data = BlendshapeModelData.from_npz(str(model_file))

    def setUp(self) -> None:
        self.output_filename = "test_result.npz"
        self.result_file = self.result_data_location / self.output_filename

    def tearDown(self) -> None:
        if self.result_file.exists():
            self.result_file.unlink()

    def test_default_output_short_iter(self):
        compressor = model_fit.SkinCompressor(
            model_data=self.model_data, iterations=600
        )
        compressor.run(output_location=self.result_file)

        result_data = np.load(str(self.result_file))

        for key in result_data.files:
            with self.subTest(key=key):
                np.testing.assert_array_equal(
                    self.expected_data_short[key], result_data[key]
                )

    def test_default_output(self):
        compressor = model_fit.SkinCompressor(
            model_data=self.model_data,
            iterations=10000,
        )
        compressor.run(output_location=self.result_file)

        result_data = np.load(str(self.result_file))

        for key in result_data.files:
            with self.subTest(key=key):
                np.testing.assert_array_equal(self.expected_data[key], result_data[key])


if __name__ == "__main__":
    unittest.main()
