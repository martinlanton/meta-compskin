from pathlib import Path
import numpy as np
import unittest
from metacompskin import model_fit


class TestCompskin(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        location = Path(__file__).parent
        test_data_location = location / "test_data"

        expected_file = test_data_location / "expected_aura.npz"
        cls.expected_data = np.load(str(expected_file))
        expected_file_short = test_data_location / "expected_aura_600_iter.npz"
        cls.expected_data_short = np.load(str(expected_file_short))

    def setUp(self) -> None:
        self.output_filename = "test_result"
        self.output_filename_ext = self.output_filename + ".npz"
        self.result_file = model_fit.out_location / self.output_filename_ext

    def tearDown(self) -> None:
        if self.result_file.exists():
            self.result_file.unlink()

    def test_default_output_short_iter(self):
        compressor = model_fit.SkinCompressor(iterations=600)
        compressor.model = "aura"
        compressor.run(output_filename=self.output_filename)

        result_data = np.load(str(self.result_file))

        for key in result_data.files:
            with self.subTest(key=key):
                np.testing.assert_array_equal(
                    self.expected_data_short[key], result_data[key]
                )

    def test_default_output(self):
        compressor = model_fit.SkinCompressor(iterations=10000)
        compressor.run(output_filename=self.output_filename)

        result_data = np.load(str(self.result_file))

        for key in result_data.files:
            with self.subTest(key=key):
                np.testing.assert_array_equal(self.expected_data[key], result_data[key])


if __name__ == "__main__":
    unittest.main()
