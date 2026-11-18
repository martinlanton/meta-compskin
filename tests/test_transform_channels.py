"""Tests for the joint transform options: channels, the rotation penalty, the translation unit.

Each joint's change in each shape is a 3 x 4 matrix built from a few channels.
Six (rotation and translation, the paper's) is the default; nine adds a scale
along each axis and twelve adds the three symmetric shears, which together span
every affine change. The rotation penalty keeps the linearised rotations small;
the translation unit measures translations in the head's own size.
"""

import numpy as np
import pytest
from conftest import make_grid_model

from metacompskin.cli import _compressor_settings as compressor_settings
from metacompskin.cli import build_parser
from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import CompressedSkinningSolver

_FAST = {
    "iterations": 300,
    "total_nnz_B_rt": 24,
    "max_influences": 2,
}  # 24 fits one shape


def _grid_joints(n_joints: int = 4) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(5.0), np.arange(5.0))
    points = np.stack([xs.ravel(), ys.ravel(), np.zeros(25)], axis=1)[::6][:n_joints]
    matrices = np.tile(np.eye(4), (n_joints, 1, 1))
    matrices[:, :3, 3] = points
    return matrices


def _solver(model=None, **settings) -> CompressedSkinningSolver:
    return CompressedSkinningSolver(
        model_data=model or make_grid_model(),
        rest_joint_matrices=_grid_joints(),
        **{**_FAST, **settings},
    )


def _stretched_model() -> BlendshapeModelData:
    """The grid with one shape that stretches it 20% along x about its centre."""
    model = make_grid_model()
    rest = np.asarray(model.rest_verts)
    deltas = np.zeros((1, len(rest), 3), dtype=np.float32)
    deltas[0, :, 0] = 0.2 * (rest[:, 0] - rest[:, 0].mean())
    return BlendshapeModelData(
        model_name="stretched_grid",
        deltas=deltas,
        rest_verts=model.rest_verts,
        rest_faces=model.rest_faces,
        inbetween_info={},
        combination_info={},
        alpha=None,
    )


def _rotation_size(output) -> float:
    """Mean size of the linearised rotation in every joint change of a result."""
    data = np.load(output)
    n_joints = data["weights"].shape[1]
    blocks = data["shapeXform"].reshape(-1, 3, n_joints, 4)[..., :3]  # (S, 3, P, 3)
    blocks = blocks.transpose(0, 2, 1, 3)  # (S, P, 3, 3)
    skew = 0.5 * (blocks - blocks.transpose(0, 1, 3, 2))
    return float(np.abs(skew).mean())


class TestChannels:
    def test_six_by_default(self):
        assert _solver().buildTR().shape[0] == 6

    @pytest.mark.parametrize(("channels", "rank"), [(6, 6), (9, 9), (12, 12)])
    def test_each_channel_adds_an_independent_change(self, channels, rank):
        basis = _solver(transform_channels=channels).buildTR().reshape(channels, 12)

        assert np.linalg.matrix_rank(basis.cpu().numpy()) == rank

    def test_nine_adds_a_scale_along_each_axis(self):
        extra = _solver(transform_channels=9).buildTR().reshape(9, 3, 4)[6:]

        np.testing.assert_array_equal(
            extra[:, :, :3].cpu().numpy(), np.eye(3)[:, None] * np.eye(3)
        )

    def test_twelve_adds_the_symmetric_shears(self):
        shears = (
            _solver(transform_channels=12)
            .buildTR()
            .reshape(12, 3, 4)[9:, :, :3]
            .cpu()
            .numpy()
        )

        np.testing.assert_array_equal(shears, shears.transpose(0, 2, 1))
        assert np.all(np.diagonal(shears, axis1=1, axis2=2) == 0)

    def test_scales_fit_a_stretch_better_than_rotations_alone(self, tmp_path):
        model = _stretched_model()
        errors = {}
        for channels in (6, 9):
            solver = _solver(model, transform_channels=channels)
            solver.run(tmp_path / f"c{channels}.npz")
            errors[channels] = solver.reconstruction_error.mean_abs

        assert errors[9] < 0.5 * errors[6]

    @pytest.mark.parametrize("channels", [0, 7, 15])
    def test_only_six_nine_or_twelve(self, channels):
        with pytest.raises(ValueError, match="transform_channels"):
            _solver(transform_channels=channels)

    def test_scales_and_shears_do_not_go_through_a_joint_hierarchy(self):
        with pytest.raises(ValueError, match="transform_channels"):
            _solver(transform_channels=9, joint_parents=[-1, 0, 0, 0])


class TestRotationPenalty:
    def test_off_by_default(self, tmp_path):
        _solver().run(tmp_path / "default.npz")
        _solver(rotation_penalty=0.0).run(tmp_path / "off.npz")

        np.testing.assert_array_equal(
            np.load(tmp_path / "default.npz")["shapeXform"],
            np.load(tmp_path / "off.npz")["shapeXform"],
        )

    def test_a_strong_penalty_shrinks_the_rotations(self, tmp_path):
        _solver().run(tmp_path / "free.npz")
        _solver(rotation_penalty=1000.0).run(tmp_path / "held.npz")

        assert _rotation_size(tmp_path / "held.npz") < 0.5 * _rotation_size(
            tmp_path / "free.npz"
        )

    def test_must_not_be_negative(self):
        with pytest.raises(ValueError, match="rotation_penalty"):
            _solver(rotation_penalty=-1.0)


class TestTranslationUnit:
    def test_model_units_by_default(self):
        translations = _solver().buildTR().reshape(6, 3, 4)[3:, :, 3].cpu().numpy()

        np.testing.assert_array_equal(translations, np.eye(3))

    def test_head_measures_translations_in_the_heads_size(self):
        model = make_grid_model()
        rest = np.asarray(model.rest_verts, dtype=np.float64)
        radius = np.sqrt(((rest - rest.mean(axis=0)) ** 2).sum(axis=1).mean())

        translations = (
            _solver(model, translation_unit="head").buildTR().reshape(6, 3, 4)[3:, :, 3]
        )

        np.testing.assert_allclose(
            translations.cpu().numpy(), radius * np.eye(3), rtol=1e-6
        )

    def test_rejects_other_units(self):
        with pytest.raises(ValueError, match="translation_unit"):
            _solver(translation_unit="metres")


class TestCommandLine:
    def test_passes_the_transform_options_to_the_solver(self):
        args = build_parser().parse_args(
            [
                "in.npz",
                "out.npz",
                "--transform-channels",
                "12",
                "--rotation-penalty",
                "3",
                "--translation-unit",
                "head",
            ]
        )

        settings = compressor_settings(args)

        assert settings["transform_channels"] == 12
        assert settings["rotation_penalty"] == 3.0
        assert settings["translation_unit"] == "head"

    def test_leaves_the_solver_defaults_when_not_given(self):
        settings = compressor_settings(build_parser().parse_args(["in.npz", "out.npz"]))

        assert (
            not {"transform_channels", "rotation_penalty", "translation_unit"}
            & settings.keys()
        )
