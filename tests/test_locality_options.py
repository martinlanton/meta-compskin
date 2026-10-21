"""Tests for the locality and accuracy options of the joint path.

Covers the start falloff, the distance penalty on the weights, the
residual-target smoothing and pinning still vertices to the root.
"""

import numpy as np
import pytest
from conftest import make_grid_model, write_model_npz

from metacompskin.cli import main
from metacompskin.maya_pipeline import CompressionSettings, compression_command
from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import (
    SkinCompressor,
    distance_penalty_weights,
    distance_weights,
    geodesic_joint_distances,
)

_FAST = {"iterations": 200, "total_nnz_B_rt": 60, "max_influences": 2}


def _grid_joints(n_joints: int = 6) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(5.0), np.arange(5.0))
    points = np.stack([xs.ravel(), ys.ravel(), np.zeros(25)], axis=1)[::4][:n_joints]
    matrices = np.tile(np.eye(4), (n_joints, 1, 1))
    matrices[:, :3, 3] = points
    return matrices


def _with_still_region(
    model: BlendshapeModelData, scale: float = 0.0
) -> BlendshapeModelData:
    deltas = model.deltas.copy()
    deltas[:, 20:, :] *= scale  # the last row of the grid barely moves (or not at all)
    return BlendshapeModelData(
        model_name=model.model_name,
        deltas=deltas,
        rest_verts=model.rest_verts,
        rest_faces=model.rest_faces,
        inbetween_info=model.inbetween_info,
        combination_info=model.combination_info,
        alpha=model.alpha,
    )


def _distances(model: BlendshapeModelData) -> np.ndarray:
    return geodesic_joint_distances(
        model.rest_verts, model.rest_faces, _grid_joints()[:, :3, 3]
    )


def _nearest_share(weights: np.ndarray, model: BlendshapeModelData) -> float:
    """Share of the joints' weight that sits on each vertex's nearest joint."""
    joint_weights = weights[:, :-1].T  # drop the synthetic root column, (P, N)
    nearest = _distances(model).argmin(axis=0)
    return float(
        joint_weights[nearest, np.arange(joint_weights.shape[1])].sum()
        / joint_weights.sum()
    )


def _run(tmp_path, name, model=None, **settings) -> np.ndarray:
    output = tmp_path / f"{name}.npz"
    SkinCompressor(
        model_data=model or make_grid_model(),
        rest_joint_matrices=_grid_joints(),
        **{**_FAST, **settings},
    ).run(output)
    return np.load(output)["weights"]


class TestStartFalloff:
    distances = np.array([[0.0, 3.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])  # (P=4, N=2)

    def test_the_default_falloff_is_the_square(self):
        weights = distance_weights(self.distances, max_influences=3)

        expected = np.array([1.0, (1 - 1 / 3) ** 2, (1 - 2 / 3) ** 2])
        np.testing.assert_allclose(weights[:3, 0], expected / expected.sum())

    def test_a_higher_power_concentrates_the_weight(self):
        smooth = distance_weights(self.distances, max_influences=3)
        sharp = distance_weights(self.distances, max_influences=3, falloff_power=4)

        assert sharp[0, 0] > smooth[0, 0]
        np.testing.assert_allclose(sharp.sum(axis=0), 1.0)


class TestDistancePenaltyWeights:
    def test_zero_on_the_nearest_joint_and_growing_with_distance(self):
        distances = np.array([[1.0], [2.0], [3.0]])

        rho = distance_penalty_weights(distances)

        np.testing.assert_allclose(rho[:, 0], [0.0, 1.0, 4.0])

    def test_independent_of_the_scene_unit(self):
        distances = np.array([[1.0, 4.0], [2.0, 1.0], [5.0, 3.0]])

        np.testing.assert_allclose(
            distance_penalty_weights(distances),
            distance_penalty_weights(10 * distances),
        )

    def test_a_joint_sitting_on_the_vertex_does_not_explode(self):
        rho = distance_penalty_weights(np.array([[0.0], [1.0]]), floor=0.5)

        np.testing.assert_allclose(rho[:, 0], [0.0, 1.0])


class TestDistancePenaltySetting:
    def test_needs_joint_matrices(self, grid_model_data):
        with pytest.raises(ValueError, match="rest_joint_matrices"):
            SkinCompressor(
                model_data=grid_model_data, number_of_bones=10, distance_penalty=0.1
            )

    def test_must_not_be_negative(self, grid_model_data):
        with pytest.raises(ValueError, match="distance_penalty"):
            SkinCompressor(
                model_data=grid_model_data,
                rest_joint_matrices=_grid_joints(),
                distance_penalty=-1.0,
            )

    def test_zero_changes_nothing(self, tmp_path):
        np.testing.assert_array_equal(
            _run(tmp_path, "off", distance_penalty=0.0), _run(tmp_path, "default")
        )

    def test_a_strong_penalty_moves_weight_to_the_nearest_joint(self, tmp_path):
        model = make_grid_model()
        free = _run(
            tmp_path, "free", model, weight_init="noise", candidate_joints_per_vertex=4
        )
        pulled = _run(
            tmp_path,
            "pulled",
            model,
            weight_init="noise",
            candidate_joints_per_vertex=4,
            distance_penalty=10.0,
        )

        assert _nearest_share(pulled, model) > _nearest_share(free, model) + 0.05


class TestSmoothing:
    def test_unknown_values_are_rejected(self, grid_model_data):
        with pytest.raises(ValueError, match="smoothing"):
            SkinCompressor(model_data=grid_model_data, smoothing="weights")

    def test_reconstruction_is_the_default(self, tmp_path):
        np.testing.assert_array_equal(
            _run(tmp_path, "explicit", smoothing="reconstruction"),
            _run(tmp_path, "default"),
        )

    def test_residual_smoothing_does_not_blur_the_fit(self, tmp_path):
        model = make_grid_model()
        strong = BlendshapeModelData(
            model_name=model.model_name,
            deltas=model.deltas,
            rest_verts=model.rest_verts,
            rest_faces=model.rest_faces,
            inbetween_info={},
            combination_info={},
            alpha=1000.0,
        )
        errors = {}
        for smoothing in ("reconstruction", "residual"):
            compressor = SkinCompressor(
                model_data=strong,
                rest_joint_matrices=_grid_joints(),
                smoothing=smoothing,
                **{**_FAST, "iterations": 400},
            )
            compressor.run(tmp_path / f"{smoothing}.npz")
            errors[smoothing] = compressor.reconstruction_error.mean_abs

        assert errors["residual"] < errors["reconstruction"]


class TestStillVertices:
    def test_need_a_root(self, grid_model_data):
        with pytest.raises(ValueError, match="root"):
            SkinCompressor(
                model_data=grid_model_data,
                rest_joint_matrices=_grid_joints(),
                root_joint="none",
                still_vertices="static",
            )

    @pytest.mark.parametrize("value", ["quiet", -0.1])
    def test_invalid_values_are_rejected(self, grid_model_data, value):
        with pytest.raises(ValueError, match="still_vertices"):
            SkinCompressor(
                model_data=grid_model_data,
                rest_joint_matrices=_grid_joints(),
                still_vertices=value,
            )

    def test_static_vertices_are_pinned_to_the_root(self, tmp_path):
        model = _with_still_region(make_grid_model())
        compressor = SkinCompressor(
            model_data=model,
            rest_joint_matrices=_grid_joints(),
            still_vertices="static",
            **_FAST,
        )
        compressor.run(tmp_path / "pinned.npz")

        weights = np.load(tmp_path / "pinned.npz")["weights"]
        assert weights.shape == (25, 7)
        np.testing.assert_array_equal(weights[20:, -1], 1.0)
        np.testing.assert_array_equal(weights[20:, :-1], 0.0)
        assert (weights[:20, :-1].sum(axis=1) > 0).all()

    def test_a_distance_threshold_pins_slow_vertices(self, tmp_path):
        model = _with_still_region(make_grid_model(), scale=1e-3)
        largest = np.linalg.norm(model.deltas, axis=2).max()
        output = tmp_path / "slow.npz"
        SkinCompressor(
            model_data=model,
            rest_joint_matrices=_grid_joints(),
            still_vertices=0.01 * largest,
            **_FAST,
        ).run(output)

        weights = np.load(output)["weights"]
        np.testing.assert_array_equal(weights[20:, -1], 1.0)

    def test_the_reported_error_covers_every_vertex(self, tmp_path):
        model = _with_still_region(make_grid_model(), scale=1e-3)
        largest = np.linalg.norm(model.deltas, axis=2).max()
        compressor = SkinCompressor(
            model_data=model,
            rest_joint_matrices=_grid_joints(),
            still_vertices=0.01 * largest,
            **_FAST,
        )
        compressor.run(tmp_path / "slow.npz")

        result = np.load(tmp_path / "slow.npz")
        rest_h = np.concatenate([result["rest"], np.ones((25, 1))], axis=1)
        weighted = (
            (result["weights"].T[:, :, None] * rest_h[None])
            .transpose(0, 2, 1)
            .reshape(-1, 25)
        )
        rebuilt = (result["shapeXform"] @ weighted).T.reshape(25, -1, 3)
        error = np.abs(rebuilt - model.deltas.transpose(1, 0, 2))
        assert np.isclose(
            error.mean(), compressor.reconstruction_error.mean_abs, rtol=1e-4
        )
        assert np.isclose(
            error.max(), compressor.reconstruction_error.max_abs, rtol=1e-4
        )

    def test_pinning_nothing_changes_nothing(self, tmp_path):
        np.testing.assert_array_equal(
            _run(tmp_path, "none_pinned", still_vertices=0.0), _run(tmp_path, "default")
        )


class TestCommandLine:
    def _model(self, tmp_path):
        model = _with_still_region(make_grid_model())
        return write_model_npz(
            tmp_path / "grid.npz", model, rest_joint_matrices=_grid_joints()
        )

    def test_every_option_is_accepted(self, tmp_path):
        output = tmp_path / "out.npz"

        main(
            [
                str(self._model(tmp_path)),
                str(output),
                "--iterations",
                "200",
                "--total-nnz-b-rt",
                "60",
                "--max-influences",
                "2",
                "--distance-penalty",
                "0.1",
                "--smoothing",
                "residual",
                "--still-vertices",
                "static",
            ]
        )

        np.testing.assert_array_equal(np.load(output)["weights"][20:, -1], 1.0)

    def test_still_vertices_takes_a_distance(self, tmp_path):
        output = tmp_path / "out.npz"

        main(
            [
                str(self._model(tmp_path)),
                str(output),
                "--iterations",
                "200",
                "--total-nnz-b-rt",
                "60",
                "--max-influences",
                "2",
                "--still-vertices",
                "0.001",
            ]
        )

        np.testing.assert_array_equal(np.load(output)["weights"][20:, -1], 1.0)

    def test_unknown_smoothing_is_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            main(
                [
                    str(self._model(tmp_path)),
                    str(tmp_path / "o.npz"),
                    "--smoothing",
                    "weights",
                ]
            )


class TestPipelineCommand:
    def test_the_defaults_add_nothing(self, tmp_path):
        command = compression_command(
            "python", tmp_path / "m.npz", tmp_path / "c.npz", CompressionSettings()
        )

        for flag in ("--distance-penalty", "--smoothing", "--still-vertices"):
            assert flag not in command

    def test_the_options_are_forwarded(self, tmp_path):
        settings = CompressionSettings(
            distance_penalty=0.05, smoothing="residual", still_vertices="static"
        )

        command = compression_command(
            "python", tmp_path / "m.npz", tmp_path / "c.npz", settings
        )

        assert command[command.index("--distance-penalty") + 1] == "0.05"
        assert command[command.index("--smoothing") + 1] == "residual"
        assert command[command.index("--still-vertices") + 1] == "static"
