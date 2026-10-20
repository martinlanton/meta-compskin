"""Tests for solving joint deltas through a joint hierarchy."""

import numpy as np
import pytest
import torch
from conftest import make_grid_model, write_model_npz

from metacompskin.cli import main
from metacompskin.maya_exporter import nearest_listed_ancestors
from metacompskin.maya_pipeline import CompressionSettings, compression_command
from metacompskin.model_fit import (
    SkinCompressor,
    accumulate_joint_transforms,
    hierarchical_coefficients,
    joint_chain_matrix,
)

_FAST = ["--iterations", "200", "--total-nnz-b-rt", "100"]
# Six joints on the 5x5 grid: a chain 0 -> 1 -> 2 and a chain 3 -> 4 -> 5.
_PARENTS = [-1, 0, 1, -1, 3, 4]


def _joint_matrices(n_joints: int = 6) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(5.0), np.arange(5.0))
    points = np.stack([xs.ravel(), ys.ravel(), np.zeros(25)], axis=1)[::4][:n_joints]
    matrices = np.tile(np.eye(4), (n_joints, 1, 1))
    matrices[:, :3, 3] = points
    return matrices


def _compressor(**settings) -> SkinCompressor:
    return SkinCompressor(
        model_data=make_grid_model(),
        iterations=200,
        rest_joint_matrices=_joint_matrices(),
        max_influences=2,
        total_nnz_B_rt=60,
        **settings,
    )


class TestJointChainMatrix:
    def test_each_joint_sums_itself_and_its_ancestors(self):
        chain = joint_chain_matrix([-1, 0, 1, 0])

        np.testing.assert_array_equal(
            chain,
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 1]],
        )

    def test_parents_may_come_after_their_children(self):
        chain = joint_chain_matrix([2, -1, 1])

        np.testing.assert_array_equal(chain[0], [1, 1, 1])

    @pytest.mark.parametrize(
        "parents",
        [[0], [-1, 5], [-1, -2], [1, 0], [-1, 2, 1]],
        ids=["self", "out-of-range", "below-minus-one", "two-cycle", "three-cycle"],
    )
    def test_invalid_hierarchies_are_rejected(self, parents):
        with pytest.raises(ValueError):
            joint_chain_matrix(parents)


class TestHierarchicalCoefficients:
    def test_a_local_rotation_turns_about_the_joint_itself(self):
        pivots = torch.tensor([[2.0, -1.0, 3.0]])
        local = torch.zeros(6, 1, 1, 1, 1)
        local[:3, 0, 0, 0, 0] = torch.tensor([0.1, -0.2, 0.3])  # rotation only

        coefficients = hierarchical_coefficients(local, pivots, torch.eye(1))

        rotation = coefficients[:3, 0, 0, 0, 0]
        translation = coefficients[3:, 0, 0, 0, 0]
        moved = torch.linalg.cross(rotation, pivots[0]) + translation
        torch.testing.assert_close(moved, torch.zeros(3))

    def test_a_child_inherits_its_parents_delta(self):
        pivots = torch.zeros(2, 3)
        local = torch.zeros(6, 1, 2, 1, 1)
        local[3, 0, 0, 0, 0] = 1.0  # parent translates along x
        chain = torch.from_numpy(joint_chain_matrix([-1, 0])).float()

        coefficients = hierarchical_coefficients(local, pivots, chain)

        torch.testing.assert_close(coefficients[:, 0, 1], coefficients[:, 0, 0])

    def test_an_identity_chain_at_the_origin_changes_nothing(self):
        local = torch.randn(6, 3, 4, 1, 1)

        coefficients = hierarchical_coefficients(local, torch.zeros(4, 3), torch.eye(4))

        torch.testing.assert_close(coefficients, local)


class TestAccumulateJointTransforms:
    def test_a_childs_transform_adds_its_parents(self):
        local = np.zeros((3, 8))  # one shape, two joints
        local[:, 0:4] = 1.0
        local[:, 4:8] = 2.0

        accumulated = accumulate_joint_transforms(local, [-1, 0])

        np.testing.assert_array_equal(accumulated[:, 0:4], 1.0)
        np.testing.assert_array_equal(accumulated[:, 4:8], 3.0)


class TestSkinCompressorJointParents:
    def test_parents_need_joint_matrices(self):
        with pytest.raises(ValueError, match="rest_joint_matrices"):
            SkinCompressor(
                model_data=make_grid_model(), number_of_bones=6, joint_parents=_PARENTS
            )

    def test_parents_need_one_entry_per_joint(self):
        with pytest.raises(ValueError, match="one parent per joint"):
            _compressor(joint_parents=_PARENTS[:-1])

    def test_an_invalid_hierarchy_is_rejected_at_construction(self):
        with pytest.raises(ValueError):
            _compressor(joint_parents=[1, 0, 1, -1, 3, 4])

    def test_without_parents_the_output_keeps_its_keys(self, tmp_path):
        output = tmp_path / "flat.npz"
        _compressor().run(output)

        assert set(np.load(output).files) == {
            "rest",
            "quads",
            "weights",
            "restXform",
            "shapeXform",
        }

    def test_run_writes_the_hierarchy_and_its_local_deltas(self, tmp_path):
        output = tmp_path / "hierarchy.npz"
        _compressor(joint_parents=_PARENTS).run(output)

        result = np.load(output)
        np.testing.assert_array_equal(result["jointParents"], [*_PARENTS, -1])  # + root
        assert result["localShapeXform"].shape == result["shapeXform"].shape

    def test_accumulated_local_deltas_reproduce_the_joint_transforms(self, tmp_path):
        output = tmp_path / "hierarchy.npz"
        _compressor(joint_parents=_PARENTS).run(output)

        result = np.load(output)
        accumulated = accumulate_joint_transforms(
            result["localShapeXform"], result["jointParents"]
        )
        np.testing.assert_allclose(accumulated, result["shapeXform"], atol=1e-6)

    def test_the_delta_budget_applies_to_the_local_deltas(self, tmp_path):
        output = tmp_path / "hierarchy.npz"
        _compressor(joint_parents=_PARENTS).run(output)

        local = np.load(output)["localShapeXform"]  # (3S, 4P')
        blocks = local.reshape(local.shape[0] // 3, 3, -1, 4).transpose(0, 2, 1, 3)
        assert np.count_nonzero(np.abs(blocks).sum(axis=(2, 3))) <= 60

    def test_the_root_keeps_a_zero_transform(self, tmp_path):
        output = tmp_path / "hierarchy.npz"
        _compressor(joint_parents=_PARENTS).run(output)

        transforms = np.load(output)["shapeXform"]
        np.testing.assert_array_equal(transforms[:, -4:], 0.0)

    def test_the_fit_is_reported(self, tmp_path):
        compressor = _compressor(joint_parents=_PARENTS)
        compressor.run(tmp_path / "hierarchy.npz")

        assert np.isfinite(compressor.reconstruction_error.mean_abs)


class TestDeltasAboutJoints:
    def test_without_parent_links_the_default_matches_the_flat_solve(self, tmp_path):
        flat, linked = tmp_path / "flat.npz", tmp_path / "linked.npz"
        _compressor().run(flat)
        _compressor(joint_parents=[-1] * 6).run(linked)

        np.testing.assert_allclose(
            np.load(linked)["shapeXform"], np.load(flat)["shapeXform"], atol=1e-6
        )

    def test_deltas_about_joints_needs_parents(self):
        with pytest.raises(ValueError, match="joint_parents"):
            _compressor(deltas_about_joints=True)

    def test_deltas_about_joints_still_adds_up(self, tmp_path):
        output = tmp_path / "about.npz"
        _compressor(joint_parents=_PARENTS, deltas_about_joints=True).run(output)

        result = np.load(output)
        accumulated = accumulate_joint_transforms(
            result["localShapeXform"], result["jointParents"]
        )
        np.testing.assert_allclose(accumulated, result["shapeXform"], atol=1e-6)


class TestCliJointParents:
    def _model(self, tmp_path):
        return write_model_npz(
            tmp_path / "grid.npz",
            make_grid_model(),
            rest_joint_matrices=_joint_matrices(),
            joint_parents=np.array(_PARENTS),
        )

    def test_the_hierarchy_is_off_unless_asked_for(self, tmp_path):
        output = tmp_path / "out.npz"

        main([str(self._model(tmp_path)), str(output), *_FAST, "--max-influences", "2"])

        assert "jointParents" not in np.load(output).files

    def test_parents_stored_in_the_model_file_are_used_when_asked(self, tmp_path):
        output = tmp_path / "out.npz"

        main(
            [
                str(self._model(tmp_path)),
                str(output),
                *_FAST,
                "--max-influences",
                "2",
                "--joint-hierarchy",
            ]
        )

        assert "jointParents" in np.load(output).files

    def test_ignoring_joint_matrices_also_ignores_the_hierarchy(self, tmp_path):
        output = tmp_path / "out.npz"

        main(
            [
                str(self._model(tmp_path)),
                str(output),
                *_FAST,
                "--max-influences",
                "2",
                "--joint-hierarchy",
                "--number-of-bones",
                "10",
                "--ignore-joint-matrices",
            ]
        )

        assert "jointParents" not in np.load(output).files


class TestPipelineCommand:
    def test_the_hierarchy_is_off_by_default(self, tmp_path):
        command = compression_command(
            "python", tmp_path / "m.npz", tmp_path / "c.npz", CompressionSettings()
        )

        assert "--joint-hierarchy" not in command

    def test_the_hierarchy_can_be_switched_on(self, tmp_path):
        command = compression_command(
            "python",
            tmp_path / "m.npz",
            tmp_path / "c.npz",
            CompressionSettings(use_joint_hierarchy=True),
        )

        assert "--joint-hierarchy" in command


class TestNearestListedAncestors:
    def test_parents_are_the_nearest_exported_ancestor(self):
        dag = {
            "jaw": "head",
            "lip": "jaw_offset",
            "jaw_offset": "jaw",
            "head": None,
            "brow": "head",
        }

        parents = nearest_listed_ancestors(["jaw", "lip", "brow"], dag.get)

        assert parents == [-1, 0, -1]
