import numpy as np
import pytest

from metacompskin.model_data import BlendshapeModelData


def _save_valid_npz(path, n_shapes=3, n_verts=10, n_faces=4, **overrides):
    arrays = {
        "deltas": np.zeros((n_shapes, n_verts, 3), dtype=np.float32),
        "rest_verts": np.zeros((n_verts, 3), dtype=np.float32),
        "rest_faces": np.array([[0, 1, 2, 3]] * n_faces, dtype=np.int32),
        "inbetween_info": np.array({}, dtype=object),
        "combination_info": np.array({}, dtype=object),
    }
    arrays.update(overrides)
    np.savez(str(path), **arrays)


class TestFromNpzFileHandling:
    def test_raises_file_not_found_for_missing_file(self, tmp_path):
        # Arrange
        missing = tmp_path / "nonexistent.npz"
        # Act / Assert
        with pytest.raises(FileNotFoundError):
            BlendshapeModelData.from_npz(str(missing))

    def test_raises_key_error_when_required_key_missing(self, tmp_path):
        # Arrange — save without 'deltas'
        npz_path = tmp_path / "model.npz"
        np.savez(
            str(npz_path),
            rest_verts=np.zeros((10, 3), dtype=np.float32),
            rest_faces=np.array([[0, 1, 2, 3]], dtype=np.int32),
            inbetween_info=np.array({}, dtype=object),
            combination_info=np.array({}, dtype=object),
        )
        # Act / Assert
        with pytest.raises(KeyError, match="deltas"):
            BlendshapeModelData.from_npz(str(npz_path))


class TestFromNpzValidation:
    def test_raises_value_error_for_vertex_count_mismatch(self, tmp_path):
        # Arrange — deltas has 10 verts, rest_verts has 8
        npz_path = tmp_path / "model.npz"
        _save_valid_npz(
            npz_path,
            deltas=np.zeros((3, 10, 3), dtype=np.float32),
            rest_verts=np.zeros((8, 3), dtype=np.float32),
        )
        # Act / Assert
        with pytest.raises(ValueError, match="vertex count"):
            BlendshapeModelData.from_npz(str(npz_path))

    def test_raises_value_error_for_unsupported_face_type(self, tmp_path):
        # Arrange — pentagons (5 verts per face) are not supported
        npz_path = tmp_path / "model.npz"
        _save_valid_npz(
            npz_path,
            rest_faces=np.array([[0, 1, 2, 3, 4]], dtype=np.int32),
        )
        # Act / Assert
        with pytest.raises(ValueError, match="triangles"):
            BlendshapeModelData.from_npz(str(npz_path))

    def test_raises_value_error_for_out_of_range_face_index(self, tmp_path):
        # Arrange — face references vertex 99, but mesh only has 10 verts
        npz_path = tmp_path / "model.npz"
        _save_valid_npz(
            npz_path,
            rest_faces=np.array([[0, 1, 2, 99]], dtype=np.int32),
        )
        # Act / Assert
        with pytest.raises(ValueError, match="invalid vertex index"):
            BlendshapeModelData.from_npz(str(npz_path))

    def test_raises_value_error_for_non_float_deltas(self, tmp_path):
        # Arrange — integer deltas should be rejected
        npz_path = tmp_path / "model.npz"
        _save_valid_npz(
            npz_path,
            deltas=np.zeros((3, 10, 3), dtype=np.int32),
        )
        # Act / Assert
        with pytest.raises(ValueError, match="floating point"):
            BlendshapeModelData.from_npz(str(npz_path))


class TestFromNpzAlpha:
    def test_alpha_uses_known_model_constant(self, tmp_path):
        # Arrange
        npz_path = tmp_path / "aura.npz"
        _save_valid_npz(npz_path)
        # Act
        model = BlendshapeModelData.from_npz(str(npz_path))
        # Assert
        assert model.alpha == 10.0

    def test_alpha_override_takes_precedence_over_constant(self, tmp_path):
        # Arrange
        npz_path = tmp_path / "aura.npz"
        _save_valid_npz(npz_path)
        # Act
        model = BlendshapeModelData.from_npz(str(npz_path), alpha=99.0)
        # Assert
        assert model.alpha == 99.0

    def test_alpha_falls_back_to_default_for_unknown_model(self, tmp_path):
        # Arrange
        npz_path = tmp_path / "unknown_model.npz"
        _save_valid_npz(npz_path)
        # Act
        model = BlendshapeModelData.from_npz(str(npz_path))
        # Assert
        assert model.alpha == 10.0


class TestBlendshapeModelDataProperties:
    def test_properties_reflect_loaded_array_shapes(self, tmp_path):
        # Arrange
        npz_path = tmp_path / "model.npz"
        _save_valid_npz(npz_path, n_shapes=5, n_verts=20, n_faces=8)
        # Act
        model = BlendshapeModelData.from_npz(str(npz_path))
        # Assert
        assert model.n_blendshapes == 5
        assert model.n_vertices == 20
        assert model.n_faces == 8
