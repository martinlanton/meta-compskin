"""Tests for the solver's public name and the deprecated SkinCompressor alias."""

import pytest

import metacompskin
from metacompskin import model_fit


def test_compressed_skinning_solver_is_exported_from_the_package():
    solver_class = metacompskin.CompressedSkinningSolver

    assert solver_class is model_fit.CompressedSkinningSolver


def test_skin_compressor_from_the_package_warns_and_is_the_same_class():
    with pytest.warns(DeprecationWarning, match="CompressedSkinningSolver"):
        old_class = metacompskin.SkinCompressor

    assert old_class is model_fit.CompressedSkinningSolver


def test_skin_compressor_from_the_module_warns_and_is_the_same_class():
    with pytest.warns(DeprecationWarning, match="CompressedSkinningSolver"):
        old_class = model_fit.SkinCompressor

    assert old_class is model_fit.CompressedSkinningSolver


def test_skin_compressor_is_not_in_the_public_names():
    public_names = metacompskin.__all__

    assert "SkinCompressor" not in public_names
    assert "CompressedSkinningSolver" in public_names
