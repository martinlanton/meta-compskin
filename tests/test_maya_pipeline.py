"""Tests for the torch-free, Maya-free parts of the one-call Maya pipeline."""

import sys
import tempfile
from pathlib import Path

import pytest

import metacompskin.maya_pipeline as pipeline
from metacompskin.maya_pipeline import (
    CompressionSettings,
    TorchProbe,
    candidate_interpreters,
    compression_command,
    default_output_dir,
    detect_python_executable,
    probe_torch,
    resolve_python_executable,
)


class TestResolvePythonExecutable:
    def test_uses_the_argument_when_given(self, tmp_path, monkeypatch):
        monkeypatch.delenv("METACOMPSKIN_PYTHON", raising=False)
        python = tmp_path / "python"
        python.touch()

        assert resolve_python_executable(python) == python

    def test_falls_back_to_the_environment_variable(self, tmp_path, monkeypatch):
        python = tmp_path / "python"
        python.touch()
        monkeypatch.setenv("METACOMPSKIN_PYTHON", str(python))

        assert resolve_python_executable(None) == python

    def test_auto_detects_when_neither_is_set(self, tmp_path, monkeypatch):
        monkeypatch.delenv("METACOMPSKIN_PYTHON", raising=False)
        detected = tmp_path / "python"
        detected.touch()
        monkeypatch.setattr(pipeline, "detect_python_executable", lambda: detected)

        assert resolve_python_executable(None) == detected

    def test_fails_clearly_when_nothing_is_found(self, monkeypatch):
        monkeypatch.delenv("METACOMPSKIN_PYTHON", raising=False)
        monkeypatch.setattr(pipeline, "detect_python_executable", lambda: None)

        with pytest.raises(ValueError, match="METACOMPSKIN_PYTHON"):
            resolve_python_executable(None)

    def test_rejects_an_interpreter_that_does_not_exist(self, tmp_path, monkeypatch):
        monkeypatch.delenv("METACOMPSKIN_PYTHON", raising=False)

        with pytest.raises(FileNotFoundError):
            resolve_python_executable(tmp_path / "nope")


class TestDefaultOutputDir:
    def test_sits_beside_a_saved_scene(self):
        assert default_output_dir("/shots/s01/head.ma") == Path("/shots/s01/compskin")

    def test_uses_a_temp_directory_for_an_unsaved_scene(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setattr(tempfile, "tempdir", None)  # re-read TMPDIR

        output_dir = default_output_dir("")

        assert output_dir.parent == tmp_path
        assert output_dir.name.startswith("compskin")


class TestCompressionCommand:
    def test_runs_the_package_cli_with_every_setting(self):
        settings = CompressionSettings(
            iterations=500,
            number_of_bones=12,
            max_influences=4,
            total_nnz_B_rt=900,
            init_weight=1e-2,
            power=12,
            alpha=20.0,
            use_joint_matrices=False,
        )

        command = compression_command(
            Path("/env/bin/python"), Path("in.npz"), Path("out.npz"), settings
        )

        assert command[:5] == [
            "/env/bin/python",
            "-m",
            "metacompskin",
            "in.npz",
            "out.npz",
        ]
        assert command[5:] == [
            "--iterations",
            "500",
            "--number-of-bones",
            "12",
            "--max-influences",
            "4",
            "--total-nnz-b-rt",
            "900",
            "--init-weight",
            "0.01",
            "--power",
            "12",
            "--alpha",
            "20.0",
            "--ignore-joint-matrices",
        ]

    def test_defaults_only_pass_the_iteration_count(self):
        command = compression_command(
            Path("python"), Path("in.npz"), Path("out.npz"), CompressionSettings()
        )

        assert command[5:] == ["--iterations", "10000"]


class TestDetectPythonExecutable:
    def test_prefers_a_cuda_interpreter_over_an_earlier_cpu_one(self):
        probes = {
            Path("cpu"): TorchProbe(cuda=False),
            Path("cuda"): TorchProbe(cuda=True),
        }

        chosen = detect_python_executable(
            [Path("none"), Path("cpu"), Path("cuda")], probe=probes.get
        )

        assert chosen == Path("cuda")

    def test_falls_back_to_the_first_interpreter_with_torch(self):
        probes = {Path("a"): TorchProbe(cuda=False), Path("b"): TorchProbe(cuda=False)}

        chosen = detect_python_executable(
            [Path("none"), Path("a"), Path("b")], probe=probes.get
        )

        assert chosen == Path("a")

    def test_returns_none_when_no_candidate_has_torch(self):
        chosen = detect_python_executable([Path("a"), Path("b")], probe=lambda _: None)

        assert chosen is None

    def test_probe_recognises_this_interpreter_as_having_torch(self):
        probe = probe_torch(Path(sys.executable))

        assert probe is not None
        assert isinstance(probe.cuda, bool)

    def test_probe_reports_none_for_an_interpreter_without_torch(self, tmp_path):
        fake = tmp_path / "python"
        fake.write_text("#!/bin/sh\nexit 1\n")
        fake.chmod(0o755)

        assert probe_torch(fake) is None

    def test_candidates_list_existing_project_venvs_and_named_environments(
        self, tmp_path, monkeypatch
    ):
        home = tmp_path / "home"
        conda_env = home / "miniconda3" / "envs" / "torch" / "bin"
        conda_env.mkdir(parents=True)
        (conda_env / "python").touch()
        (home / "miniconda3" / "envs" / "empty").mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        monkeypatch.setattr(pipeline.shutil, "which", lambda _: None)
        package_root = tmp_path / "src"
        venv = package_root / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").touch()
        monkeypatch.setattr(pipeline, "_package_root", lambda: package_root)

        candidates = candidate_interpreters()

        assert candidates == [venv / "python", conda_env / "python"]

    def test_candidates_skip_maya_interpreters_and_duplicates(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nohome"))
        monkeypatch.setattr(pipeline, "_package_root", lambda: tmp_path / "nowhere")
        python = tmp_path / "bin" / "python"
        python.parent.mkdir()
        python.touch()
        mayapy = tmp_path / "bin" / "mayapy"
        mayapy.touch()
        on_path = {"python3": str(python), "python": str(python), "mayapy": str(mayapy)}
        monkeypatch.setattr(pipeline.shutil, "which", on_path.get)

        candidates = candidate_interpreters()

        assert candidates == [python]
