"""Tests for TrainingPhase and build_training_schedule."""

import pytest

from metacompskin.model_fit import TrainingPhase, build_training_schedule


class TestTrainingPhase:
    def test_holds_its_values(self):
        phase = TrainingPhase(
            iterations=10, max_influences=8, total_nnz_B_rt=6000, normalize_weights=True
        )

        assert (phase.iterations, phase.max_influences, phase.total_nnz_B_rt) == (
            10,
            8,
            6000,
        )
        assert phase.normalize_weights is True

    @pytest.mark.parametrize(
        "field", ["iterations", "max_influences", "total_nnz_B_rt"]
    )
    def test_rejects_counts_below_one(self, field):
        values = {
            "iterations": 10,
            "max_influences": 8,
            "total_nnz_B_rt": 6000,
            "normalize_weights": False,
        }
        values[field] = 0

        with pytest.raises(ValueError, match=field):
            TrainingPhase(**values)


class TestBuildTrainingSchedule:
    def _phases(self, schedule):
        return [
            (p.iterations, p.max_influences, p.total_nnz_B_rt, p.normalize_weights)
            for p in schedule
        ]

    def test_one_stage_gives_the_classic_two_phases(self):
        schedule = build_training_schedule((100,), 8, 6000)

        assert self._phases(schedule) == [(100, 8, 6000, False), (100, 8, 6000, True)]

    def test_four_stages_halve_the_influences_after_a_warm_up(self):
        schedule = build_training_schedule((10, 20, 30, 40), 8, 6000)

        assert self._phases(schedule) == [
            (10, 64, 6000, False),
            (10, 64, 6000, True),
            (20, 32, 6000, True),
            (30, 16, 6000, True),
            (40, 8, 6000, True),
        ]

    def test_rejects_no_stages(self):
        with pytest.raises(ValueError, match="at least one stage"):
            build_training_schedule((), 8, 6000)

    def test_rejects_a_stage_without_iterations(self):
        with pytest.raises(ValueError, match="iterations"):
            build_training_schedule((100, 0), 8, 6000)
