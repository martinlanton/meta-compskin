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

    def test_start_influences_caps_the_first_stage(self):
        schedule = build_training_schedule((10, 20), 8, 6000, start_influences=16)

        assert self._phases(schedule) == [
            (10, 16, 6000, False),
            (10, 16, 6000, True),
            (20, 8, 6000, True),
        ]

    def test_start_influences_interpolates_geometrically_over_three_stages(self):
        schedule = build_training_schedule((10, 20, 30), 8, 6000, start_influences=16)

        assert [p.max_influences for p in schedule] == [16, 16, 11, 8]

    def test_start_influences_is_ignored_by_a_single_stage(self):
        schedule = build_training_schedule((10,), 8, 6000, start_influences=16)

        assert [p.max_influences for p in schedule] == [8, 8]

    def test_default_ceiling_matches_the_doubling_rule(self):
        implicit = build_training_schedule((10, 20, 30), 8, 6000)
        explicit = build_training_schedule((10, 20, 30), 8, 6000, start_influences=32)

        assert implicit == explicit

    def test_rejects_a_start_below_the_final_influences(self):
        with pytest.raises(ValueError, match="start_influences"):
            build_training_schedule((10, 20), 8, 6000, start_influences=4)

    def test_rejects_no_stages(self):
        with pytest.raises(ValueError, match="at least one stage"):
            build_training_schedule((), 8, 6000)

    def test_rejects_a_stage_without_iterations(self):
        with pytest.raises(ValueError, match="iterations"):
            build_training_schedule((100, 0), 8, 6000)
