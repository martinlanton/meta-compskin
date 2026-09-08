"""Tests for the pure helpers in scripts/compare_schedules.py."""

import numpy as np
import pytest
from compare_schedules import (
    RunResult,
    build_parser,
    build_variants,
    format_summary,
    iterations_per_phase,
    summarize,
    write_csv,
)


def test_iterations_per_phase_splits_the_total():
    assert iterations_per_phase(40000, 2) == 20000
    assert iterations_per_phase(40000, 4) == 10000


def test_iterations_per_phase_rejects_totals_below_one_per_phase():
    with pytest.raises(ValueError, match="total_iterations"):
        iterations_per_phase(3, 4)


def test_build_variants_gives_every_variant_the_same_total_steps():
    variants = {v.name: v for v in build_variants(40000, 8, 50000, 151200, 3)}

    assert set(variants) == {"baseline", "baseline_control", "anneal_k", "anneal_kl"}
    assert all(
        sum(p.iterations for p in v.schedule) == 40000 for v in variants.values()
    )
    assert variants["baseline"].phases == 2
    assert variants["anneal_k"].phases == 4


def test_build_variants_stages():
    variants = {v.name: v for v in build_variants(40000, 8, 50000, 151200, 3)}

    assert variants["anneal_k"].influence_stages == "32,16,8"
    assert variants["anneal_k"].nnz_stages == "50000,50000,50000"
    assert variants["anneal_kl"].nnz_stages == "151200,100000,50000"  # 4L capped at 6SP
    assert variants["baseline_control"].influence_stages == "8,8,8"


def _result(variant, seed, max_abs, mean_abs):
    return RunResult(
        variant, seed, 2, 10, 20, "8", "100", max_abs, mean_abs, 1.0, "cpu", "x"
    )


def test_summarize_reports_mean_std_min_max_per_variant():
    results = [
        _result("a", 1, 1.0, 0.1),
        _result("a", 2, 3.0, 0.3),
        _result("b", 1, 5.0, 0.5),
    ]

    summary = {s.variant: s for s in summarize(results)}

    assert summary["a"].runs == 2
    assert summary["a"].max_abs_mean == pytest.approx(2.0)
    assert summary["a"].max_abs_std == pytest.approx(np.std([1.0, 3.0], ddof=1))
    assert (summary["a"].max_abs_min, summary["a"].max_abs_max) == (1.0, 3.0)
    assert summary["b"].max_abs_std == 0.0


def test_format_summary_has_one_line_per_variant():
    lines = format_summary(
        summarize([_result("a", 1, 1.0, 0.1), _result("b", 1, 2.0, 0.2)])
    ).splitlines()

    assert any(line.startswith("a") for line in lines)
    assert any(line.startswith("b") for line in lines)


def test_write_csv_round_trips(tmp_path):
    path = tmp_path / "results.csv"

    write_csv([_result("a", 1, 1.0, 0.1)], path)

    text = path.read_text()
    assert text.splitlines()[0].startswith("variant,seed,phases")
    assert "a,1,2,10,20,8,100,1.0,0.1" in text


def test_parser_defaults():
    args = build_parser().parse_args(["m.npz", "out"])

    assert args.seeds == (1, 2, 3, 4, 5)
    assert args.total_iterations == 40000
    assert args.anneal_stages == 3
    assert args.variants is None
    assert args.dry_run is False
