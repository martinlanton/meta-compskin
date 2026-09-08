r"""Compares plain compression against annealed influence-budget schedules.

Runs four variants of `SkinCompressor` — the plain two-phase baseline, a
phase-count control at the same K/L throughout, and two annealed schedules
(weights only, and weights plus deltas) — over several seeds, all at the
same *total* step count, and reports mean/std/min/max of the final MXE and
MAE per variant. This answers the question "does annealing the influence
budget beat the baseline by more than seed-to-seed noise?" for a given
model; see plan/research.md Section 8 for how to read the result.

Usage::

    python scripts/compare_schedules.py MODEL.npz OUT_DIR \\
        --number-of-bones 200 --max-influences 8 --total-nnz-b-rt 50000 \\
        --alpha 7 --seeds 1,2,3,4,5 --total-iterations 40000 --anneal-stages 3

Fairness: every variant runs for exactly `--total-iterations` steps in
total, split evenly across its phases (see `iterations_per_phase`), so a
variant with more phases is not simply given more compute.

Outputs, under OUT_DIR:
    <variant>/seed_<seed>.npz: the compressed archive for each run.
    results.csv: one row per run (see RunResult).
    summary.txt: the printed mean/std/min/max table.
"""

import argparse
import csv
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch

from metacompskin.cli import _stored_joint_matrices
from metacompskin.model_data import BlendshapeModelData
from metacompskin.model_fit import (
    SkinCompressor,
    TrainingPhase,
    build_training_schedule,
)

_VARIANT_NAMES = ("baseline", "baseline_control", "anneal_k", "anneal_kl")


@dataclasses.dataclass(frozen=True)
class Variant:
    """One compression schedule to compare, sized for a fixed total step count.

    Attributes:
        name: One of "baseline", "baseline_control", "anneal_k", "anneal_kl".
        schedule: The TrainingPhase objects run() executes for this variant.
    """

    name: str
    schedule: tuple[TrainingPhase, ...]

    @property
    def phases(self) -> int:
        """Number of TrainingPhase objects in this variant's schedule."""
        return len(self.schedule)

    @property
    def influence_stages(self) -> str:
        """Comma-separated K per stage (the schedule's phases after warm-up)."""
        return ",".join(str(phase.max_influences) for phase in self.schedule[1:])

    @property
    def nnz_stages(self) -> str:
        """Comma-separated L per stage (the schedule's phases after warm-up)."""
        return ",".join(str(phase.total_nnz_B_rt) for phase in self.schedule[1:])


@dataclasses.dataclass(frozen=True)
class RunResult:
    """One (variant, seed) compression run's outcome.

    Attributes:
        variant: The Variant.name this run used.
        seed: The seed this run used.
        phases: Number of phases the schedule had.
        iterations_per_phase: Steps in each phase (uniform across a variant).
        total_steps: Sum of iterations over every phase.
        influence_stages: Comma-separated K per stage.
        nnz_stages: Comma-separated L per stage.
        max_abs: Final MXE (reconstruction_error.max_abs).
        mean_abs: Final MAE (reconstruction_error.mean_abs).
        seconds: Wall time of the compression run.
        device: "cuda" or "cpu".
        torch_version: torch.__version__ at run time.
    """

    variant: str
    seed: int
    phases: int
    iterations_per_phase: int
    total_steps: int
    influence_stages: str
    nnz_stages: str
    max_abs: float
    mean_abs: float
    seconds: float
    device: str
    torch_version: str


@dataclasses.dataclass(frozen=True)
class VariantSummary:
    """Aggregate MXE/MAE statistics for every run of one variant.

    Attributes:
        variant: The Variant.name summarised.
        runs: Number of RunResult entries this summary was built from.
        max_abs_mean: Mean of RunResult.max_abs across runs.
        max_abs_std: Sample standard deviation (ddof=1) of max_abs; 0 for a
            single run.
        max_abs_min: Minimum max_abs across runs.
        max_abs_max: Maximum max_abs across runs.
        mean_abs_mean: Mean of RunResult.mean_abs across runs.
        mean_abs_std: Sample standard deviation (ddof=1) of mean_abs; 0 for a
            single run.
    """

    variant: str
    runs: int
    max_abs_mean: float
    max_abs_std: float
    max_abs_min: float
    max_abs_max: float
    mean_abs_mean: float
    mean_abs_std: float


def iterations_per_phase(total_iterations: int, phases: int) -> int:
    """Splits a total step budget evenly across a variant's phases.

    Args:
        total_iterations: Total steps the variant should run.
        phases: Number of phases in the variant's schedule.

    Returns:
        total_iterations // phases.

    Raises:
        ValueError: If that would be less than one step per phase.
    """
    per_phase = total_iterations // phases
    if per_phase < 1:
        raise ValueError(
            f"total_iterations={total_iterations} split over {phases} phases "
            "gives less than 1 step per phase"
        )
    return per_phase


def build_variants(
    total_iterations: int,
    max_influences: int,
    total_nnz_B_rt: int,  # noqa: N803 (matches SkinCompressor)
    n_coefficients: int,
    anneal_stages: int,
) -> list[Variant]:
    """Builds the four comparison variants, each at the given total step count.

    Args:
        total_iterations: Total steps every variant runs, in total.
        max_influences: K reached by every variant (the shipped budget).
        total_nnz_B_rt: L reached by every variant (the shipped budget).
        n_coefficients: 6*S*P, the cap applied to annealed L stages.
        anneal_stages: N, the number of stages in the annealed variants.

    Returns:
        [baseline, baseline_control, anneal_k, anneal_kl].

    References:
        plan/spec.md Section 6.5 (the variant table).
    """
    baseline_iterations = iterations_per_phase(total_iterations, 2)
    baseline = Variant(
        "baseline",
        build_training_schedule((baseline_iterations,), max_influences, total_nnz_B_rt),
    )

    phases = anneal_stages + 1
    stage_iterations = iterations_per_phase(total_iterations, phases)

    anneal_k = Variant(
        "anneal_k",
        build_training_schedule(
            (stage_iterations,) * anneal_stages, max_influences, total_nnz_B_rt
        ),
    )

    baseline_control = Variant(
        "baseline_control",
        (
            TrainingPhase(
                stage_iterations,
                max_influences,
                total_nnz_B_rt,
                normalize_weights=False,
            ),
            *(
                TrainingPhase(
                    stage_iterations,
                    max_influences,
                    total_nnz_B_rt,
                    normalize_weights=True,
                )
                for _ in range(anneal_stages)
            ),
        ),
    )

    k_stages = [
        max_influences * 2 ** (anneal_stages - 1 - i) for i in range(anneal_stages)
    ]
    l_stages = [
        min(n_coefficients, total_nnz_B_rt * 2 ** (anneal_stages - 1 - i))
        for i in range(anneal_stages)
    ]
    anneal_kl = Variant(
        "anneal_kl",
        (
            TrainingPhase(
                stage_iterations, k_stages[0], l_stages[0], normalize_weights=False
            ),
            *(
                TrainingPhase(
                    stage_iterations, k_stages[i], l_stages[i], normalize_weights=True
                )
                for i in range(anneal_stages)
            ),
        ),
    )

    return [baseline, baseline_control, anneal_k, anneal_kl]


def summarize(results: list[RunResult]) -> list[VariantSummary]:
    """Aggregates MXE/MAE mean, std, min, max per variant.

    Args:
        results: Runs to summarise, any number of variants and seeds mixed.

    Returns:
        One VariantSummary per distinct RunResult.variant seen, in order of
        first appearance.
    """
    by_variant: dict[str, list[RunResult]] = {}
    for result in results:
        by_variant.setdefault(result.variant, []).append(result)

    summaries = []
    for variant, runs in by_variant.items():
        max_abs = [run.max_abs for run in runs]
        mean_abs = [run.mean_abs for run in runs]
        n = len(runs)
        summaries.append(
            VariantSummary(
                variant=variant,
                runs=n,
                max_abs_mean=sum(max_abs) / n,
                max_abs_std=float(np.std(max_abs, ddof=1)) if n > 1 else 0.0,
                max_abs_min=min(max_abs),
                max_abs_max=max(max_abs),
                mean_abs_mean=sum(mean_abs) / n,
                mean_abs_std=float(np.std(mean_abs, ddof=1)) if n > 1 else 0.0,
            )
        )
    return summaries


def format_summary(summaries: list[VariantSummary]) -> str:
    """Renders a VariantSummary list as an aligned text table.

    Args:
        summaries: Rows to render, in the order given.

    Returns:
        A multi-line string, one header line and one line per summary.
    """
    header = (
        f"{'variant':<20} {'n':>3} {'MXE mean':>10} {'MXE std':>10} "
        f"{'MXE min':>10} {'MXE max':>10} {'MAE mean':>10} {'MAE std':>10}"
    )
    lines = [header]
    for s in summaries:
        lines.append(
            f"{s.variant:<20} {s.runs:>3} {s.max_abs_mean:>10.4f} "
            f"{s.max_abs_std:>10.4f} {s.max_abs_min:>10.4f} {s.max_abs_max:>10.4f} "
            f"{s.mean_abs_mean:>10.4f} {s.mean_abs_std:>10.4f}"
        )
    return "\n".join(lines)


def write_csv(results: list[RunResult], path: Path) -> None:
    """Writes every result to a CSV file, one row per run.

    Args:
        results: Rows to write, in the order given.
        path: File to write; overwritten if it already exists.
    """
    fieldnames = [field.name for field in dataclasses.fields(RunResult)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(dataclasses.asdict(result))


def run_variant(
    model_data: BlendshapeModelData,
    variant: Variant,
    seed: int,
    settings: dict,
    output: Path,
) -> RunResult:
    """Compresses the model once with one variant's schedule and one seed.

    Args:
        model_data: The loaded model to compress.
        variant: Which schedule to run.
        seed: Random seed for this run.
        settings: Extra SkinCompressor keyword arguments (number_of_bones,
            max_influences, total_nnz_B_rt, init_weight, power, and
            optionally rest_joint_matrices). iterations is not included:
            variant.schedule replaces whatever schedule the constructor
            would have built.
        output: Where to write the compressed NPZ.

    Returns:
        The run's settings and final reconstruction error.
    """
    compressor = SkinCompressor(model_data=model_data, seed=seed, **settings)
    compressor.schedule = variant.schedule

    start = time.perf_counter()
    compressor.run(output)
    elapsed = time.perf_counter() - start

    error = compressor.reconstruction_error
    assert error is not None  # noqa: S101 (run() always sets it before returning)
    return RunResult(
        variant=variant.name,
        seed=seed,
        phases=variant.phases,
        iterations_per_phase=variant.schedule[0].iterations,
        total_steps=sum(phase.iterations for phase in variant.schedule),
        influence_stages=variant.influence_stages,
        nnz_stages=variant.nnz_stages,
        max_abs=error.max_abs,
        mean_abs=error.mean_abs,
        seconds=elapsed,
        device=compressor.device,
        torch_version=torch.__version__,
    )


def _int_list(text: str) -> tuple[int, ...]:
    """Parses a comma-separated list of integers for argparse.

    Args:
        text: Raw argument value, e.g. "1,2,3".

    Returns:
        The parsed integers.

    Raises:
        argparse.ArgumentTypeError: If any item is not an integer.
    """
    try:
        return tuple(int(item) for item in text.split(","))
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {text!r}"
        ) from e


def _str_list(text: str) -> tuple[str, ...]:
    """Parses a comma-separated list of names for argparse.

    Args:
        text: Raw argument value, e.g. "baseline,anneal_k".

    Returns:
        The parsed names.
    """
    return tuple(text.split(","))


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for this script.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="compare_schedules.py",
        description=(
            "Compare the plain compression baseline against annealed "
            "influence-budget schedules, across seeds."
        ),
    )
    parser.add_argument("model", help="Input model NPZ.")
    parser.add_argument(
        "out_dir", help="Directory for per-run NPZs, results.csv and summary.txt."
    )
    parser.add_argument("--seeds", type=_int_list, default=(1, 2, 3, 4, 5))
    parser.add_argument("--total-iterations", type=int, default=40000)
    parser.add_argument(
        "--anneal-stages",
        type=int,
        default=3,
        help="N: number of annealing stages in anneal_k/anneal_kl/baseline_control",
    )
    parser.add_argument("--number-of-bones", type=int, default=100, help="P")
    parser.add_argument("--max-influences", type=int, default=8, help="K")
    parser.add_argument("--total-nnz-b-rt", type=int, default=6000, help="L")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--power", type=int, default=2)
    parser.add_argument("--init-weight", type=float, default=1e-3)
    parser.add_argument(
        "--variants",
        type=_str_list,
        default=None,
        help=f"comma-separated subset of {','.join(_VARIANT_NAMES)}",
    )
    parser.add_argument(
        "--ignore-joint-matrices",
        action="store_true",
        help="Do not use rest_joint_matrices stored in the model file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run matrix and exit without compressing anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    """Runs every variant at every seed and writes results.csv and summary.txt.

    Args:
        argv: Arguments without the program name; None reads sys.argv.

    Returns:
        Path to the written results.csv (results.csv under out_dir even on
        --dry-run, though nothing is written to disk in that case).

    Raises:
        ValueError: If --variants names an unknown variant.
    """
    args = build_parser().parse_args(argv)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)

    model_data = BlendshapeModelData.from_npz(str(model_path), alpha=args.alpha)
    n_coefficients = 6 * model_data.n_blendshapes * args.number_of_bones
    variants = build_variants(
        args.total_iterations,
        args.max_influences,
        args.total_nnz_b_rt,
        n_coefficients,
        args.anneal_stages,
    )
    if args.variants is not None:
        wanted = set(args.variants)
        unknown = wanted - {variant.name for variant in variants}
        if unknown:
            raise ValueError(
                f"unknown variants: {sorted(unknown)}; known: {list(_VARIANT_NAMES)}"
            )
        variants = [variant for variant in variants if variant.name in wanted]

    if args.dry_run:
        for seed in args.seeds:
            for variant in variants:
                print(
                    f"{variant.name} seed={seed} phases={variant.phases} "
                    f"iterations_per_phase={variant.schedule[0].iterations} "
                    f"K={variant.influence_stages} L={variant.nnz_stages}"
                )
        return out_dir / "results.csv"

    settings: dict = {
        "number_of_bones": args.number_of_bones,
        "max_influences": args.max_influences,
        "total_nnz_B_rt": args.total_nnz_b_rt,
        "init_weight": args.init_weight,
        "power": args.power,
    }
    if not args.ignore_joint_matrices:
        settings["rest_joint_matrices"] = _stored_joint_matrices(model_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    results: list[RunResult] = []
    for seed in args.seeds:
        for variant in variants:
            variant_dir = out_dir / variant.name
            variant_dir.mkdir(parents=True, exist_ok=True)
            output = variant_dir / f"seed_{seed}.npz"
            result = run_variant(model_data, variant, seed, settings, output)
            results.append(result)
            write_csv(results, csv_path)  # rewritten after each run: crash-safe

    table = format_summary(summarize(results))
    print(table)
    (out_dir / "summary.txt").write_text(table + "\n", encoding="utf-8")
    return csv_path


if __name__ == "__main__":
    main()
