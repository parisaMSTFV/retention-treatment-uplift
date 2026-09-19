"""Run and summarize the pre-declared synthetic seed-stability set."""

from __future__ import annotations

import argparse
from pathlib import Path

from retention_uplift.stability import DEFAULT_STABILITY_SEEDS, run_seed_stability


def _seeds(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated-seed synthetic policy validation.")
    parser.add_argument(
        "--seeds",
        type=_seeds,
        default=DEFAULT_STABILITY_SEEDS,
        help="Comma-separated seeds; defaults to 1,7,42,123,2026.",
    )
    parser.add_argument("--customers", type=int, default=18_000)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("local-runs/seed-stability"),
        help="Ignored directory for per-seed run artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/seed_stability.csv"),
        help="Aggregate CSV path.",
    )
    args = parser.parse_args()
    result = run_seed_stability(
        args.run_root.resolve(),
        args.output.resolve(),
        seeds=args.seeds,
        n_customers=args.customers,
    )
    print(
        "Completed seed stability: "
        f"runs={len(result)}, "
        f"positive_CI={int(result['ci_excludes_zero'].sum())}, "
        f"beats_risk_only={int(result['beats_risk_only'].sum())}"
    )


if __name__ == "__main__":
    main()
