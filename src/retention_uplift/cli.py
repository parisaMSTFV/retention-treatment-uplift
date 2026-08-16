"""Command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from retention_uplift.config import ProjectConfig
from retention_uplift.external_validation import run_criteo_validation
from retention_uplift.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the synthetic retention pipeline or external Criteo validation."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("local-runs/latest"),
        help="Output root; defaults to an ignored local-run directory.",
    )
    parser.add_argument("--customers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--external-criteo",
        type=Path,
        help="Run the checksum-verified CRITEO-UPLIFTv2.1 validation from this file.",
    )
    parser.add_argument("--external-sample-size", type=int, default=300_000)
    parser.add_argument("--external-target-share", type=float, default=0.30)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.external_criteo is not None:
        if args.customers is not None:
            parser.error("--customers applies only to the synthetic pipeline")
        try:
            metrics = run_criteo_validation(
                args.project_root.resolve(),
                args.external_criteo.resolve(),
                sample_size=args.external_sample_size,
                target_share=args.external_target_share,
                seed=args.seed,
            )
        except (FileNotFoundError, ValueError) as error:
            parser.exit(2, f"External validation stopped: {error}\n")
        policy = metrics["test_policy"]
        print(
            "Completed external randomized-benchmark validation: "
            f"model={metrics['selected_model']}, "
            f"test_reach={policy['targeted_share']:.1%}, "
            "AIPW visit effect among targeted="
            f"{policy['aipw_effect_among_targeted']:.3%}"
        )
        return

    config = ProjectConfig(seed=args.seed)
    if args.customers is not None:
        config = replace(config, n_customers=args.customers)
    metrics = run_pipeline(args.project_root.resolve(), config)
    print(
        "Completed retention uplift pipeline: "
        f"model={metrics['selected_model']}, "
        f"true_incremental_value={metrics['selected_true_incremental_value']:.0f}"
    )
