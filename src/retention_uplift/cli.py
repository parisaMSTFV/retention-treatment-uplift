"""Command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from retention_uplift.config import ProjectConfig
from retention_uplift.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the synthetic retention uplift and policy pipeline."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--customers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ProjectConfig(seed=args.seed)
    if args.customers is not None:
        config = replace(config, n_customers=args.customers)
    metrics = run_pipeline(args.project_root.resolve(), config)
    print(
        "Completed retention uplift pipeline: "
        f"model={metrics['selected_model']}, "
        f"true_incremental_value={metrics['selected_true_incremental_value']:.0f}"
    )
