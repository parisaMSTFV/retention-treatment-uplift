"""Contract, provenance, and deterministic sampling for CRITEO-UPLIFTv2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_NAME = "CRITEO-UPLIFTv2.1"
DATASET_FILENAME = "criteo-research-uplift-v2.1.csv.gz"
DATASET_PAGE = "https://ailab.criteo.com/criteo-uplift-prediction-dataset/"
DATASET_REPOSITORY = "https://huggingface.co/datasets/criteo/criteo-uplift"
DATASET_REVISION = "2424920019e49d52d72c13ac1143ec5d53af276b"
DATASET_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/"
    f"{DATASET_REVISION}/{DATASET_FILENAME}"
)
DATASET_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
DATASET_BYTES = 311_422_618
DATASET_ROWS = 13_979_592
DATASET_LICENSE = "CC-BY-NC-SA-4.0"
DATASET_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
DATASET_PAPER = "https://ama.liglab.fr/~amini/Publis/large-scale-benchmark.pdf"

FEATURE_COLUMNS = tuple(f"f{index}" for index in range(12))
BINARY_COLUMNS = ("treatment", "conversion", "visit", "exposure")
REQUIRED_COLUMNS = FEATURE_COLUMNS + BINARY_COLUMNS
SAMPLER_NAME = "splitmix64-bottom-k-v1"


@dataclass(frozen=True)
class CriteoLoadResult:
    """Validated sample and full-file audit statistics."""

    sample: pd.DataFrame
    sha256: str
    file_bytes: int
    source_rows: int
    full_counts: dict[str, dict[str, int]]


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_keys(row_numbers: np.ndarray, seed: int) -> np.ndarray:
    """Return deterministic uint64 sampling keys independent of CSV chunk size."""

    with np.errstate(over="ignore"):
        values = row_numbers.astype(np.uint64) + np.uint64(seed)
        values += np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        return values ^ (values >> np.uint64(31))


def partition_labels(source_rows: pd.Series, seed: int) -> pd.Series:
    """Create a deterministic 60/20/20 split from source row positions."""

    keys = _stable_keys(source_rows.to_numpy(dtype=np.uint64), seed + 1_000_003)
    fractions = keys.astype(np.float64) / np.float64(np.iinfo(np.uint64).max)
    return pd.Series(
        np.select(
            [fractions < 0.60, fractions < 0.80],
            ["train", "validation"],
            default="test",
        ),
        index=source_rows.index,
        dtype="string",
    )


def _validate_chunk(frame: pd.DataFrame, *, first_chunk: bool) -> pd.DataFrame:
    if first_chunk and set(frame.columns) != set(REQUIRED_COLUMNS):
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - set(REQUIRED_COLUMNS))
        raise ValueError(
            f"Criteo schema mismatch; missing={missing}, unexpected={unexpected}"
        )

    clean = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    feature_values = clean.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError("Criteo features must contain finite numeric values")

    for column in BINARY_COLUMNS:
        values = pd.to_numeric(clean[column], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"Criteo {column} must be binary 0/1")
        clean[column] = values.astype("int8")
    if clean["conversion"].gt(clean["visit"]).any():
        raise ValueError("Criteo conversion must imply a visit")
    if clean["exposure"].gt(clean["treatment"]).any():
        raise ValueError("Criteo exposure cannot occur in the randomized control arm")
    return clean


def scan_and_sample_criteo(
    path: str | Path,
    *,
    sample_size: int,
    seed: int,
    expected_sha256: str = DATASET_SHA256,
    expected_bytes: int | None = DATASET_BYTES,
    expected_rows: int | None = DATASET_ROWS,
    chunk_size: int = 250_000,
) -> CriteoLoadResult:
    """Verify the source, validate every row, and retain an exact deterministic sample."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Criteo input does not exist: {source}")
    if sample_size < 5_000:
        raise ValueError("sample_size must be at least 5,000")
    if chunk_size < 1_000:
        raise ValueError("chunk_size must be at least 1,000")

    actual_bytes = source.stat().st_size
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ValueError(
            f"Criteo file size mismatch: expected {expected_bytes}, found {actual_bytes}"
        )
    actual_sha256 = file_sha256(source)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Criteo SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )

    sample = pd.DataFrame()
    source_rows = 0
    full_counts = {
        outcome: {
            "control_rows": 0,
            "control_positive": 0,
            "treatment_rows": 0,
            "treatment_positive": 0,
        }
        for outcome in ("visit", "conversion")
    }
    for raw_chunk in pd.read_csv(source, chunksize=chunk_size):
        chunk = _validate_chunk(raw_chunk, first_chunk=source_rows == 0)
        row_numbers = np.arange(
            source_rows,
            source_rows + len(chunk),
            dtype=np.uint64,
        )
        for outcome in ("visit", "conversion"):
            for treatment, label in ((0, "control"), (1, "treatment")):
                mask = chunk["treatment"].eq(treatment)
                full_counts[outcome][f"{label}_rows"] += int(mask.sum())
                full_counts[outcome][f"{label}_positive"] += int(
                    chunk.loc[mask, outcome].sum()
                )

        chunk["_source_row"] = row_numbers
        chunk["_sample_key"] = _stable_keys(row_numbers, seed)
        pool = pd.concat([sample, chunk], ignore_index=True)
        if len(pool) > sample_size:
            keys = pool["_sample_key"].to_numpy(dtype=np.uint64)
            keep = np.argpartition(keys, sample_size - 1)[:sample_size]
            sample = pool.iloc[keep].copy()
        else:
            sample = pool
        source_rows += len(chunk)

    if expected_rows is not None and source_rows != expected_rows:
        raise ValueError(
            f"Criteo row-count mismatch: expected {expected_rows}, found {source_rows}"
        )
    if source_rows < sample_size:
        raise ValueError(
            f"sample_size={sample_size} exceeds the validated source rows={source_rows}"
        )

    sample = sample.sort_values("_sample_key").reset_index(drop=True)
    sample["partition"] = partition_labels(sample["_source_row"], seed)
    for partition in ("train", "validation", "test"):
        group = sample.loc[sample["partition"].eq(partition)]
        if len(group) < 500 or group["treatment"].nunique() != 2:
            raise ValueError(f"deterministic {partition} partition lacks randomized-arm support")
        if group["visit"].sum() < 20:
            raise ValueError(f"deterministic {partition} partition has too few visit outcomes")

    return CriteoLoadResult(
        sample=sample.drop(columns="_sample_key"),
        sha256=actual_sha256,
        file_bytes=actual_bytes,
        source_rows=source_rows,
        full_counts=full_counts,
    )


def binary_difference_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    """Difference in binary means and a large-sample 95% interval."""

    n_control = counts["control_rows"]
    n_treatment = counts["treatment_rows"]
    control_rate = counts["control_positive"] / n_control
    treatment_rate = counts["treatment_positive"] / n_treatment
    effect = treatment_rate - control_rate
    standard_error = float(
        np.sqrt(
            treatment_rate * (1.0 - treatment_rate) / n_treatment
            + control_rate * (1.0 - control_rate) / n_control
        )
    )
    return {
        **counts,
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "difference_in_means": effect,
        "standard_error": standard_error,
        "ci_low": effect - 1.96 * standard_error,
        "ci_high": effect + 1.96 * standard_error,
    }
