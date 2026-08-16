"""Download the pinned CRITEO-UPLIFTv2.1 source and verify its identity."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from retention_uplift.criteo import (
    DATASET_BYTES,
    DATASET_FILENAME,
    DATASET_LICENSE,
    DATASET_LICENSE_URL,
    DATASET_SHA256,
    DATASET_URL,
    file_sha256,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify the official Criteo uplift benchmark."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external") / DATASET_FILENAME,
    )
    parser.add_argument(
        "--accept-license",
        required=True,
        help=f"Must be exactly {DATASET_LICENSE}; see {DATASET_LICENSE_URL}",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.accept_license != DATASET_LICENSE:
        raise SystemExit(
            f"Download stopped: pass --accept-license {DATASET_LICENSE} only after "
            f"reviewing {DATASET_LICENSE_URL}"
        )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        digest = file_sha256(output)
        if output.stat().st_size == DATASET_BYTES and digest == DATASET_SHA256:
            print(f"Already verified: {output} ({digest})")
            return
        raise SystemExit(
            "Download stopped: output already exists but does not match the official "
            "size and SHA-256; move it aside and retry."
        )

    temporary = output.with_suffix(output.suffix + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "retention-treatment-uplift/0.1 data-verifier"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open(
            "wb"
        ) as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
                digest.update(block)
        actual_bytes = temporary.stat().st_size
        actual_sha256 = digest.hexdigest()
        if actual_bytes != DATASET_BYTES or actual_sha256 != DATASET_SHA256:
            raise ValueError(
                "download identity mismatch: "
                f"bytes={actual_bytes}, sha256={actual_sha256}"
            )
        temporary.replace(output)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Download failed safely: {error}") from error
    print(f"Downloaded and verified: {output} ({DATASET_SHA256})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Download interrupted")
