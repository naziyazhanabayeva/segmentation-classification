#!/usr/bin/env python3
"""Clean SIGN segments in segments.json and write cleaned_segments.json.

Rules:
  - Keep SENTENCE segments unchanged.
  - For SIGN segments, drop intervals shorter than 100 ms.
    - Merge neighboring SIGN intervals when the gap between them is
        less than 80 ms.

Usage:
    python clean_segments.py --input kazakh_segments.json \
            --output kazakh_cleaned_segments.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

MIN_SIGN_DURATION_MS = 100
MERGE_GAP_MS = 80


def _validate_segment(segment: dict) -> tuple[int, int]:
    start_ms = int(segment["start_ms"])
    end_ms = int(segment["end_ms"])
    if end_ms < start_ms:
        raise ValueError(f"Invalid segment with end_ms < start_ms: {segment}")
    return start_ms, end_ms


def clean_sign_segments(
    sign_segments: list[dict],
    min_duration_ms: int,
    merge_gap_ms: int,
) -> list[dict]:
    """Remove short segments and merge close neighbors."""
    normalized: list[tuple[int, int]] = []
    for segment in sign_segments:
        start_ms, end_ms = _validate_segment(segment)
        if end_ms - start_ms >= min_duration_ms:
            normalized.append((start_ms, end_ms))

    if not normalized:
        return []

    normalized.sort(key=lambda item: (item[0], item[1]))
    merged: list[tuple[int, int]] = [normalized[0]]

    for start_ms, end_ms in normalized[1:]:
        prev_start_ms, prev_end_ms = merged[-1]
        gap_ms = start_ms - prev_end_ms
        if gap_ms < merge_gap_ms:
            merged[-1] = (prev_start_ms, max(prev_end_ms, end_ms))
        else:
            merged.append((start_ms, end_ms))

    return [
        {"start_ms": start_ms, "end_ms": end_ms}
        for start_ms, end_ms in merged
    ]


def load_segments(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(
            f"Input file must contain a JSON object at the top level: {path}"
        )
    return data


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description='Clean SIGN segments and merge nearby intervals'
    )
    parser.add_argument(
        '--input',
        default='segments.json',
        help='input segments JSON file (default: segments.json)',
    )
    parser.add_argument(
        '--output',
        default='cleaned_segments.json',
        help='output cleaned JSON file (default: cleaned_segments.json)',
    )
    parser.add_argument(
        '--min-duration-ms',
        type=int,
        default=MIN_SIGN_DURATION_MS,
        help='minimum SIGN duration to keep in milliseconds (default: 100)',
    )
    parser.add_argument(
        '--merge-gap-ms',
        type=int,
        default=MERGE_GAP_MS,
        help=(
            'merge neighboring SIGN segments when the gap is smaller than '
            'this many milliseconds (default: 80)'
        ),
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    data = load_segments(input_path)

    sentence_segments = data.get("SENTENCE", [])
    sign_segments = data.get("SIGN", [])

    if not isinstance(sentence_segments, list):
        raise ValueError("SENTENCE must be a list")
    if not isinstance(sign_segments, list):
        raise ValueError("SIGN must be a list")

    cleaned_sign_segments = clean_sign_segments(
        sign_segments,
        args.min_duration_ms,
        args.merge_gap_ms,
    )
    cleaned = {
        "SIGN": cleaned_sign_segments,
        "SENTENCE": sentence_segments,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(cleaned, handle, indent=2)
        handle.write("\n")

    print(f"Original SIGN segments: {len(sign_segments)}")
    print(f"Cleaned SIGN segments: {len(cleaned_sign_segments)}")
    print(f"SENTENCE segments: {len(sentence_segments)}")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
