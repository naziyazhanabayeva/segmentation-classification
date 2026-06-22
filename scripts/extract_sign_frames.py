#!/usr/bin/env python3
"""Extract one frame per SIGN interval from example.mp4.

Reads cleaned_segments.json, extracts the middle frame for each SIGN interval,
and saves frames plus metadata under sign_frames/.

Usage:
    python extract_sign_frames.py --video kazakh_video.mp4 \
            --segments kazakh_cleaned_segments.json \
            --output-dir kazakh_sign_frames
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required for frame extraction but is not installed.\n"
            "Install it with: pip install opencv-python"
        ) from exc
    return cv2


def load_segments(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing segments file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(
            "Segments file must contain a JSON object at the top level: "
            f"{path}"
        )
    return data


def _segment_middle_ms(segment: dict) -> float:
    start_ms = float(segment["start_ms"])
    end_ms = float(segment["end_ms"])
    if end_ms < start_ms:
        raise ValueError(
            f"Invalid SIGN segment with end_ms < start_ms: {segment}"
        )
    return (start_ms + end_ms) / 2.0


def _format_filename(index: int, start_ms: float, end_ms: float) -> str:
    return f"sign_{index:04d}_{int(round(start_ms))}_{int(round(end_ms))}.jpg"


def extract_frame_at_ms(cv2, video_path: Path, middle_ms: float):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, middle_ms)
        success, frame = capture.read()
        if not success or frame is None:
            raise RuntimeError(
                f"Could not read frame at {middle_ms:.3f} ms from {video_path}"
            )
        return frame
    finally:
        capture.release()


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description='Extract one frame per SIGN interval'
    )
    parser.add_argument(
        '--video',
        default='example.mp4',
        help='input video path (default: example.mp4)',
    )
    parser.add_argument(
        '--segments',
        default='cleaned_segments.json',
        help='cleaned segments JSON file (default: cleaned_segments.json)',
    )
    parser.add_argument(
        '--output-dir',
        default='sign_frames',
        help='directory to save extracted frames and metadata.json '
        '(default: sign_frames)',
    )
    args = parser.parse_args(argv)

    cv2 = require_cv2()

    video_path = Path(args.video)
    segments_path = Path(args.segments)
    output_dir = Path(args.output_dir)
    metadata_path = output_dir / "metadata.json"

    if not video_path.exists():
        parser.error(f"Video file not found: {video_path}")

    data = load_segments(segments_path)
    sign_segments = data.get("SIGN", [])
    if not isinstance(sign_segments, list):
        raise ValueError(f"SIGN must be a list in {segments_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for index, segment in enumerate(sign_segments, start=1):
        start_ms = float(segment["start_ms"])
        end_ms = float(segment["end_ms"])
        middle_ms = _segment_middle_ms(segment)
        frame = extract_frame_at_ms(cv2, video_path, middle_ms)

        filename = _format_filename(index, start_ms, end_ms)
        output_path = output_dir / filename
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Could not write image: {output_path}")

        metadata.append(
            {
                "frame_filename": filename,
                "sign_index": index,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "middle_ms": middle_ms,
            }
        )

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(f"Saved {len(metadata)} sign frames to {output_dir}/")
    print(f"Saved metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
