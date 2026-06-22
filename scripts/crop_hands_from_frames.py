#!/usr/bin/env python3
"""Crop left/right hands from sign frame images using MediaPipe Hands.

Reads sign_frames/metadata.json and the matching frame images, detects hands,
and writes cropped hand images under hand_crops/right/ and hand_crops/left/.
It also writes hand_crops/metadata.json with one record per detected hand, or
one record noting the miss when no hand is detected.

Usage:
    python crop_hands_from_frames.py --frames-dir kazakh_sign_frames \
            --metadata kazakh_sign_frames/metadata.json \
            --output-dir kazakh_hand_crops
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required but not installed.\n"
            "Install it with: pip install opencv-python"
        ) from exc
    return cv2


def require_mediapipe():
    try:
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks import python  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore
    except Exception as exc:
        raise SystemExit(
            f"Failed to import MediaPipe correctly:\n{exc}\n\n"
            "Install it with: python3 -m pip install mediapipe"
        ) from exc

    return mp, python, vision


def resolve_hand_model_path() -> Path:
    """Resolve the MediaPipe hand landmarker model path.

    The installed MediaPipe package in this environment does not ship a default
    model asset, so the path must be provided explicitly or placed in a common
    local location.
    """
    candidates = [
        os.environ.get("HAND_LANDMARKER_MODEL"),
        os.environ.get("HAND_LANDMARKER_MODEL_PATH"),
        "hand_landmarker.task",
        "models/hand_landmarker.task",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        "Could not find a MediaPipe hand landmarker model file.\n"
        "Set HAND_LANDMARKER_MODEL to a local .task file path, for example:\n"
        "  export HAND_LANDMARKER_MODEL=/path/to/hand_landmarker.task\n"
        "This repository does not currently include a hand_landmarker.task "
        "asset."
    )


def load_frame_metadata(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing frame metadata file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(
            f"Frame metadata file must contain a JSON list: {path}"
        )
    return data


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def landmarks_to_bbox(
    landmarks,
    width: int,
    height: int,
    padding: float = 0.15,
) -> tuple[int, int, int, int]:
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_pad = max(1.0, (x_max - x_min) * padding)
    y_pad = max(1.0, (y_max - y_min) * padding)

    x1 = clamp(int(x_min - x_pad), 0, width)
    y1 = clamp(int(y_min - y_pad), 0, height)
    x2 = clamp(int(x_max + x_pad), 0, width)
    y2 = clamp(int(y_max + y_pad), 0, height)
    return x1, y1, x2, y2


def normalize_hand_side(label: str) -> str:
    side = label.strip().lower()
    if side in {"left", "right"}:
        return side
    return side


def crop_hand(frame, bbox: tuple[int, int, int, int]):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description='Crop left/right hands from extracted sign frames'
    )
    parser.add_argument(
        '--frames-dir',
        default='sign_frames',
        help='directory containing extracted sign frames '
        '(default: sign_frames)',
    )
    parser.add_argument(
        '--metadata',
        default='sign_frames/metadata.json',
        help='metadata.json from frame extraction '
        '(default: sign_frames/metadata.json)',
    )
    parser.add_argument(
        '--output-dir',
        default='hand_crops',
        help='directory to save hand crops and metadata.json '
        '(default: hand_crops)',
    )
    args = parser.parse_args(argv)

    cv2 = require_cv2()
    mp, mp_python, vision = require_mediapipe()

    frames_dir = Path(args.frames_dir)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    right_dir = output_dir / "right"
    left_dir = output_dir / "left"
    output_metadata_path = output_dir / "metadata.json"

    if not frames_dir.exists():
        parser.error(f"Frames directory not found: {frames_dir}")
    if not frames_dir.is_dir():
        parser.error(f"Frames path is not a directory: {frames_dir}")

    if not metadata_path.exists():
        parser.error(f"Metadata file not found: {metadata_path}")

    frame_records = load_frame_metadata(metadata_path)

    model_path = resolve_hand_model_path()

    output_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    left_dir.mkdir(parents=True, exist_ok=True)

    metadata: list[dict] = []

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
    )

    with vision.HandLandmarker.create_from_options(options) as hands:
        for record in frame_records:
            frame_filename = record.get("frame_filename")
            sign_index = int(record.get("sign_index"))
            start_ms = record.get("start_ms")
            end_ms = record.get("end_ms")

            if not frame_filename:
                raise ValueError(f"Missing frame_filename in record: {record}")

            frame_path = frames_dir / frame_filename
            if not frame_path.exists():
                raise FileNotFoundError(f"Missing frame image: {frame_path}")

            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                raise RuntimeError(f"Could not read image: {frame_path}")

            height, width = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb,
            )
            result = hands.detect(mp_frame)

            if not result.hand_landmarks or not result.handedness:
                metadata.append(
                    {
                        "original_frame_filename": frame_filename,
                        "sign_index": sign_index,
                        "hand_side": None,
                        "crop_filename": None,
                        "bbox": None,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "detected": False,
                        "note": "No hand detected",
                    }
                )
                continue

            best_by_side: dict[str, tuple[object, object]] = {}
            for landmarks, handedness in zip(
                result.hand_landmarks,
                result.handedness,
            ):
                if not handedness:
                    continue
                category = handedness[0]
                side = normalize_hand_side(category.category_name or "")
                if side not in {"left", "right"}:
                    continue
                # Keep the first detection for each side to avoid duplicate
                # crops.
                if side not in best_by_side:
                    best_by_side[side] = (landmarks, handedness)

            if not best_by_side:
                metadata.append(
                    {
                        "original_frame_filename": frame_filename,
                        "sign_index": sign_index,
                        "hand_side": None,
                        "crop_filename": None,
                        "bbox": None,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "detected": False,
                        "note": (
                            "Hands were detected but no left/right label was "
                            "usable"
                        ),
                    }
                )
                continue

            for side in ("right", "left"):
                if side not in best_by_side:
                    continue
                landmarks, _handedness = best_by_side[side]
                bbox = landmarks_to_bbox(landmarks, width=width, height=height)
                cropped = crop_hand(frame_bgr, bbox)
                if cropped is None or cropped.size == 0:
                    metadata.append(
                        {
                            "original_frame_filename": frame_filename,
                            "sign_index": sign_index,
                            "hand_side": side,
                            "crop_filename": None,
                            "bbox": {
                                "x1": bbox[0],
                                "y1": bbox[1],
                                "x2": bbox[2],
                                "y2": bbox[3],
                            },
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "detected": False,
                            "note": "Empty crop after bounding-box extraction",
                        }
                    )
                    continue

                crop_filename = f"{side}_sign_{sign_index:04d}.jpg"
                crop_path = (
                    right_dir if side == "right" else left_dir
                ) / crop_filename
                if not cv2.imwrite(str(crop_path), cropped):
                    raise RuntimeError(
                        f"Could not write crop image: {crop_path}"
                    )

                metadata.append(
                    {
                        "original_frame_filename": frame_filename,
                        "sign_index": sign_index,
                        "hand_side": side,
                        "crop_filename": crop_filename,
                        "bbox": {
                            "x1": bbox[0],
                            "y1": bbox[1],
                            "x2": bbox[2],
                            "y2": bbox[3],
                        },
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "detected": True,
                    }
                )

    with output_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    detected_count = sum(1 for item in metadata if item.get("detected"))
    missed_count = sum(1 for item in metadata if not item.get("detected"))
    print(f"Processed {len(frame_records)} sign frames")
    print(f"Saved {detected_count} hand crops")
    print(f"Recorded {missed_count} no-detection entries")
    print(f"Saved metadata to {output_metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
