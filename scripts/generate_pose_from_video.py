#!/usr/bin/env python3
"""Convert an MP4 video into a pose-format `.pose` file using MediaPipe Holistic."""

from __future__ import annotations

import argparse
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a .pose file from an MP4 video")
    parser.add_argument("--video", required=True, type=Path, help="input video path")
    parser.add_argument("--output", required=True, type=Path, help="output pose file path")
    return parser


def generate_pose(video_path: Path, output_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    try:
        from simple_video_utils.frames import read_frames_exact
        from simple_video_utils.metadata import video_metadata
    except ImportError as exc:
        raise ImportError(
            f"simple_video_utils is required to read video files: {exc}"
        ) from exc

    try:
        from pose_format.utils.holistic import load_holistic
    except ImportError as exc:
        raise ImportError(
            f"pose-format and MediaPipe are required to generate `.pose` files: {exc}"
        ) from exc

    print(f"Video path: {video_path}")

    metadata = video_metadata(str(video_path))
    frames = list(read_frames_exact(str(video_path)))
    frame_count = len(frames)
    print(f"Number of frames read: {frame_count}")

    pose = load_holistic(
        frames,
        fps=metadata.fps,
        width=metadata.width,
        height=metadata.height,
        progress=True,
        additional_holistic_config={"model_complexity": 1, "refine_face_landmarks": True},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("wb") as file_handle:
            pose.write(file_handle)
    except OSError as exc:
        raise OSError(f"Could not write pose file: {output_path}") from exc

    print(f"Number of processed frames: {frame_count}")
    print(f"Output pose path: {output_path}")


def main() -> int:
    args = _build_parser().parse_args()
    generate_pose(args.video, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())