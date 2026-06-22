#!/usr/bin/env python3
"""Hierarchical handshape classification using the 8 uploaded .keras models.

Broad model (7-way: categories "1".."7") picks a broad category, then the
matching sub_cat<N> model picks the exact subcategory within it.

These .keras checkpoints have Rescaling(1/255) + Normalization baked in as
the first layers, so images must be fed as raw float32 in [0, 255] -- do NOT
divide by 255 again here (unlike the older .hdf5-based classify scripts).

Class order per broad category is the alphabetically sorted list of
train_by_category subfolders for that category, with categories that had too
few images to train on excluded (confirmed with the model trainer):
  - category 1: 1_5 excluded (only 4 images)
  - category 7: 7_6_2 excluded (only 19 images)

Usage:
    python classify_hand_crops_keras8.py --crops-dir hand_crops \
        --models-dir ../classification-models
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BROAD_CATEGORIES = ["1", "2", "3", "4", "5", "6", "7"]

CLASS_GROUPS = {
    "1": ["1_1", "1_3"],
    "2": ["2_1", "2_2", "2_3", "2_5"],
    "3": ["3_1", "3_2", "3_3", "3_5"],
    "4": ["4_1", "4_2", "4_3", "4_5"],
    "5": ["5_1", "5_2", "5_3", "5_4", "5_5"],
    "6": ["6_1", "6_2", "6_3", "6_5"],
    "7": ["7_1_1", "7_1_2", "7_1_3", "7_1_4", "7_2_2", "7_4_1", "7_4_2", "7_4_3", "7_6_1", "7_6_3", "7_6_4"],
}

LABEL_MEANINGS = {
    "1_1": "Closed fist",
    "1_3": "Fist, slightly different thumb/angle",
    "1_5": "Fist, compact variant",
    "2_1": "Index finger extended straight up",
    "2_2": "One finger extended, bent forward",
    "2_3": "One finger extended, pointing down/forward",
    "3_1": "Two fingers extended together, pointing up",
    "3_2": "Fingers bent/curled",
    "3_3": "Finger(s) hooked/bent",
    "3_5": "Two fingers extended, together",
    "4_1": "Two fingers spread (peace-sign \"V\")",
    "4_2": "One finger bent/hooked downward",
    "4_3": "Fingers bent into a claw/hook shape",
    "4_5": "Fingers bent, hand near chest",
    "5_1": "Fingers together, slightly closed",
    "5_2": "Flat open palm",
    "5_3": "Rounded/cupped hand",
    "5_4": "Fingers loosely bent, relaxed",
    "5_5": "Fingers closed together, slightly bent",
    "6_1": "Open hand, all fingers fully spread",
    "6_2": "Curled, claw-like fingers",
    "6_3": "Relaxed, slightly open hand",
    "6_5": "Fingers spread, slightly bent",
    "7_1_1": "Index finger extended (pointing), thumb opposing, rest fisted",
    "7_1_2": "Thumb opposing near index finger, rest curled",
    "7_1_3": "Thumb-index pinch (like an \"OK\" pinch)",
    "7_1_4": "Fist with thumb crossing over",
    "7_2_2": "Thumb crossing between two extended fingers",
    "7_4_1": "Thumb touching multiple fingertips (pinch/circle shape)",
    "7_4_2": "Pinch gesture, thumb-to-fingertips",
    "7_4_3": "Cupped pinch, fingers together touching thumb",
    "7_6_1": "Relaxed open curved hand, thumb separate",
    "7_6_2": "Fingers spread loosely, casual",
    "7_6_3": "Fingers spread, hand mid-air",
    "7_6_4": "Open hand, fingers spread",
}

MODEL_FILENAME_GLOBS = {
    "broad": "broad_model*.keras",
    "1": "sub_cat1_*.keras",
    "2": "sub_cat2_*.keras",
    "3": "sub_cat3_*.keras",
    "4": "sub_cat4_*.keras",
    "5": "sub_cat5_*.keras",
    "6": "sub_cat6_*.keras",
    "7": "sub_cat7_*.keras",
}


def find_model(models_dir: Path, glob_pattern: str) -> Path:
    matches = sorted(models_dir.glob(glob_pattern))
    if not matches:
        raise FileNotFoundError(f"No model matching {glob_pattern!r} in {models_dir}")
    return matches[0]


def load_classifier(model_path: Path):
    import tensorflow as tf

    model = tf.keras.models.load_model(str(model_path), compile=False)
    height, width = model.input_shape[1:3]
    return model, (width, height)


def preprocess_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_CUBIC)
    return image.astype("float32")[None, ...]


class HierarchicalClassifier:
    def __init__(self, models_dir: Path):
        self.broad_model, self.broad_size = load_classifier(find_model(models_dir, MODEL_FILENAME_GLOBS["broad"]))
        self.sub_models: dict[str, tuple] = {}
        for category in BROAD_CATEGORIES:
            path = find_model(models_dir, MODEL_FILENAME_GLOBS[category])
            self.sub_models[category] = load_classifier(path)

    def classify(self, image_path: Path) -> dict:
        broad_batch = preprocess_image(image_path, self.broad_size)
        broad_probs = self.broad_model.predict(broad_batch, verbose=0)[0]
        broad_index = int(np.argmax(broad_probs))
        category = BROAD_CATEGORIES[broad_index]

        sub_model, sub_size = self.sub_models[category]
        sub_batch = preprocess_image(image_path, sub_size)
        sub_probs = sub_model.predict(sub_batch, verbose=0)[0]
        sub_index = int(np.argmax(sub_probs))
        class_names = CLASS_GROUPS[category]
        predicted_label = class_names[sub_index]

        return {
            "broad_category": category,
            "broad_confidence": float(broad_probs[broad_index]),
            "predicted_label": predicted_label,
            "predicted_meaning": LABEL_MEANINGS.get(predicted_label, "Unknown"),
            "sub_confidence": float(sub_probs[sub_index]),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hierarchical handshape classification using the 8 .keras models")
    parser.add_argument("--crops-dir", default="hand_crops", help="directory from crop_hands_from_frames.py (default: hand_crops)")
    parser.add_argument("--metadata", default=None, help="metadata.json inside crops-dir (default: <crops-dir>/metadata.json)")
    parser.add_argument("--models-dir", default="../classification-models", help="directory containing the 8 .keras checkpoints")
    parser.add_argument("--output", default=None, help="output predictions JSON (default: <crops-dir>/predictions_keras8.json)")
    args = parser.parse_args(argv)

    crops_dir = Path(args.crops_dir)
    metadata_path = Path(args.metadata) if args.metadata else crops_dir / "metadata.json"
    output_path = Path(args.output) if args.output else crops_dir / "predictions_keras8.json"

    with metadata_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    classifier = HierarchicalClassifier(Path(args.models_dir))

    results = []
    for record in records:
        if not record.get("detected"):
            results.append(record)
            continue

        crop_path = crops_dir / record["hand_side"] / record["crop_filename"]
        prediction = classifier.classify(crop_path)
        results.append({**record, **prediction})

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")

    classified = sum(1 for r in results if "predicted_label" in r)
    print(f"Classified {classified} hand crops")
    print(f"Saved predictions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
