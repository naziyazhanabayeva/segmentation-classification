# Sign Language Pipeline (mp4 → pose → segments → crops → predictions)

## Setup (one-time)

Two environments are needed because MediaPipe and TensorFlow/Keras conflict
on protobuf versions in this stack:

```bash
# Env A: pose extraction + hand cropping (MediaPipe)
python3 -m venv venv-mediapipe
source venv-mediapipe/bin/activate
pip install mediapipe opencv-python pose_format numpy
deactivate

# Env B: segmentation + classification (TensorFlow/Keras + the segmentation package)
python3 -m venv venv-main
source venv-main/bin/activate
pip install -e .                     # installs sign_language_segmentation + pose_to_segments CLI
pip install tensorflow opencv-python simple-video-utils
deactivate
```

## Run (per video)

Replace `input.mp4` with your video. Run from this folder.

```bash
VIDEO=input.mp4
NAME=$(basename "$VIDEO" .mp4)

# 1. Video -> pose   (venv-mediapipe)
venv-mediapipe/bin/python scripts/generate_pose_from_video.py \
    --video "$VIDEO" --output "$NAME.pose"

# 2. Pose -> segments (.eaf)   (venv-main)
source venv-main/bin/activate
pose_to_segments --pose "$NAME.pose" --elan "$NAME.eaf" --video "$VIDEO"

# 3. .eaf -> segments.json
python scripts/parse_eaf.py --input "$NAME.eaf" --output "${NAME}_segments.json"

# 4. Clean/merge segments
python scripts/clean_segments.py --input "${NAME}_segments.json" \
    --output "${NAME}_cleaned_segments.json"
deactivate

# 5. Segments -> sign frames (middle frame per sign)   (venv-main, plain OpenCV, no mediapipe needed)
source venv-main/bin/activate
python scripts/extract_sign_frames.py --video "$VIDEO" \
    --segments "${NAME}_cleaned_segments.json" --output-dir "${NAME}_sign_frames/"
deactivate

# 6. Sign frames -> hand crops   (venv-mediapipe)
venv-mediapipe/bin/python scripts/crop_hands_from_frames.py \
    --frames-dir "${NAME}_sign_frames/" \
    --metadata "${NAME}_sign_frames/metadata.json" \
    --output-dir "${NAME}_hand_crops/"

# 7. Hand crops -> predictions (8 keras models)   (venv-main)
source venv-main/bin/activate
python scripts/classify_hand_crops_keras8.py \
    --crops-dir "${NAME}_hand_crops" \
    --models-dir classification-models \
    --output "${NAME}_hand_crops/predictions_keras8.json"
deactivate
```

Final predictions land in `${NAME}_hand_crops/predictions_keras8.json`.

## Folder contents

- `scripts/` — the 6 pipeline scripts (pose extraction, parsing/cleaning
  segments, frame/crop extraction, classification)
- `sign_language_segmentation/` — the segmentation model package, including
  the trained checkpoint at `sign_language_segmentation/dist/2026/`
  (provides the `pose_to_segments` CLI used in step 2)
- `classification-models/` — the 8 finetuned `.keras` checkpoints used in
  step 7 (matched by filename prefix: `broad_model*`, `sub_cat1_*` … `sub_cat7_*`).
  **Not tracked in git** (several files exceed GitHub's 100MB limit) — download
  them separately and place them in this folder before running step 7.
- `models/hand_landmarker.task` — MediaPipe hand landmark model used by
  `crop_hands_from_frames.py`. **Not tracked in git** — download it from
  [MediaPipe's model index](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
  and place it here.
- `pyproject.toml` — package metadata to `pip install -e .` the segmentation package

## Not included in this repo

`venv-main/` and `venv-mediapipe/` (regenerate via the Setup steps above),
the `.keras`/`.task` model weights above, and sample/per-run artifacts
(`*.mp4`, `*.pose`, `*.eaf`, `*_segments.json`, `*_sign_frames/`,
`*_hand_crops/`) are all gitignored. Supply your own input video and the
model weights to run the pipeline.
