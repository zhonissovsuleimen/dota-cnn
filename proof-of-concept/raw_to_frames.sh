#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

cd "$(dirname "$0")"

TRAIN_PERCENTAGE=70
FPS=0
INPUT_DIR="raw"
IMAGE_SIZE=256

command -v ffmpeg >/dev/null || {
  echo "FFmpeg is not installed."
  exit 1
}

videos=()

for f in "$INPUT_DIR"/*.mp4; do
  filename="${f##*/}"

  [[ "$filename" == train_* || "$filename" == eval_* ]] && continue

  videos+=("$f")
done

if (( ${#videos[@]} == 0 )); then
  echo "No videos found."
  exit 1
fi

FILTER="scale=${IMAGE_SIZE}:${IMAGE_SIZE}"

if [[ "$FPS" != "0" ]]; then
  FILTER="fps=${FPS},${FILTER}"
fi


TEMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TEMP_DIR"' EXIT

total_new_train=0
total_new_eval=0

for video in "${videos[@]}"; do
  filename="${video##*/}"
  name="${filename%.mp4}"
  hero="${name%%_*}"

  echo "Processing $filename ..."

  # extracting frames
  ffmpeg -nostdin -y -loglevel error \
      -i "$video" \
      -vf "$FILTER" \
      -an \
      "$TEMP_DIR/frame_%06d.png"

  frames=("$TEMP_DIR"/frame_*.png)

  mkdir -p "train/$hero" "eval/$hero"

  train_files=("train/$hero"/*.png)
  eval_files=("eval/$hero"/*.png)

  train_index=${#train_files[@]}
  eval_index=${#eval_files[@]}

  train_new=0
  eval_new=0

  for frame in "${frames[@]}"; do
    if (( RANDOM % 100 < TRAIN_PERCENTAGE )); then
        split="train"
        index=$train_index
        ((train_index += 1))
        ((train_new += 1))
    else
        split="eval"
        index=$eval_index
        ((eval_index += 1))
        ((eval_new += 1))
    fi

    printf -v padded_index "%06d" "$index"
    mv "$frame" "$split/$hero/${name}_frame_${padded_index}.png"
  done

  echo "  Hero: $hero"
  echo "  Train: +$train_new (total: $train_index)"
  echo "  Eval:  +$eval_new (total: $eval_index)"
  echo ""
done

echo ""
echo "Total train images: $(find train -type f -name '*.png' | wc -l)"
echo "Total eval images:  $(find eval -type f -name '*.png' | wc -l)"
