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
  total="${#frames[@]}"

  # randomization
  for ((i=total-1; i>0; i--)); do
      j=$((RANDOM % (i+1)))

      tmp="${frames[i]}"
      frames[i]="${frames[j]}"
      frames[j]="$tmp"
  done

  # Calculate train/eval split
  train_total=$((total * TRAIN_PERCENTAGE / 100))

  (( train_total < 1 )) && train_total=1
  (( train_total >= total )) && train_total=$((total-1))

  mkdir -p "train/$hero" "eval/$hero"

  train_count=0
  eval_count=0

  for frame in "${frames[@]}"; do
    if (( RANDOM % 100 < TRAIN_PERCENTAGE )); then
        split="train"
        ((train_count += 1))
    else
        split="eval"
        ((eval_count += 1))
    fi

    mv "$frame" "$split/$hero/${name}_${frame##*/}"
  done

  echo "  Hero: $hero"
  echo "  Train: $train_count"
  echo "  Eval: $eval_count"
done

echo ""
echo "Dataset preparation complete!"
echo "Train images: $(find train -type f -name '*.png' | wc -l)"
echo "Eval images:  $(find eval -type f -name '*.png' | wc -l)"
