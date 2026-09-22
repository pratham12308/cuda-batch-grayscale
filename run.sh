#!/usr/bin/env bash
# Convenience runner for the batch grayscale project.
# Builds if needed, then processes data/input -> data/output.

set -e

INPUT_DIR="${INPUT_DIR:-data/input}"
OUTPUT_DIR="${OUTPUT_DIR:-data/output}"
THREADS="${THREADS:-16}"

if [[ ! -x ./batch_grayscale ]]; then
  echo "==> Building batch_grayscale..."
  make build
fi

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

echo "==> Running batch grayscale conversion"
echo "    input : $INPUT_DIR"
echo "    output: $OUTPUT_DIR"
echo "    block : ${THREADS}x${THREADS}"
echo

./batch_grayscale -i "$INPUT_DIR" -o "$OUTPUT_DIR" -t "$THREADS" | tee docs/execution.log

echo
echo "==> Done. Output images in: $OUTPUT_DIR"
echo "==> Log saved to: docs/execution.log"
