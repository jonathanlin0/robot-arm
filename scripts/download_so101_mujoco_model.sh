#!/usr/bin/env bash

set -euo pipefail

MODEL_DIR="models/so101"

if [[ -e "${MODEL_DIR}" ]]; then
    echo "Error: ${MODEL_DIR} already exists." >&2
    exit 1
fi

mkdir -p models
DOWNLOAD_DIR="$(mktemp -d models/.so101-download.XXXXXX)"
trap 'rm -rf -- "${DOWNLOAD_DIR}"' EXIT

git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git \
    "${DOWNLOAD_DIR}/mujoco_menagerie"

git -C "${DOWNLOAD_DIR}/mujoco_menagerie" \
    sparse-checkout set robotstudio_so101

mv "${DOWNLOAD_DIR}/mujoco_menagerie/robotstudio_so101" "${MODEL_DIR}"

echo "Downloaded SO-101 model to ${MODEL_DIR}"
