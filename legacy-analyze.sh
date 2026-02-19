#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$(pwd)}"
OUTPUT="${2:-${ROOT}/analysis/legacy-context.json}"

if [[ $# -ge 1 ]]; then
  shift
fi
if [[ $# -ge 1 ]]; then
  shift
fi

python3 "${SCRIPT_DIR}/tools/legacy_analyzer.py" \
  --root "${ROOT}" \
  --output "${OUTPUT}" \
  --pretty \
  "$@"

echo "Analysis written to: ${OUTPUT}"
