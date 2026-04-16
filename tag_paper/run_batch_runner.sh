#!/usr/bin/env bash
set -euo pipefail

PORT=2081
export ALL_PROXY=socks5h://127.0.0.1:$PORT
export HTTP_PROXY=socks5h://127.0.0.1:$PORT
export HTTPS_PROXY=socks5h://127.0.0.1:$PORT


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  run_batch_runner.sh [safe|danger] [RANGE] [extra batch_runner args...]
  run_batch_runner.sh [safe|danger] [START] [END] [extra batch_runner args...]

Examples:
  bash run_batch_runner.sh safe 4-6
  bash run_batch_runner.sh danger 4 6
  bash run_batch_runner.sh safe 4-6 --paper-id paper_0004
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODE="safe"
if [[ "${1:-}" == "safe" || "${1:-}" == "danger" ]]; then
  MODE="$1"
  shift || true
fi

SANDBOX_MODE="workspace-write"
if [[ "$MODE" == "danger" ]]; then
  SANDBOX_MODE="danger-full-access"
fi

RUNNER_ARGS=()
if [[ $# -ge 1 && "$1" =~ ^([0-9]+)-([0-9]+)$ ]]; then
  RUNNER_ARGS+=(--start "${BASH_REMATCH[1]}" --end "${BASH_REMATCH[2]}")
  shift || true
elif [[ $# -ge 2 && "$1" =~ ^[0-9]+$ && "$2" =~ ^[0-9]+$ ]]; then
  RUNNER_ARGS+=(--start "$1" --end "$2")
  shift 2 || true
fi

RUNNER_ARGS+=("$@")

echo "Working directory: $SCRIPT_DIR"
echo "Runner mode: $MODE"
echo "Codex sandbox: $SANDBOX_MODE"
if [[ ${#RUNNER_ARGS[@]} -gt 0 ]]; then
  echo "Runner args: ${RUNNER_ARGS[*]}"
fi

python "$SCRIPT_DIR/batch_runner.py" \
  --manifest "$SCRIPT_DIR/data/manifest.jsonl" \
  --sandbox "$SANDBOX_MODE" \
  "${RUNNER_ARGS[@]}"
