#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python executable not found. Install python3 or ensure python is on PATH." >&2
  exit 1
fi

if [[ -n "${BATCH_RUNNER_PROXY_PORT:-}" ]]; then
  PROXY_URL="socks5h://127.0.0.1:${BATCH_RUNNER_PROXY_PORT}"
  export ALL_PROXY="$PROXY_URL"
  export HTTP_PROXY="$PROXY_URL"
  export HTTPS_PROXY="$PROXY_URL"
fi

usage() {
  cat <<'EOF'
Usage:
  run_batch_runner.sh [safe|danger] [RANGE] [extra batch_runner args...]
  run_batch_runner.sh [safe|danger] [START] [END] [extra batch_runner args...]

Examples:
  bash run_batch_runner.sh safe 4-6
  bash run_batch_runner.sh danger 4 6
  bash run_batch_runner.sh safe 4-6 --paper-id paper_0004

Environment:
  BATCH_RUNNER_PROXY_PORT=2081  Optional local SOCKS5 proxy port.
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
echo "Python: $PYTHON_BIN"
echo "Runner mode: $MODE"
echo "Codex sandbox: $SANDBOX_MODE"
if [[ -n "${BATCH_RUNNER_PROXY_PORT:-}" ]]; then
  echo "Proxy: socks5h://127.0.0.1:${BATCH_RUNNER_PROXY_PORT}"
else
  echo "Proxy: disabled"
fi
if [[ ${#RUNNER_ARGS[@]} -gt 0 ]]; then
  echo "Runner args: ${RUNNER_ARGS[*]}"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/batch_runner.py" \
  --manifest "$SCRIPT_DIR/data/manifest.jsonl" \
  --sandbox "$SANDBOX_MODE" \
  "${RUNNER_ARGS[@]}"
