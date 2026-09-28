#!/usr/bin/env bash
# One-command setup and launch for Flutter Lab (macOS / Linux).
#   ./start.sh            set up (first run only) and open the web app
#   ./start.sh test       run the test suite
#   ./start.sh evaluate   regenerate results and figures
#   ./start.sh train      (re)train the neural policy (installs PyTorch)
set -euo pipefail
cd "$(dirname "$0")"
VENV="${FLUTTER_AI_VENV:-.venv}"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "Creating virtual environment at $VENV ..."
  python3 -m venv "$VENV"
  "$PY" -m pip install --upgrade pip --quiet
fi

EXTRAS="llm,dev"
if [ "${1:-}" = "train" ] || [ ! -f models/neural_policy_uav_flap.npz ]; then EXTRAS="llm,dev,train"; fi
STAMP="$VENV/.flutter-ai-stamp"
WANT="$(cksum pyproject.toml | cut -d' ' -f1)|$EXTRAS"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$WANT" ]; then
  echo "Installing dependencies ($EXTRAS) ..."
  "$PY" -m pip install -e ".[$EXTRAS]" --quiet
  echo "$WANT" > "$STAMP"
fi

case "${1:-serve}" in
  test)     exec "$PY" -m pytest ;;
  evaluate) exec "$PY" -m flutter_ai evaluate ;;
  train)    exec "$PY" -m flutter_ai train ;;
  *)
    [ -f models/neural_policy_uav_flap.npz ] || "$PY" -m flutter_ai train
    exec "$PY" -m flutter_ai serve ;;
esac
