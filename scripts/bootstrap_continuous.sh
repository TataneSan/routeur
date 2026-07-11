#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade "datasets>=2.20" "zstandard>=0.22"
mkdir -p /root/.config/routeur
chmod 700 /root/.config/routeur
echo "Bootstrap complete. Put EDELUX_API_KEY in /root/.config/routeur/retrain.env (chmod 600), then enable deploy/routeur-retrain.timer."
