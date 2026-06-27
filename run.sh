#!/usr/bin/env bash
set -euo pipefail
export PORT="${PORT:-7860}"
./.venv/bin/python3 gradio_app.py
