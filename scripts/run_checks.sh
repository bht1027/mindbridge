#!/usr/bin/env bash
set -euo pipefail

python -m py_compile \
  app.py \
  agents.py \
  baseline.py \
  config.py \
  demo_streamlit.py \
  evaluate.py \
  judge.py \
  metrics.py \
  pipeline.py \
  prompts.py \
  retriever.py \
  run_modes.py \
  schemas.py

python -m unittest discover -s tests
