#!/usr/bin/env bash
set -euo pipefail

# Cheap reproducibility checks that do not require an OpenAI API key.
python -m py_compile \
  agents.py \
  app.py \
  baseline.py \
  config.py \
  demo_streamlit.py \
  eval_section_4_3.py \
  eval_section_4_3_highrisk.py \
  evaluate.py \
  judge.py \
  metrics.py \
  pipeline.py \
  prompts.py \
  retriever.py \
  run_modes.py \
  schemas.py

python -m json.tool data/eval_cases.json > /dev/null
python -m json.tool data/eval_cases_multiturn.json > /dev/null
python -m json.tool data/support_kb.json > /dev/null
python -m json.tool data/user_profiles.example.json > /dev/null

python app.py --help > /dev/null
python evaluate.py --help > /dev/null

echo "Smoke checks passed."
