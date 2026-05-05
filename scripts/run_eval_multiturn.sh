#!/usr/bin/env bash
set -euo pipefail

# Multi-turn edge-case evaluation for memory, failed-advice follow-up, and safety boundaries.
python evaluate.py \
  --skip-judge \
  --cases data/eval_cases_multiturn.json \
  --runners pipeline_full \
  --output-json evaluation_results.multiturn.json \
  --output-md evaluation_report.multiturn.md
