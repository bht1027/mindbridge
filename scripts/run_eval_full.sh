#!/usr/bin/env bash
set -euo pipefail

# Full reproducible eval (with judge + significance)
JUDGE_MODEL=${JUDGE_MODEL:-gpt-4.1-mini}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-5000}

python evaluate.py \
  --judge-model "$JUDGE_MODEL" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --cases data/eval_cases.json \
  --output-json evaluation_results.json \
  --output-md evaluation_report.md
