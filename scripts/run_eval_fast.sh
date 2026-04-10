#!/usr/bin/env bash
set -euo pipefail

# Fast reproducible eval (no judge, quick for CI/local checks)
python evaluate.py \
  --skip-judge \
  --cases data/eval_cases.json \
  --output-json evaluation_results.fast.json \
  --output-md evaluation_report.fast.md
