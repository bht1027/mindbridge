"""
Targeted evaluation on medium+high risk cases (cases 25-48).
Complements eval_section_4_3.py which covered low-risk cases 1-15.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from baseline import SingleAgentBaseline
from config import get_settings
from evaluate import (
    BASELINE_RUNNER_NAME,
    _judge_pairwise_records,
    _judge_records,
    _paired_quality_deltas,
    _qualitative_case_studies,
    _quality_summary,
    _run_case,
    _runtime_summary,
    _error_analysis,
)
from judge import SupportiveResponseJudge
from pipeline import MindBridgePipeline
from run_modes import PIPELINE_MODES, get_pipeline_run_config

OUT_DIR = Path("docs/section_4_3")
CASES_FILE = Path("data/eval_cases.json")
RUNNERS_TO_USE = [
    "single_agent_baseline",
    "pipeline_full",
    "pipeline_no_safety",
]
RUNNER_SHORT = {
    "single_agent_baseline": "Baseline",
    "pipeline_full": "Full Pipeline",
    "pipeline_no_safety": "No Safety",
}
BOOTSTRAP_SAMPLES = 2000


def _build_runner_specs(settings, wanted):
    specs = [("single_agent_baseline", SingleAgentBaseline(settings))]
    for mode in PIPELINE_MODES:
        name = f"pipeline_{mode}"
        specs.append((name, MindBridgePipeline(settings, get_pipeline_run_config(mode))))
    return [(n, r) for n, r in specs if n in wanted]


def _run_runner(runner_name, runner, cases):
    return [_run_case(case, runner_name, runner) for case in cases]


def main():
    settings = get_settings()
    settings.validate()
    settings.persistent_memory_enabled = False

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_cases = json.loads(CASES_FILE.read_text())
    # Select medium and high risk cases
    cases = [c for c in all_cases if c["risk_level"] in ("medium", "high")]
    print(f"[highrisk] Cases: {len(cases)} | risk distribution: "
          + str({r: sum(1 for c in cases if c["risk_level"] == r) for r in ("medium", "high")}))

    runner_specs = _build_runner_specs(settings, set(RUNNERS_TO_USE))
    spec_map = dict(runner_specs)
    runner_specs = [(n, spec_map[n]) for n in RUNNERS_TO_USE if n in spec_map]

    t0 = perf_counter()
    all_records = []
    with ThreadPoolExecutor(max_workers=len(runner_specs)) as pool:
        futures = {pool.submit(_run_runner, n, r, cases): n for n, r in runner_specs}
        for future in as_completed(futures):
            name = futures[future]
            records = future.result()
            all_records.extend(records)
            ok = sum(1 for r in records if r["status"] == "ok")
            print(f"  [{name}] {ok}/{len(records)} ok")

    print(f"[highrisk] Runs done in {perf_counter()-t0:.1f}s")

    judge_model = settings.judge_model or settings.openai_model
    judge = SupportiveResponseJudge(settings, model=judge_model)
    print("[highrisk] Judging…")
    t1 = perf_counter()
    _judge_records(all_records, judge)
    print(f"[highrisk] Judging done in {perf_counter()-t1:.1f}s")

    print("[highrisk] Pairwise comparison…")
    pairwise_summary = _judge_pairwise_records(all_records, judge)

    quality_summary = _quality_summary(all_records)
    runtime_summary = _runtime_summary(all_records)
    error_analysis = _error_analysis(all_records)
    paired_deltas = _paired_quality_deltas(
        all_records, baseline_runner=BASELINE_RUNNER_NAME,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
    )
    qualitative = {
        "case_studies": _qualitative_case_studies(all_records),
        "error_analysis": error_analysis,
    }

    report_json = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "medium+high risk cases",
        "n_cases": len(cases),
        "runtime_summary": runtime_summary,
        "quality_summary": quality_summary,
        "paired_quality_deltas": paired_deltas,
        "pairwise_summary": pairwise_summary,
        "qualitative_summary": qualitative,
        "records": all_records,
    }
    out = OUT_DIR / "results_4_3_highrisk.json"
    out.write_text(json.dumps(report_json, ensure_ascii=False, indent=2))
    print(f"[highrisk] Saved: {out}")

    print("\n" + "=" * 60)
    print("MEDIUM+HIGH RISK EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Cases: {len(cases)} | Runners: {len(runner_specs)}")
    print()
    print("Quality (Overall):")
    for row in quality_summary:
        name = RUNNER_SHORT.get(row["runner"], row["runner"])
        print(f"  {name:20s} {row['overall_quality_mean']:.3f} ± {row['overall_quality_stdev']:.3f}"
              f"  (n={row['judged_cases']})")
    pw = pairwise_summary
    if pw.get("enabled"):
        print()
        print(f"Pairwise (pipeline_full vs baseline):")
        print(f"  Pipeline wins: {pw['challenger_wins']}/{pw['judged_pairs']} "
              f"({pw['challenger_win_rate']:.0%})")
        print(f"  Tie-adjusted win rate: {pw['tie_adjusted_challenger_win_rate']:.0%}")
    sr = error_analysis.get("pipeline_safety_recall", {})
    print()
    print(f"Safety recall (pipeline_full): "
          f"{sr.get('high_risk_recall', 0.0):.0%} "
          f"({sr.get('detected_high_risk_cases', 0)}/{sr.get('expected_high_risk_cases', 0)})")


if __name__ == "__main__":
    main()
