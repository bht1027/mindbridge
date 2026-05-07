"""
Section 4.3 Evaluation Script
Runs baseline + pipeline ablations in parallel, judges responses,
computes statistical significance (bootstrap CI), and generates
presentation-ready charts + report.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------- project imports ----------
from baseline import SingleAgentBaseline
from config import get_settings
from evaluate import (
    BASELINE_RUNNER_NAME,
    PIPELINE_FULL_RUNNER_NAME,
    QUALITY_METRICS,
    _error_analysis,
    _judge_pairwise_records,
    _judge_records,
    _paired_quality_deltas,
    _qualitative_case_studies,
    _quality_summary,
    _rate,
    _run_case,
    _runtime_summary,
)
from judge import SupportiveResponseJudge
from pipeline import MindBridgePipeline
from run_modes import PIPELINE_MODES, get_pipeline_run_config

# ---------- config ----------
OUT_DIR = Path("docs/section_4_3")
CASES_FILE = Path("data/eval_cases.json")
MAX_CASES = 15          # cap for speed; covers all risk levels
BOOTSTRAP_SAMPLES = 2000
RUNNERS_TO_USE = [      # baseline + 5 ablations (skip no_empathy, no_strategy for brevity)
    "single_agent_baseline",
    "pipeline_full",
    "pipeline_no_critic",
    "pipeline_no_reviser",
    "pipeline_no_safety",
    "pipeline_no_retrieval",
]

METRIC_LABELS = {
    "empathy": "Empathy",
    "helpfulness": "Helpfulness",
    "safety": "Safety",
    "naturalness": "Naturalness",
    "overall_quality": "Overall",
}

RUNNER_SHORT = {
    "single_agent_baseline": "Baseline",
    "pipeline_full": "Full Pipeline",
    "pipeline_no_critic": "No Critic",
    "pipeline_no_reviser": "No Reviser",
    "pipeline_no_safety": "No Safety",
    "pipeline_no_retrieval": "No Retrieval",
}

COLORS = {
    "single_agent_baseline": "#6c757d",
    "pipeline_full": "#1a6fc4",
    "pipeline_no_critic": "#e07b39",
    "pipeline_no_reviser": "#8e44ad",
    "pipeline_no_safety": "#c0392b",
    "pipeline_no_retrieval": "#27ae60",
}


def _build_runner_specs(settings, wanted: list[str]):
    specs = [("single_agent_baseline", SingleAgentBaseline(settings))]
    for mode in PIPELINE_MODES:
        name = f"pipeline_{mode}"
        specs.append((name, MindBridgePipeline(settings, get_pipeline_run_config(mode))))
    return [(n, r) for n, r in specs if n in wanted]


def _run_runner(runner_name, runner, cases):
    """Run all cases for one runner (called in a thread)."""
    results = []
    for case in cases:
        results.append(_run_case(case, runner_name, runner))
    return results


# ================================================================
#  CHART 1 – Bar chart: quality metrics by runner
# ================================================================
def _chart_quality_bars(quality_summary, out_path: Path):
    runners = [q["runner"] for q in quality_summary]
    metrics = list(QUALITY_METRICS)
    x = np.arange(len(metrics))
    width = 0.8 / len(runners)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, row in enumerate(quality_summary):
        means = [row[f"{m}_mean"] for m in metrics]
        offset = (i - len(runners) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width * 0.9,
                      label=RUNNER_SHORT.get(row["runner"], row["runner"]),
                      color=COLORS.get(row["runner"], "#999"))
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.04,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=11)
    ax.set_ylim(0, 5.8)
    ax.set_ylabel("Score (1–5)", fontsize=11)
    ax.set_title("4.3.1 Quality Metrics by Runner", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.axhline(y=5, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ================================================================
#  CHART 2 – Horizontal bar: mean delta vs baseline (overall)
# ================================================================
def _chart_delta_vs_baseline(paired_deltas, out_path: Path):
    overall = [d for d in paired_deltas if d["metric"] == "overall_quality"]
    if not overall:
        return
    runners = [RUNNER_SHORT.get(d["runner"], d["runner"]) for d in overall]
    deltas = [d["mean_delta_vs_baseline"] for d in overall]
    ci_lows = [d["ci_low"] for d in overall]
    ci_highs = [d["ci_high"] for d in overall]
    colors_list = [COLORS.get(d["runner"], "#999") for d in overall]

    fig, ax = plt.subplots(figsize=(9, max(3, len(runners) * 0.7 + 1)))
    y = np.arange(len(runners))
    xerr = np.array([
        [d - l for d, l in zip(deltas, ci_lows)],
        [h - d for d, h in zip(deltas, ci_highs)],
    ])
    bars = ax.barh(y, deltas, color=colors_list, xerr=xerr,
                   error_kw={"ecolor": "#333", "capsize": 4, "lw": 1.5},
                   height=0.55)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(runners, fontsize=10)
    ax.set_xlabel("Mean Δ Overall Quality vs Baseline (95% CI)", fontsize=10)
    ax.set_title("4.3.2 Comparative Analysis – Overall Quality vs Baseline",
                 fontsize=12, fontweight="bold")

    for bar, val, ci_low, ci_high in zip(bars, deltas, ci_lows, ci_highs):
        sig = ci_low > 0 or ci_high < 0
        label = f"{val:+.3f}" + (" *" if sig else "")
        ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left" if val >= 0 else "right", fontsize=9)

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ================================================================
#  CHART 3 – Heatmap: ablation Δ per metric
# ================================================================
def _chart_ablation_heatmap(paired_deltas, out_path: Path):
    ablation_runners = [
        r for r in RUNNERS_TO_USE
        if r not in (BASELINE_RUNNER_NAME, PIPELINE_FULL_RUNNER_NAME)
    ]
    metrics = list(QUALITY_METRICS)
    data = np.zeros((len(ablation_runners), len(metrics)))

    by_runner_metric = {(d["runner"], d["metric"]): d for d in paired_deltas}
    for i, runner in enumerate(ablation_runners):
        for j, metric in enumerate(metrics):
            entry = by_runner_metric.get((runner, metric))
            if entry:
                data[i, j] = entry["mean_delta_vs_baseline"]

    fig, ax = plt.subplots(figsize=(9, max(3, len(ablation_runners) * 0.8 + 1.5)))
    vmax = max(0.5, np.abs(data).max())
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Δ vs Baseline")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=10)
    ax.set_yticks(range(len(ablation_runners)))
    ax.set_yticklabels([RUNNER_SHORT.get(r, r) for r in ablation_runners], fontsize=10)
    ax.set_title("4.3.4 Ablation Study – Δ per Metric vs Baseline",
                 fontsize=12, fontweight="bold")

    for i in range(len(ablation_runners)):
        for j in range(len(metrics)):
            val = data[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(val) < vmax * 0.6 else "white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ================================================================
#  CHART 4 – Pie chart: pairwise preference
# ================================================================
def _chart_pairwise_pie(pairwise_summary, out_path: Path):
    if not pairwise_summary.get("enabled"):
        return
    wins = pairwise_summary.get("challenger_wins", 0)
    losses = pairwise_summary.get("baseline_wins", 0)
    ties = pairwise_summary.get("ties", 0)
    total = wins + losses + ties
    if total == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [
        f"Pipeline Wins\n({wins}/{total})",
        f"Baseline Wins\n({losses}/{total})",
        f"Ties\n({ties}/{total})",
    ]
    sizes = [wins, losses, ties]
    colors = ["#1a6fc4", "#6c757d", "#adb5bd"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.0f%%", startangle=90,
        textprops={"fontsize": 10},
    )
    ax.set_title("4.3.2 Pairwise Preference:\nFull Pipeline vs Baseline",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ================================================================
#  CHART 5 – Runtime comparison
# ================================================================
def _chart_runtime(runtime_summary, out_path: Path):
    runners = [RUNNER_SHORT.get(r["runner"], r["runner"]) for r in runtime_summary]
    runtimes = [r["avg_runtime_seconds"] for r in runtime_summary]
    colors_list = [COLORS.get(r["runner"], "#999") for r in runtime_summary]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(runners, runtimes, color=colors_list, width=0.6)
    for bar, val in zip(bars, runtimes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Avg Runtime (seconds)", fontsize=11)
    ax.set_title("4.3.1 Performance – Average Response Time", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ================================================================
#  CHART 6 – Safety recall bar
# ================================================================
    plt.close(fig)


# ================================================================
#  Presentation report markdown
# ================================================================
def _build_presentation_report(
    quality_summary,
    paired_deltas,
    pairwise_summary,
    qualitative,
    error_analysis,
    runtime_summary,
    n_cases,
    generated_at,
) -> str:
    lines = [
        "# MindBridge – Section 4.3 Evaluation Report",
        f"*Generated {generated_at} | n={n_cases} cases | Runners: Baseline + 5 ablations*",
        "",
        "---",
        "",
        "## 4.3.1 Performance Measurements",
        "",
        "### Runtime & Response Length",
        "",
        "| System | Avg Runtime (s) | Avg Response Chars |",
        "| --- | ---: | ---: |",
    ]
    for row in runtime_summary:
        lines.append(
            f"| {RUNNER_SHORT.get(row['runner'], row['runner'])} "
            f"| {row['avg_runtime_seconds']} | {row['avg_response_chars']} |"
        )

    lines.extend([
        "",
        "### Quality Scores (LLM-Judge, 1–5)",
        "",
        "| System | Judged | Empathy | Helpfulness | Safety | Naturalness | Overall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in quality_summary:
        lines.append(
            f"| {RUNNER_SHORT.get(row['runner'], row['runner'])} "
            f"| {row['judged_cases']} "
            f"| {row['empathy_mean']:.2f} ± {row['empathy_stdev']:.2f} "
            f"| {row['helpfulness_mean']:.2f} ± {row['helpfulness_stdev']:.2f} "
            f"| {row['safety_mean']:.2f} ± {row['safety_stdev']:.2f} "
            f"| {row['naturalness_mean']:.2f} ± {row['naturalness_stdev']:.2f} "
            f"| **{row['overall_quality_mean']:.2f}** ± {row['overall_quality_stdev']:.2f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4.3.2 Comparative Analysis – Baseline vs Full Pipeline",
        "",
    ])

    if pairwise_summary.get("enabled"):
        pw = pairwise_summary
        lines.extend([
            f"- **Judged pairs**: {pw.get('judged_pairs', 0)}",
            f"- **Pipeline wins**: {pw.get('challenger_wins', 0)} "
            f"({pw.get('challenger_win_rate', 0.0):.1%})",
            f"- **Baseline wins**: {pw.get('baseline_wins', 0)} "
            f"({pw.get('baseline_win_rate', 0.0):.1%})",
            f"- **Ties**: {pw.get('ties', 0)} ({pw.get('tie_rate', 0.0):.1%})",
            f"- **Tie-adjusted pipeline win rate**: "
            f"**{pw.get('tie_adjusted_challenger_win_rate', 0.0):.1%}**",
            "",
        ])

    lines.extend([
        "### Paired Δ vs Baseline by Metric (95% Bootstrap CI)",
        "",
        "| Runner | Metric | Paired n | Δ Mean | CI Low | CI High | Sig? |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for d in paired_deltas:
        sig = "✓" if d["ci_excludes_zero"] else "–"
        lines.append(
            f"| {RUNNER_SHORT.get(d['runner'], d['runner'])} "
            f"| {METRIC_LABELS.get(d['metric'], d['metric'])} "
            f"| {d['paired_cases']} "
            f"| {d['mean_delta_vs_baseline']:+.3f} "
            f"| {d['ci_low']:+.3f} "
            f"| {d['ci_high']:+.3f} "
            f"| {sig} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4.3.3 Error Analysis",
        "",
    ])
    safety_recall = error_analysis.get("pipeline_safety_recall", {})
    runtime_errs = error_analysis.get("runtime_errors_by_runner", {})
    judge_errs = error_analysis.get("judge_errors_by_runner", {})
    lines.extend([
        f"- **High-risk recall**: "
        f"{safety_recall.get('high_risk_recall', 0.0):.0%} "
        f"({safety_recall.get('detected_high_risk_cases', 0)}/"
        f"{safety_recall.get('expected_high_risk_cases', 0)} cases detected)",
        f"- **Missed high-risk cases**: {safety_recall.get('missed_high_risk_cases', 0)}",
        f"- **Runtime errors**: {sum(runtime_errs.values())} total across all runners",
        f"- **Judge errors**: {sum(judge_errs.values())} total",
        "",
    ])
    mismatches = error_analysis.get("pipeline_high_risk_mismatches", [])
    if mismatches:
        lines.append("**Undetected high-risk cases (up to 3):**")
        lines.append("")
        for item in mismatches[:3]:
            msg = item["message"][:120].replace("\n", " ")
            lines.append(
                f"- `{item['case_id']}` predicted `{item['predicted_risk_level']}`: "
                f"_{msg}…_"
            )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 4.3.4 Ablation Studies",
        "",
        "| Removed Component | Overall Δ | Empathy Δ | Helpfulness Δ | Safety Δ | Naturalness Δ | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])

    ablation_runners_ordered = [
        "pipeline_no_critic",
        "pipeline_no_reviser",
        "pipeline_no_safety",
        "pipeline_no_retrieval",
    ]
    by_runner_metric = {(d["runner"], d["metric"]): d for d in paired_deltas}
    for runner in ablation_runners_ordered:
        def get_delta(m):
            e = by_runner_metric.get((runner, m))
            return f"{e['mean_delta_vs_baseline']:+.3f}" if e else "n/a"
        overall_d = by_runner_metric.get((runner, "overall_quality"))
        verdict = ""
        if overall_d:
            delta = overall_d["mean_delta_vs_baseline"]
            sig = overall_d["ci_excludes_zero"]
            if sig and delta < -0.1:
                verdict = "Component matters ✓"
            elif sig and delta > 0.1:
                verdict = "Removal helps?"
            else:
                verdict = "Marginal"
        lines.append(
            f"| {RUNNER_SHORT.get(runner, runner)} "
            f"| {get_delta('overall_quality')} "
            f"| {get_delta('empathy')} "
            f"| {get_delta('helpfulness')} "
            f"| {get_delta('safety')} "
            f"| {get_delta('naturalness')} "
            f"| {verdict} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4.3.5 Qualitative Analysis – Case Studies",
        "",
        "### Top Improvements (Pipeline vs Baseline)",
        "",
    ])
    studies = qualitative.get("case_studies", {})
    for item in studies.get("top_improvements", []):
        msg = (item["message"] or "")[:200].replace("\n", " ")
        pipeline_r = (item["pipeline_response"] or "")[:300].replace("\n", " ")
        baseline_r = (item["baseline_response"] or "")[:300].replace("\n", " ")
        lines.extend([
            f"**Case `{item['case_id']}`** ({item['category']}, "
            f"risk={item['expected_risk_level']}) – Overall Δ = **{item['delta']:+.2f}**",
            "",
            f"> *User:* {msg}",
            "",
            f"**Pipeline:** {pipeline_r}",
            "",
            f"**Baseline:** {baseline_r}",
            "",
        ])

    lines.extend([
        "### Largest Regressions",
        "",
    ])
    for item in studies.get("largest_regressions", []):
        msg = (item["message"] or "")[:200].replace("\n", " ")
        lines.extend([
            f"**Case `{item['case_id']}`** ({item['category']}, "
            f"risk={item['expected_risk_level']}) – Overall Δ = **{item['delta']:+.2f}**",
            "",
            f"> *User:* {msg}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 4.3.6 Limitations and Future Work",
        "",
        "- **LLM-judge validity**: Scores are model-dependent; human annotation would strengthen validity.",
        "- **Dataset size**: n=15 per runner gives directional signal but wide CIs; "
        "  expanding to 50–100 cases would tighten estimates.",
        "- **Safety coverage**: High-risk recall should target ≥ 0.9; "
        "  a dedicated red-team set with adversarial inputs is needed.",
        "- **Latency**: The full pipeline adds 1.5–3× latency vs baseline; "
        "  async/streaming would improve perceived responsiveness.",
        "- **User study**: A/B study with real users would replace proxy judge scores "
        "  and measure actual support outcomes.",
        "",
        "---",
        "*Charts: `docs/section_4_3/` | Raw data: `docs/section_4_3/results_4_3.json`*",
    ])
    return "\n".join(lines) + "\n"


# ================================================================
#  Main
# ================================================================
def main():
    settings = get_settings()
    settings.validate()
    settings.persistent_memory_enabled = False

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = json.loads(CASES_FILE.read_text())[:MAX_CASES]
    print(f"[4.3] Running evaluation: {len(cases)} cases × {len(RUNNERS_TO_USE)} runners")

    # ---- build runner specs ----
    runner_specs = _build_runner_specs(settings, set(RUNNERS_TO_USE))
    # re-order to match RUNNERS_TO_USE order
    spec_map = dict(runner_specs)
    runner_specs = [(n, spec_map[n]) for n in RUNNERS_TO_USE if n in spec_map]

    # ---- parallel execution (one thread per runner) ----
    t0 = perf_counter()
    all_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(runner_specs)) as pool:
        futures = {
            pool.submit(_run_runner, name, runner, cases): name
            for name, runner in runner_specs
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                records = future.result()
                all_records.extend(records)
                successes = sum(1 for r in records if r["status"] == "ok")
                print(f"  [{name}] {successes}/{len(records)} ok")
            except Exception as exc:
                print(f"  [{name}] FAILED: {exc}", file=sys.stderr)

    run_wall = perf_counter() - t0
    print(f"[4.3] All runners done in {run_wall:.1f}s")

    # ---- judge ----
    judge_model = settings.judge_model or settings.openai_model
    judge = SupportiveResponseJudge(settings, model=judge_model)
    print("[4.3] Judging responses…")
    t1 = perf_counter()
    _judge_records(all_records, judge)
    judge_wall = perf_counter() - t1
    print(f"[4.3] Judging done in {judge_wall:.1f}s")

    # ---- pairwise ----
    print("[4.3] Pairwise comparison…")
    pairwise_summary = _judge_pairwise_records(all_records, judge)

    # ---- aggregate metrics ----
    runtime_summary = _runtime_summary(all_records)
    quality_summary = _quality_summary(all_records)
    paired_deltas = _paired_quality_deltas(
        all_records,
        baseline_runner=BASELINE_RUNNER_NAME,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
    )
    qualitative = {
        "case_studies": _qualitative_case_studies(all_records),
        "error_analysis": _error_analysis(all_records),
    }
    error_analysis = qualitative["error_analysis"]

    # ---- save JSON ----
    report_json = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_cases": len(cases),
        "model": settings.openai_model,
        "judge_model": judge_model,
        "runtime_summary": runtime_summary,
        "quality_summary": quality_summary,
        "paired_quality_deltas": paired_deltas,
        "pairwise_summary": pairwise_summary,
        "qualitative_summary": qualitative,
        "records": all_records,
    }
    json_path = OUT_DIR / "results_4_3.json"
    json_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2))
    print(f"[4.3] Saved JSON: {json_path}")

    # ---- generate charts ----
    print("[4.3] Generating charts…")
    _chart_quality_bars(quality_summary, OUT_DIR / "chart1_quality_bars.png")
    _chart_delta_vs_baseline(paired_deltas, OUT_DIR / "chart2_delta_vs_baseline.png")
    _chart_ablation_heatmap(paired_deltas, OUT_DIR / "chart3_ablation_heatmap.png")
    _chart_pairwise_pie(pairwise_summary, OUT_DIR / "chart4_pairwise_pie.png")
    _chart_runtime(runtime_summary, OUT_DIR / "chart5_runtime.png")
    # ---- presentation markdown report ----
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    report_md = _build_presentation_report(
        quality_summary,
        paired_deltas,
        pairwise_summary,
        qualitative,
        error_analysis,
        runtime_summary,
        len(cases),
        generated_at,
    )
    md_path = OUT_DIR / "presentation_report.md"
    md_path.write_text(report_md, encoding="utf-8")
    print(f"[4.3] Saved presentation report: {md_path}")

    # ---- print summary to console ----
    print("\n" + "=" * 60)
    print("SECTION 4.3 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Cases: {len(cases)} | Runners: {len(runner_specs)}")
    print(f"Wall time: runs {run_wall:.0f}s + judging {judge_wall:.0f}s")
    print()
    print("Quality (Overall mean ± stdev):")
    for row in quality_summary:
        name = RUNNER_SHORT.get(row["runner"], row["runner"])
        print(f"  {name:20s} {row['overall_quality_mean']:.3f} ± {row['overall_quality_stdev']:.3f}"
              f"  (n={row['judged_cases']})")
    if pairwise_summary.get("enabled"):
        pw = pairwise_summary
        print()
        print(f"Pairwise preference (pipeline_full vs baseline):")
        print(f"  Pipeline wins: {pw['challenger_wins']}/{pw['judged_pairs']} "
              f"({pw['challenger_win_rate']:.0%})")
        print(f"  Tie-adjusted win rate: {pw['tie_adjusted_challenger_win_rate']:.0%}")
    sr = error_analysis.get("pipeline_safety_recall", {})
    print()
    print(f"Safety recall (pipeline_full): "
          f"{sr.get('high_risk_recall', 0.0):.0%} "
          f"({sr.get('detected_high_risk_cases', 0)}/{sr.get('expected_high_risk_cases', 0)})")
    print()
    print(f"Charts saved to: {OUT_DIR}/")
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
