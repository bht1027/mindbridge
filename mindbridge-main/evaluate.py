from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from baseline import SingleAgentBaseline
from config import get_settings
from judge import SupportiveResponseJudge
from metrics import mean, paired_bootstrap_mean_diff_ci, stdev
from pipeline import MindBridgePipeline
from run_modes import PIPELINE_MODES, get_pipeline_run_config

QUALITY_METRICS = (
    "empathy",
    "helpfulness",
    "safety",
    "naturalness",
    "overall_quality",
)
BASELINE_RUNNER_NAME = "single_agent_baseline"
PIPELINE_FULL_RUNNER_NAME = "pipeline_full"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _runner_specs(settings) -> list[tuple[str, object]]:
    specs: list[tuple[str, object]] = [
        (BASELINE_RUNNER_NAME, SingleAgentBaseline(settings))
    ]
    for mode in PIPELINE_MODES:
        specs.append(
            (
                f"pipeline_{mode}",
                MindBridgePipeline(settings, get_pipeline_run_config(mode)),
            )
        )
    return specs


def _run_case(case: dict[str, Any], runner_name: str, runner: object) -> dict[str, Any]:
    started = perf_counter()
    try:
        result = runner.run(case["message"])
        payload = result.to_dict()
        final_response = getattr(result, "final_response", "")
        error = None
        status = "ok"
    except Exception as exc:
        payload = {}
        final_response = ""
        error = str(exc)
        status = "error"
    elapsed = perf_counter() - started
    return {
        "case_id": case["id"],
        "category": case["category"],
        "expected_risk_level": case["risk_level"],
        "message": case["message"],
        "runner": runner_name,
        "status": status,
        "runtime_seconds": round(elapsed, 3),
        "response_chars": len(final_response),
        "final_response": final_response,
        "error": error,
        "details": payload,
        "judge_status": "skipped",
        "judge_error": None,
        "quality": {},
    }


def _judge_records(records: list[dict[str, Any]], judge: SupportiveResponseJudge) -> None:
    for record in records:
        if record["status"] != "ok":
            record["judge_status"] = "skipped"
            continue
        try:
            judged = judge.judge(
                user_message=record["message"],
                response_text=record["final_response"],
                expected_risk_level=record["expected_risk_level"],
            )
            record["quality"] = judged.to_dict()
            record["judge_status"] = "ok"
            record["judge_error"] = None
        except Exception as exc:
            record["quality"] = {}
            record["judge_status"] = "error"
            record["judge_error"] = str(exc)


def _runtime_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["runner"], []).append(record)

    summary = []
    for runner_name in sorted(grouped):
        runner_records = grouped[runner_name]
        ok_records = [record for record in runner_records if record["status"] == "ok"]
        judged_records = [record for record in runner_records if record["judge_status"] == "ok"]
        avg_runtime = 0.0
        avg_length = 0.0
        if ok_records:
            avg_runtime = sum(record["runtime_seconds"] for record in ok_records) / len(ok_records)
            avg_length = sum(record["response_chars"] for record in ok_records) / len(ok_records)
        summary.append(
            {
                "runner": runner_name,
                "cases": len(runner_records),
                "successes": len(ok_records),
                "errors": len(runner_records) - len(ok_records),
                "judged_cases": len(judged_records),
                "avg_runtime_seconds": round(avg_runtime, 3),
                "avg_response_chars": round(avg_length, 1),
            }
        )
    return summary


def _quality_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["runner"], []).append(record)

    summary = []
    for runner_name in sorted(grouped):
        runner_records = grouped[runner_name]
        judged_records = [
            record
            for record in runner_records
            if record["status"] == "ok" and record["judge_status"] == "ok"
        ]
        row: dict[str, Any] = {
            "runner": runner_name,
            "judged_cases": len(judged_records),
        }
        for metric in QUALITY_METRICS:
            metric_values = [
                float(record["quality"].get(metric, 0.0))
                for record in judged_records
                if metric in record["quality"]
            ]
            row[f"{metric}_mean"] = round(mean(metric_values), 3)
            row[f"{metric}_stdev"] = round(stdev(metric_values), 3)
        summary.append(row)
    return summary


def _paired_quality_deltas(
    records: list[dict[str, Any]],
    baseline_runner: str,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    runners: set[str] = set()

    for record in records:
        runners.add(record["runner"])
        if record["status"] != "ok" or record["judge_status"] != "ok":
            continue
        by_case.setdefault(record["case_id"], {})[record["runner"]] = record

    if baseline_runner not in runners:
        return []

    deltas: list[dict[str, Any]] = []
    for runner_name in sorted(runners):
        if runner_name == baseline_runner:
            continue

        for metric in QUALITY_METRICS:
            left: list[float] = []
            right: list[float] = []
            for case_records in by_case.values():
                if runner_name not in case_records or baseline_runner not in case_records:
                    continue
                left.append(float(case_records[runner_name]["quality"].get(metric, 0.0)))
                right.append(float(case_records[baseline_runner]["quality"].get(metric, 0.0)))

            if not left:
                deltas.append(
                    {
                        "runner": runner_name,
                        "baseline_runner": baseline_runner,
                        "metric": metric,
                        "paired_cases": 0,
                        "mean_delta_vs_baseline": 0.0,
                        "ci_low": 0.0,
                        "ci_high": 0.0,
                        "ci_excludes_zero": False,
                    }
                )
                continue

            delta_values = [l - r for l, r in zip(left, right)]
            ci_low, ci_high = paired_bootstrap_mean_diff_ci(
                left,
                right,
                n_resamples=bootstrap_samples,
            )
            deltas.append(
                {
                    "runner": runner_name,
                    "baseline_runner": baseline_runner,
                    "metric": metric,
                    "paired_cases": len(left),
                    "mean_delta_vs_baseline": round(mean(delta_values), 3),
                    "ci_low": round(ci_low, 3),
                    "ci_high": round(ci_high, 3),
                    "ci_excludes_zero": ci_low > 0 or ci_high < 0,
                }
            )

    return deltas


def _truncate_text(text: str, limit: int = 280) -> str:
    cleaned = (text or "").strip().replace("\n", " ")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _qualitative_case_studies(
    records: list[dict[str, Any]],
    pipeline_runner: str = PIPELINE_FULL_RUNNER_NAME,
    baseline_runner: str = BASELINE_RUNNER_NAME,
    top_k: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_case.setdefault(record["case_id"], {})[record["runner"]] = record

    paired: list[dict[str, Any]] = []
    for case_id, case_records in by_case.items():
        if pipeline_runner not in case_records or baseline_runner not in case_records:
            continue
        left = case_records[pipeline_runner]
        right = case_records[baseline_runner]
        if left["judge_status"] != "ok" or right["judge_status"] != "ok":
            continue
        left_score = float(left["quality"].get("overall_quality", 0.0))
        right_score = float(right["quality"].get("overall_quality", 0.0))
        paired.append(
            {
                "case_id": case_id,
                "category": left["category"],
                "expected_risk_level": left["expected_risk_level"],
                "message": left["message"],
                "pipeline_score": left_score,
                "baseline_score": right_score,
                "delta": round(left_score - right_score, 3),
                "pipeline_response": left["final_response"],
                "baseline_response": right["final_response"],
            }
        )

    paired.sort(key=lambda row: row["delta"], reverse=True)
    return {
        "top_improvements": paired[:top_k],
        "largest_regressions": list(reversed(paired[-top_k:])) if paired else [],
    }


def _error_analysis(
    records: list[dict[str, Any]],
    pipeline_runner: str = PIPELINE_FULL_RUNNER_NAME,
) -> dict[str, Any]:
    runtime_errors_by_runner: dict[str, int] = {}
    judge_errors_by_runner: dict[str, int] = {}
    high_risk_mismatches: list[dict[str, Any]] = []

    for record in records:
        runner = str(record["runner"])
        if record["status"] == "error":
            runtime_errors_by_runner[runner] = runtime_errors_by_runner.get(runner, 0) + 1
        if record["judge_status"] == "error":
            judge_errors_by_runner[runner] = judge_errors_by_runner.get(runner, 0) + 1

        if (
            runner == pipeline_runner
            and record["status"] == "ok"
            and str(record["expected_risk_level"]).lower() == "high"
        ):
            details = record.get("details", {})
            safety = details.get("safety", {}) if isinstance(details, dict) else {}
            predicted = str(safety.get("risk_level", "unknown")).lower()
            if predicted != "high":
                high_risk_mismatches.append(
                    {
                        "case_id": record["case_id"],
                        "message": record["message"],
                        "expected_risk_level": record["expected_risk_level"],
                        "predicted_risk_level": predicted,
                        "safety_flags": safety.get("flags", []),
                    }
                )

    return {
        "runtime_errors_by_runner": runtime_errors_by_runner,
        "judge_errors_by_runner": judge_errors_by_runner,
        "pipeline_high_risk_mismatches": high_risk_mismatches,
    }


def _build_markdown_report(
    cases_file: Path,
    output_json: Path,
    runtime_summary: list[dict[str, Any]],
    quality_summary: list[dict[str, Any]],
    paired_deltas: list[dict[str, Any]],
    qualitative_summary: dict[str, Any],
    records: list[dict[str, Any]],
    judge_enabled: bool,
) -> str:
    lines = [
        "# MindBridge Evaluation Report",
        "",
        f"- Generated at: {datetime.now(UTC).isoformat()}",
        f"- Cases file: `{cases_file}`",
        f"- Detailed JSON: `{output_json}`",
        f"- Judge enabled: `{judge_enabled}`",
        "",
        "## Runtime Summary",
        "",
        "| Runner | Cases | Successes | Errors | Judged | Avg runtime (s) | Avg response chars |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in runtime_summary:
        lines.append(
            f"| {item['runner']} | {item['cases']} | {item['successes']} | "
            f"{item['errors']} | {item['judged_cases']} | {item['avg_runtime_seconds']} | "
            f"{item['avg_response_chars']} |"
        )

    if judge_enabled:
        lines.extend(
            [
                "",
                "## Quality Summary",
                "",
                "| Runner | Judged | Empathy | Helpfulness | Safety | Naturalness | Overall |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in quality_summary:
            lines.append(
                f"| {item['runner']} | {item['judged_cases']} | {item['empathy_mean']} | "
                f"{item['helpfulness_mean']} | {item['safety_mean']} | "
                f"{item['naturalness_mean']} | {item['overall_quality_mean']} |"
            )

        lines.extend(
            [
                "",
                "## Paired Delta Vs Baseline",
                "",
                "| Runner | Metric | Paired cases | Mean delta | 95% CI low | 95% CI high | CI excludes 0 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in paired_deltas:
            lines.append(
                f"| {item['runner']} | {item['metric']} | {item['paired_cases']} | "
                f"{item['mean_delta_vs_baseline']} | {item['ci_low']} | {item['ci_high']} | "
                f"{item['ci_excludes_zero']} |"
            )

        lines.extend(["", "## Qualitative Analysis", ""])
        studies = qualitative_summary.get("case_studies", {})
        improvements = studies.get("top_improvements", [])
        regressions = studies.get("largest_regressions", [])

        lines.extend(["### Case Studies: Top Improvements", ""])
        if not improvements:
            lines.append("_Not enough judged paired cases to build case studies._")
        else:
            for item in improvements:
                lines.extend(
                    [
                        f"- `{item['case_id']}` ({item['category']}, expected risk: `{item['expected_risk_level']}`), "
                        f"overall delta vs baseline: `{item['delta']}`",
                        f"  Message: {_truncate_text(item['message'])}",
                        f"  Pipeline: {_truncate_text(item['pipeline_response'])}",
                        f"  Baseline: {_truncate_text(item['baseline_response'])}",
                    ]
                )

        lines.extend(["", "### Case Studies: Largest Regressions", ""])
        if not regressions:
            lines.append("_No regression pairs found._")
        else:
            for item in regressions:
                lines.extend(
                    [
                        f"- `{item['case_id']}` ({item['category']}, expected risk: `{item['expected_risk_level']}`), "
                        f"overall delta vs baseline: `{item['delta']}`",
                        f"  Message: {_truncate_text(item['message'])}",
                        f"  Pipeline: {_truncate_text(item['pipeline_response'])}",
                        f"  Baseline: {_truncate_text(item['baseline_response'])}",
                    ]
                )

        errors = qualitative_summary.get("error_analysis", {})
        lines.extend(["", "### Error Analysis", ""])
        lines.extend(
            [
                f"- Runtime errors by runner: `{json.dumps(errors.get('runtime_errors_by_runner', {}), ensure_ascii=False)}`",
                f"- Judge errors by runner: `{json.dumps(errors.get('judge_errors_by_runner', {}), ensure_ascii=False)}`",
            ]
        )
        mismatches = errors.get("pipeline_high_risk_mismatches", [])
        lines.append(
            f"- Pipeline high-risk mismatches (expected `high`, predicted not `high`): `{len(mismatches)}`"
        )
        for item in mismatches[:5]:
            lines.append(
                f"  - `{item['case_id']}` predicted `{item['predicted_risk_level']}`; message: {_truncate_text(item['message'])}"
            )

        lines.extend(
            [
                "",
                "### Limitations and Future Work",
                "",
                "- LLM-judge scores are useful but still model-dependent; add human annotation for stronger validity.",
                "- Current dataset is moderate size; expand domain coverage and adversarial edge cases.",
                "- Safety evaluation should include stricter risk-recall targets and dedicated red-team sets.",
            ]
        )

    for record in records:
        lines.extend(
            [
                "",
                f"## {record['case_id']} - {record['runner']}",
                "",
                f"- Category: `{record['category']}`",
                f"- Expected risk level: `{record['expected_risk_level']}`",
                f"- Status: `{record['status']}`",
                f"- Runtime seconds: `{record['runtime_seconds']}`",
                f"- Judge status: `{record['judge_status']}`",
                "",
                "### User message",
                "",
                record["message"],
                "",
                "### Final response",
                "",
                record["final_response"] or "_No response generated._",
            ]
        )

        if record.get("quality"):
            lines.extend(
                [
                    "",
                    "### Quality scores",
                    "",
                    "```json",
                    json.dumps(record["quality"], ensure_ascii=False, indent=2),
                    "```",
                ]
            )

        if record["error"]:
            lines.extend(
                [
                    "",
                    "### Error",
                    "",
                    f"```text\n{record['error']}\n```",
                ]
            )
        if record.get("judge_error"):
            lines.extend(
                [
                    "",
                    "### Judge error",
                    "",
                    f"```text\n{record['judge_error']}\n```",
                ]
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline and ablation variants.")
    parser.add_argument(
        "--cases",
        type=str,
        default="data/eval_cases.json",
        help="Path to the evaluation cases JSON file.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="evaluation_results.json",
        help="Path for detailed evaluation results.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="evaluation_report.md",
        help="Path for the Markdown report.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip quality judging and significance analysis.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        help="Optional judge model override.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of paired bootstrap samples for CI.",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.validate()
    settings.persistent_memory_enabled = False

    cases_path = Path(args.cases)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    cases = _load_cases(cases_path)

    runners = _runner_specs(settings)
    records = []
    for case in cases:
        for runner_name, runner in runners:
            records.append(_run_case(case, runner_name, runner))

    judge_enabled = not args.skip_judge
    judge_model = args.judge_model or settings.judge_model or settings.openai_model
    if judge_enabled:
        judge = SupportiveResponseJudge(settings, model=judge_model)
        _judge_records(records, judge)

    runtime_summary = _runtime_summary(records)
    quality_summary = _quality_summary(records) if judge_enabled else []
    paired_deltas = (
        _paired_quality_deltas(
            records,
            baseline_runner=BASELINE_RUNNER_NAME,
            bootstrap_samples=max(1, args.bootstrap_samples),
        )
        if judge_enabled
        else []
    )
    qualitative_summary = (
        {
            "case_studies": _qualitative_case_studies(records),
            "error_analysis": _error_analysis(records),
        }
        if judge_enabled
        else {}
    )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": settings.openai_model,
        "judge_enabled": judge_enabled,
        "judge_model": judge_model if judge_enabled else None,
        "cases_file": str(cases_path),
        "summary": runtime_summary,
        "runtime_summary": runtime_summary,
        "quality_summary": quality_summary,
        "paired_quality_deltas": paired_deltas,
        "qualitative_summary": qualitative_summary,
        "records": records,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(
        _build_markdown_report(
            cases_path,
            output_json,
            runtime_summary,
            quality_summary,
            paired_deltas,
            qualitative_summary,
            records,
            judge_enabled,
        ),
        encoding="utf-8",
    )

    print(f"Saved detailed evaluation results to: {output_json}")
    print(f"Saved Markdown report to: {output_md}")


if __name__ == "__main__":
    main()
