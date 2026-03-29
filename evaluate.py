from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
from time import perf_counter
from typing import Any

from baseline import SingleAgentBaseline
from config import get_settings
from pipeline import MindBridgePipeline
from run_modes import PIPELINE_MODES, get_pipeline_run_config


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _runner_specs(settings) -> list[tuple[str, object]]:
    specs: list[tuple[str, object]] = [("single_agent_baseline", SingleAgentBaseline(settings))]
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
    }


def _summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["runner"], []).append(record)

    summary = []
    for runner_name, runner_records in grouped.items():
        ok_records = [record for record in runner_records if record["status"] == "ok"]
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
                "avg_runtime_seconds": round(avg_runtime, 3),
                "avg_response_chars": round(avg_length, 1),
            }
        )
    return summary


def _build_markdown_report(
    cases_file: Path,
    output_json: Path,
    summary: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> str:
    lines = [
        "# MindBridge Evaluation Report",
        "",
        f"- Generated at: {datetime.now(UTC).isoformat()}",
        f"- Cases file: `{cases_file}`",
        f"- Detailed JSON: `{output_json}`",
        "",
        "## Summary",
        "",
        "| Runner | Cases | Successes | Errors | Avg runtime (s) | Avg response chars |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary:
        lines.append(
            f"| {item['runner']} | {item['cases']} | {item['successes']} | "
            f"{item['errors']} | {item['avg_runtime_seconds']} | {item['avg_response_chars']} |"
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
        if record["error"]:
            lines.extend(
                [
                    "",
                    "### Error",
                    "",
                    f"```text\n{record['error']}\n```",
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
    args = parser.parse_args()

    settings = get_settings()
    settings.validate()

    cases_path = Path(args.cases)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    cases = _load_cases(cases_path)

    runners = _runner_specs(settings)
    records = []
    for case in cases:
        for runner_name, runner in runners:
            records.append(_run_case(case, runner_name, runner))

    summary = _summarize(records)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": settings.openai_model,
        "cases_file": str(cases_path),
        "summary": summary,
        "records": records,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(
        _build_markdown_report(cases_path, output_json, summary, records),
        encoding="utf-8",
    )

    print(f"Saved detailed evaluation results to: {output_json}")
    print(f"Saved Markdown report to: {output_md}")


if __name__ == "__main__":
    main()
