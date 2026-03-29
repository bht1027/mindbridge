import argparse
import json
from pathlib import Path

from baseline import SingleAgentBaseline
from config import get_settings
from pipeline import MindBridgePipeline
from run_modes import PIPELINE_MODES, get_pipeline_run_config


def _build_markdown_output(
    title: str,
    final_response: str,
    payload: dict[str, object],
    include_intermediate: bool,
) -> str:
    sections = [
        f"# {title}",
        "",
        "## Final response",
        "",
        final_response or "_No response generated._",
    ]
    if include_intermediate and payload:
        sections.extend(
            [
                "",
                "## Run details",
                "",
                "```json",
                json.dumps(payload, ensure_ascii=False, indent=2),
                "```",
            ]
        )
    return "\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MindBridge pipeline.")
    parser.add_argument(
        "--message",
        type=str,
        help="User message to send into the supportive dialogue pipeline.",
    )
    parser.add_argument(
        "--system",
        choices=("pipeline", "baseline"),
        default="pipeline",
        help="Which system to run.",
    )
    parser.add_argument(
        "--mode",
        choices=PIPELINE_MODES,
        default="full",
        help="Pipeline mode for ablation comparisons.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        help="Optional path to save the result as a Markdown file.",
    )
    args = parser.parse_args()

    user_message = args.message or input("User: ").strip()
    settings = get_settings()
    settings.validate()

    if args.system == "baseline":
        runner = SingleAgentBaseline(settings)
        result = runner.run(user_message)
        title = "MindBridge Single-Agent Baseline Output"
        final_response = result.final_response
        payload = result.to_dict()
    else:
        pipeline = MindBridgePipeline(
            settings,
            run_config=get_pipeline_run_config(args.mode),
        )
        result = pipeline.run(user_message)
        title = f"MindBridge Pipeline Output ({args.mode})"
        final_response = result.final_response
        payload = result.to_dict()

    print("\nFinal response:\n")
    print(final_response)

    if settings.show_intermediate:
        print("\nRun details:\n")
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _build_markdown_output(
                title,
                final_response,
                payload,
                settings.show_intermediate,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved Markdown output to: {output_path}")


if __name__ == "__main__":
    main()
