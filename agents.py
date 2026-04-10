from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from config import Settings
from prompts import (
    ANALYZER_PROMPT,
    COORDINATOR_PROMPT,
    CRITIC_PROMPT,
    EMPATHY_PROMPT,
    FINAL_CHECKER_PROMPT,
    REVISER_PROMPT,
    SAFETY_PROMPT,
    STRATEGY_PROMPT,
)


def _strip_code_fences(text: str) -> str: 
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("\n", 1)[0]
    return cleaned.strip()


def _safe_json_loads(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        return {"raw_output": parsed}
    except json.JSONDecodeError:
        return {"raw_output": text}


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    return str(response)


@dataclass
class JsonAgent:
    name: str
    system_prompt: str
    client: OpenAI
    settings: Settings

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.responses.create(
            model=self.settings.openai_model,
            temperature=self.settings.temperature,
            instructions=self.system_prompt,
            input=json.dumps(payload, ensure_ascii=False, indent=2),
        )
        parsed = _safe_json_loads(_response_text(response))
        parsed.setdefault("_agent", self.name)
        return parsed


def build_agents(settings: Settings) -> dict[str, JsonAgent]:
    client = OpenAI(api_key=settings.openai_api_key)
    return {
        "analyzer": JsonAgent("analyzer", ANALYZER_PROMPT, client, settings),
        "empathy": JsonAgent("empathy", EMPATHY_PROMPT, client, settings),
        "strategy": JsonAgent("strategy", STRATEGY_PROMPT, client, settings),
        "safety": JsonAgent("safety", SAFETY_PROMPT, client, settings),
        "coordinator": JsonAgent(
            "coordinator", COORDINATOR_PROMPT, client, settings
        ),
        "critic": JsonAgent("critic", CRITIC_PROMPT, client, settings),
        "reviser": JsonAgent("reviser", REVISER_PROMPT, client, settings),
        "final_checker": JsonAgent(
            "final_checker", FINAL_CHECKER_PROMPT, client, settings
        ),
    }
