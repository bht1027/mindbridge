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

AGENT_OUTPUT_DEFAULTS: dict[str, dict[str, Any]] = {
    "analyzer": {
        "emotion": "unknown",
        "intent": "support_request",
        "topic": "general_support",
        "risk_level": "low",
        "needs_actionable_advice": False,
        "notes": "",
    },
    "empathy": {
        "acknowledgement": "",
        "validation": "",
        "tone_guidance": "warm and concise",
    },
    "strategy": {
        "problem_frame": "",
        "scenario": "general_support",
        "memory_ack": "",
        "failure_mechanism_hypothesis": "",
        "exploration_question": "",
        "suggestions": [],
        "next_step": "",
        "used_knowledge_ids": [],
        "failed_methods_considered": [],
    },
    "safety": {
        "risk_detected": False,
        "risk_level": "low",
        "flags": [],
        "constraints": [],
        "required_actions": [],
    },
    "coordinator": {
        "draft_response": "",
        "used_knowledge_ids": [],
    },
    "critic": {
        "reflection_line": "",
        "thinking_pattern": "",
        "underlying_need": "",
        "reframe_statement": "",
        "issues": [],
        "revision_goals": [],
    },
    "reviser": {
        "final_response": "",
    },
    "final_checker": {
        "approved": False,
        "final_response": "",
        "notes": [],
    },
}

LIST_FIELDS_BY_AGENT = {
    "strategy": {"suggestions", "used_knowledge_ids", "failed_methods_considered"},
    "safety": {"flags", "constraints", "required_actions"},
    "coordinator": {"used_knowledge_ids"},
    "critic": {"issues", "revision_goals"},
    "final_checker": {"notes"},
}

BOOL_FIELDS = {"needs_actionable_advice", "risk_detected", "approved"}
RISK_LEVELS = {"low", "medium", "high", "unknown"}


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


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _repair_agent_output(agent_name: str, parsed: dict[str, Any]) -> dict[str, Any]:
    defaults = AGENT_OUTPUT_DEFAULTS.get(agent_name)
    if defaults is None:
        return parsed

    warnings: list[str] = []
    repaired: dict[str, Any] = {}
    raw_output = parsed.get("raw_output")
    list_fields = LIST_FIELDS_BY_AGENT.get(agent_name, set())

    for key, default_value in defaults.items():
        if key not in parsed:
            value = default_value
            warnings.append(f"missing:{key}")
        else:
            value = parsed[key]

        if key in list_fields:
            repaired[key] = _coerce_list(value)
        elif key in BOOL_FIELDS:
            repaired[key] = _coerce_bool(value)
        elif key == "risk_level":
            risk_level = str(value).strip().lower()
            if risk_level not in RISK_LEVELS:
                warnings.append(f"invalid:risk_level:{risk_level}")
                risk_level = str(default_value)
            repaired[key] = risk_level
        else:
            repaired[key] = "" if value is None else str(value).strip()

    for key, value in parsed.items():
        if key not in repaired and key != "raw_output":
            repaired[key] = value

    if raw_output is not None:
        repaired["raw_output"] = raw_output
        warnings.append("non_json_or_non_object_output")

    if warnings:
        repaired["_validation_warnings"] = warnings

    return repaired


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
        parsed = _repair_agent_output(self.name, parsed)
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
