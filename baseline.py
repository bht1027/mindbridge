from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from openai import OpenAI

from config import Settings
from prompts import BASELINE_PROMPT
from utils import _response_text


@dataclass
class BaselineResult:
    user_input: str
    final_response: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SingleAgentBaseline:
    """One-step LLM baseline used as the comparison system.

    This baseline intentionally does not use MindBridge's analyzer, retriever,
    strategy router, critic, reviser, safety agent, or memory. It tests how far
    a single prompted LLM can go before adding the multi-agent architecture.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)

    def run(self, user_input: str) -> BaselineResult:
        payload = {"user_input": user_input}
        response = self.client.responses.create(
            model=self.settings.openai_model,
            temperature=self.settings.temperature,
            instructions=BASELINE_PROMPT,
            input=json.dumps(payload, ensure_ascii=False, indent=2),
        )
        return BaselineResult(
            user_input=user_input,
            final_response=_response_text(response).strip(),
            metadata={
                "system": "baseline",
                "baseline_type": "single_agent_llm",
                "model": self.settings.openai_model,
                "temperature": self.settings.temperature,
                "uses_input_analyzer": False,
                "uses_retrieval": False,
                "uses_strategy_router": False,
                "uses_parallel_agents": False,
                "uses_reflection_critic": False,
                "uses_reviser": False,
                "uses_final_safety_checker": False,
                "uses_memory": False,
                "comparison_target": "pipeline_full",
            },
        )
