from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from openai import OpenAI

from config import Settings
from prompts import BASELINE_PROMPT


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    return str(response)


@dataclass
class BaselineResult:
    user_input: str
    final_response: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SingleAgentBaseline:
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
            metadata={"system": "baseline"},
        )
