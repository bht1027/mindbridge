from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DialogueState:
    user_input: str
    system_name: str = "pipeline"
    run_mode: str = "full"
    profile_id: str = "default"
    session_stage: str = "rapport_and_safety"
    stage_goal: str = ""
    memory_context: list[dict[str, Any]] = field(default_factory=list)
    user_state: dict[str, Any] = field(default_factory=dict)
    analyzer: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    retrieved_knowledge: list[dict[str, Any]] = field(default_factory=list)
    strategy_route: dict[str, Any] = field(default_factory=dict)
    empathy: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    safety_trace: dict[str, Any] = field(default_factory=dict)
    coordinator: dict[str, Any] = field(default_factory=dict)
    critic: dict[str, Any] = field(default_factory=dict)
    reviser: dict[str, Any] = field(default_factory=dict)
    final_check: dict[str, Any] = field(default_factory=dict)
    final_response_sections: dict[str, str] = field(default_factory=dict)
    final_response: str = ""
    early_turn_override: str = ""  # set on turns 1-3 to bypass reviser/checker output

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
