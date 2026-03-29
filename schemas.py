from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DialogueState:
    user_input: str
    system_name: str = "pipeline"
    run_mode: str = "full"
    analyzer: dict[str, Any] = field(default_factory=dict)
    empathy: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    coordinator: dict[str, Any] = field(default_factory=dict)
    critic: dict[str, Any] = field(default_factory=dict)
    reviser: dict[str, Any] = field(default_factory=dict)
    final_check: dict[str, Any] = field(default_factory=dict)
    final_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
