from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


def _as_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    judge_model: str = os.getenv("JUDGE_MODEL", "")
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    show_intermediate: bool = _as_bool(os.getenv("SHOW_INTERMEDIATE"), True)
    support_kb_path: str = os.getenv("SUPPORT_KB_PATH", "data/support_kb.json")
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    memory_window: int = int(os.getenv("MEMORY_WINDOW", "4"))
    response_style: str = os.getenv("RESPONSE_STYLE", "conversational").strip().lower()
    therapist_flow_strict: bool = _as_bool(
        os.getenv("THERAPIST_FLOW_STRICT"),
        True,
    )
    persistent_memory_enabled: bool = _as_bool(
        os.getenv("PERSISTENT_MEMORY_ENABLED"),
        True,
    )
    persistent_memory_path: str = os.getenv(
        "PERSISTENT_MEMORY_PATH",
        "data/user_profiles.json",
    )
    persistent_history_limit: int = int(os.getenv("PERSISTENT_HISTORY_LIMIT", "200"))

    def validate(self) -> None:
        if not self.openai_api_key:
            raise ValueError(
                "Missing OPENAI_API_KEY. Copy .env.example to .env and fill it in."
            )
        if self.retrieval_top_k < 1:
            raise ValueError("RETRIEVAL_TOP_K must be >= 1.")
        if self.memory_window < 0:
            raise ValueError("MEMORY_WINDOW must be >= 0.")
        if self.persistent_history_limit < 1:
            raise ValueError("PERSISTENT_HISTORY_LIMIT must be >= 1.")
        if self.response_style not in {"structured", "conversational"}:
            raise ValueError("RESPONSE_STYLE must be one of: structured, conversational.")


def get_settings() -> Settings:
    return Settings()
