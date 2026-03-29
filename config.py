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
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    show_intermediate: bool = _as_bool(os.getenv("SHOW_INTERMEDIATE"), True)

    def validate(self) -> None:
        if not self.openai_api_key:
            raise ValueError(
                "Missing OPENAI_API_KEY. Copy .env.example to .env and fill it in."
            )


def get_settings() -> Settings:
    return Settings()
