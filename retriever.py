from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TOKEN_PATTERN = re.compile(r"[a-zA-Z]{2,}")


@dataclass(frozen=True)
class _IndexedItem:
    data: dict[str, Any]
    tag_tokens: set[str]
    body_tokens: set[str]


class SupportKnowledgeRetriever:
    def __init__(self, kb_path: str | Path) -> None:
        self.kb_path = Path(kb_path)
        self._indexed_items = self._load_items()

    def _load_items(self) -> list[_IndexedItem]:
        raw = json.loads(self.kb_path.read_text(encoding="utf-8"))
        items: list[_IndexedItem] = []
        for entry in raw:
            tags = " ".join(entry.get("tags", []))
            body = " ".join(
                [
                    entry.get("title", ""),
                    entry.get("category", ""),
                    entry.get("content", ""),
                    tags,
                ]
            )
            items.append(
                _IndexedItem(
                    data=entry,
                    tag_tokens=self._tokenize(tags),
                    body_tokens=self._tokenize(body),
                )
            )
        return items

    def _tokenize(self, text: str) -> set[str]:
        return set(_TOKEN_PATTERN.findall(text.lower()))

    def _build_query_text(self, user_input: str, analysis: dict[str, Any] | None) -> str:
        if not analysis:
            return user_input
        chunks = [
            user_input,
            str(analysis.get("emotion", "")),
            str(analysis.get("intent", "")),
            str(analysis.get("topic", "")),
            str(analysis.get("risk_level", "")),
            str(analysis.get("notes", "")),
        ]
        return "\n".join(chunks)

    def retrieve(
        self,
        user_input: str,
        analysis: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        query_tokens = self._tokenize(self._build_query_text(user_input, analysis))

        scored: list[tuple[int, dict[str, Any]]] = []
        for item in self._indexed_items:
            tag_overlap = len(query_tokens & item.tag_tokens)
            body_overlap = len(query_tokens & item.body_tokens)
            score = tag_overlap * 3 + body_overlap
            if score <= 0:
                continue
            payload = dict(item.data)
            payload["match_score"] = score
            scored.append((score, payload))

        scored.sort(key=lambda row: (row[0], row[1].get("id", "")), reverse=True)
        return [payload for _, payload in scored[:top_k]]
