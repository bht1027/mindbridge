from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TOKEN_PATTERN = re.compile(r"[a-zA-Z]{2,}")


@dataclass(frozen=True)
class _IndexedItem:
    data: dict[str, Any]
    tag_tokens: set[str]
    body_tokens: set[str]
    token_counts: dict[str, int]
    text: str


class SupportKnowledgeRetriever:
    def __init__(self, kb_path: str | Path) -> None:
        self.kb_path = Path(kb_path)
        self._indexed_items = self._load_items()
        self._idf = self._build_idf()

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
                    token_counts=self._token_counts(body),
                    text=body,
                )
            )
        return items

    def _tokenize(self, text: str) -> set[str]:
        return set(_TOKEN_PATTERN.findall(text.lower()))

    def _token_counts(self, text: str) -> dict[str, int]:
        return dict(Counter(_TOKEN_PATTERN.findall(text.lower())))

    def _build_idf(self) -> dict[str, float]:
        document_count = len(self._indexed_items)
        if document_count == 0:
            return {}

        document_frequency: Counter[str] = Counter()
        for item in self._indexed_items:
            document_frequency.update(item.token_counts.keys())

        return {
            token: math.log((document_count + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequency.items()
        }

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
            str(analysis.get("strategy_route", "")),
            " ".join(str(item) for item in analysis.get("recommended_focus", []) or []),
        ]
        return "\n".join(chunks)

    def _lexical_score(self, query_tokens: set[str], item: _IndexedItem) -> int:
        tag_overlap = len(query_tokens & item.tag_tokens)
        body_overlap = len(query_tokens & item.body_tokens)
        return tag_overlap * 3 + body_overlap

    def _vector_score(
        self,
        query_counts: dict[str, int],
        item_counts: dict[str, int],
    ) -> float:
        if not query_counts or not item_counts:
            return 0.0

        dot = 0.0
        query_norm = 0.0
        item_norm = 0.0
        default_idf = math.log(len(self._indexed_items) + 1) + 1.0

        for token, count in query_counts.items():
            idf = self._idf.get(token, default_idf)
            query_weight = count * idf
            query_norm += query_weight * query_weight
            if token in item_counts:
                item_weight = item_counts[token] * idf
                dot += query_weight * item_weight

        for token, count in item_counts.items():
            weight = count * self._idf.get(token, default_idf)
            item_norm += weight * weight

        if query_norm == 0.0 or item_norm == 0.0:
            return 0.0
        return dot / math.sqrt(query_norm * item_norm)

    def _rank_items(
        self,
        query_tokens: set[str],
        query_counts: dict[str, int],
        pool_size: int,
    ) -> tuple[list[tuple[_IndexedItem, int]], list[tuple[_IndexedItem, float]]]:
        lexical_ranked: list[tuple[_IndexedItem, int]] = []
        vector_ranked: list[tuple[_IndexedItem, float]] = []

        for item in self._indexed_items:
            lexical_score = self._lexical_score(query_tokens, item)
            vector_score = self._vector_score(query_counts, item.token_counts)
            if lexical_score > 0:
                lexical_ranked.append((item, lexical_score))
            if vector_score > 0:
                vector_ranked.append((item, vector_score))

        lexical_ranked.sort(
            key=lambda row: (row[1], row[0].data.get("id", "")),
            reverse=True,
        )
        vector_ranked.sort(
            key=lambda row: (row[1], row[0].data.get("id", "")),
            reverse=True,
        )
        return lexical_ranked[:pool_size], vector_ranked[:pool_size]

    def _rrf_scores(
        self,
        rankings: list[list[_IndexedItem]],
        rank_constant: int = 60,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, item in enumerate(ranking, start=1):
                item_id = str(item.data.get("id", ""))
                scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (
                    rank_constant + rank
                )
        return scores

    def retrieve(
        self,
        user_input: str,
        analysis: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        query_text = self._build_query_text(user_input, analysis)
        query_tokens = self._tokenize(query_text)
        query_counts = self._token_counts(query_text)
        pool_size = max(top_k * 4, 10)
        lexical_ranked, vector_ranked = self._rank_items(
            query_tokens,
            query_counts,
            pool_size,
        )

        lexical_scores = {
            str(item.data.get("id", "")): score for item, score in lexical_ranked
        }
        vector_scores = {
            str(item.data.get("id", "")): score for item, score in vector_ranked
        }
        candidate_by_id = {
            str(item.data.get("id", "")): item
            for item, _ in [*lexical_ranked, *vector_ranked]
        }
        rrf_scores = self._rrf_scores(
            [
                [item for item, _ in lexical_ranked],
                [item for item, _ in vector_ranked],
            ]
        )

        scored: list[tuple[float, str, dict[str, Any]]] = []
        route = str((analysis or {}).get("strategy_route", ""))
        for item_id, item in candidate_by_id.items():
            lexical_score = lexical_scores.get(item_id, 0)
            vector_score = vector_scores.get(item_id, 0.0)
            route_bonus = 0.01 if route and route == item.data.get("category") else 0.0
            final_score = rrf_scores.get(item_id, 0.0) + route_bonus
            if final_score <= 0:
                continue

            payload = dict(item.data)
            matched_terms = sorted(query_tokens & item.body_tokens)
            payload.update(
                {
                    "match_score": round(final_score, 6),
                    "lexical_score": lexical_score,
                    "vector_score": round(vector_score, 6),
                    "rrf_score": round(rrf_scores.get(item_id, 0.0), 6),
                    "retrieval_method": "hybrid_rrf",
                    "matched_terms": matched_terms[:8],
                }
            )
            scored.append((final_score, item_id, payload))

        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [payload for _, _, payload in scored[:top_k]]
