from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agents import build_agents
from config import Settings
from retriever import SupportKnowledgeRetriever
from run_modes import PipelineRunConfig, get_pipeline_run_config
from schemas import DialogueState

HIGH_RISK_PATTERNS = (
    "hopeless",
    "nothing matters",
    "nothing matters anymore",
    "end it",
    "end everything",
    "no reason to live",
    "cannot go on",
    "can't go on",
    "giving up",
    "hurt myself",
    "harm myself",
    "kill myself",
    "end my life",
    "do not want to keep going",
    "don't want to keep going",
    "want to disappear",
    "better off without me",
    "wish i was gone",
    "wish i were gone",
    "hurt someone",
    "not feel safe at home",
)

STRATEGY_PLAYBOOK = (
    {
        "scenario": "safety_escalation",
        "priority": 100,
        "keywords": (
            "hopeless",
            "nothing matters",
            "giving up",
            "want to disappear",
            "end it",
            "end my life",
            "hurt myself",
            "kill myself",
        ),
        "focus": (
            "immediate safety anchoring",
            "reach trusted person now",
            "reduce isolation right away",
            "use crisis resources if risk rises",
        ),
    },
    {
        "scenario": "conflict_resolution",
        "priority": 90,
        "keywords": (
            "roommate",
            "housemate",
            "conflict",
            "argue",
            "fight",
            "tension at home",
        ),
        "focus": (
            "de-escalation",
            "communication timing",
            "boundary setting",
            "temporary space strategy",
        ),
    },
    {
        "scenario": "sleep_support",
        "priority": 80,
        "keywords": (
            "sleep",
            "insomnia",
            "night",
            "cannot sleep",
            "can't sleep",
            "sleep issue",
            "racing thoughts",
        ),
        "focus": (
            "sleep hygiene",
            "worry scheduling",
            "journaling",
            "future uncertainty reframing",
        ),
    },
    {
        "scenario": "decision_support",
        "priority": 75,
        "keywords": (
            "decision",
            "cannot decide",
            "can't decide",
            "choose",
            "choice",
            "stuck between",
            "overwhelmed by options",
            "decision overwhelm",
        ),
        "focus": (
            "decision criteria clarification",
            "tradeoff comparison",
            "small reversible experiment",
            "deadline-backed commitment",
        ),
    },
    {
        "scenario": "job_search_support",
        "priority": 70,
        "keywords": (
            "job",
            "application",
            "resume",
            "interview",
            "internship",
            "rejection",
            "career",
        ),
        "focus": (
            "resume refinement",
            "application funnel diagnosis",
            "feedback loop",
            "networking strategy",
        ),
    },
    {
        "scenario": "academic_overload",
        "priority": 60,
        "keywords": ("assignment", "school", "exam", "study", "deadline"),
        "focus": (
            "task decomposition",
            "priority by deadline and effort",
            "study block planning",
            "instructor support request",
        ),
    },
)

GENERIC_ACTIONS_TO_AVOID = ("drink water", "go outside", "take a deep breath")

FAILURE_CUES = (
    "didn't help",
    "did not help",
    "didn't work",
    "did not work",
    "not helping",
    "not working",
    "still wake",
    "still waking",
    "still can't sleep",
    "still cannot sleep",
    "no relief",
)

METHOD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "journaling": ("journaling", "journal entry"),
    "breathing": ("breathing", "slow breathing", "slow exhale", "deep breath"),
    "worry_scheduling": ("worry scheduling", "worry time"),
    "notepad_capture": ("notepad", "thought parking", "jot down"),
    "wind_down_routine": ("wind-down", "wind down", "dimmed lights", "offline calming"),
    "sleep_hygiene": (
        "sleep hygiene",
        "caffeine cutoff",
        "consistent wake time",
        "bed is for sleep",
    ),
    "resume_refinement": ("resume refinement", "resume bullet", "resume"),
    "application_funnel": ("application funnel", "funnel diagnosis", "applications"),
}

METHOD_LABELS: dict[str, str] = {
    "journaling": "journaling",
    "breathing": "breathing",
    "worry_scheduling": "worry scheduling",
    "notepad_capture": "notepad thought-capture",
    "wind_down_routine": "wind-down routine",
    "sleep_hygiene": "sleep hygiene",
    "resume_refinement": "resume refinement",
    "application_funnel": "application funnel",
}

ROUTE_FALLBACK_STEPS: dict[str, list[str]] = {
    "sleep_support": [
        "Would it feel manageable to set a 10-minute worry window earlier this evening, so bedtime can be for rest instead of problem-solving?",
        "If your mind spins at night, would a one-line 'parking note' on paper help you tell your brain you won't lose the thought?",
    ],
    "conflict_resolution": [
        "Would it feel safer to delay the conversation until things are calmer and start with one 'I felt...' sentence instead of solving everything tonight?",
        "If home feels tense, could you create a short reset window first (walk, shower, or a quiet room) before talking?",
    ],
    "decision_support": [
        "Would it help to pick just one decision criterion for tonight, instead of trying to solve every tradeoff at once?",
        "Could you run one small reversible test this week, then decide with more information?",
    ],
    "job_search_support": [
        "Would it help to pick one application bottleneck tonight (resume bullet, role targeting, or outreach) and work only on that?",
        "Could you send one low-pressure feedback message to someone you trust, just to unblock your next step?",
    ],
    "academic_overload": [
        "Would it feel lighter to choose one 20-minute starter task and ignore the rest until that timer ends?",
        "Could you write a tiny 'must-do today' list with only one non-negotiable item?",
    ],
    "safety_escalation": [
        "Please text or call one trusted person right now and let them stay with you while this wave passes.",
        "If you might act on harmful thoughts, contact local emergency or crisis resources immediately.",
    ],
    "general_support": [
        "Would it help to choose one small step for tonight that lowers pressure, not perfection?",
        "Could we pick the easiest next 10-minute action together?",
    ],
}

FAILED_COPING_FOLLOWUP: dict[str, dict[str, Any]] = {
    "journaling": {
        "mechanism": (
            "Sometimes journaling does not bring relief because the page turns into more nighttime problem-solving instead of release."
        ),
        "questions": (
            "When you wrote last night, did your mind feel clearer, or did the thoughts multiply?",
            "As you were journaling, did it feel like relief or like pressure to solve everything before sleep?",
            "What happened after journaling, did your body settle at all, or stay activated?",
        ),
    },
    "breathing": {
        "mechanism": (
            "Sometimes breathing helps for a moment, but the mind jumps back into threat-scanning once the exercise ends."
        ),
        "questions": (
            "When breathing helped a bit, what pulled you back into worry right after?",
            "Did the breathing calm your body but not the future-related thoughts?",
            "Was the hard part the breathing itself, or what your mind did immediately after?",
        ),
    },
    "worry_scheduling": {
        "mechanism": (
            "Sometimes worry-time fails when the mind does not trust that concerns will be handled later, so it keeps them active at night."
        ),
        "questions": (
            "When you tried worry-time, did any part of you still feel it was unsafe to postpone the worries?",
            "Did writing worries earlier make bedtime lighter, or did the same worries return with urgency?",
            "What did your mind seem afraid would happen if it stopped worrying at night?",
        ),
    },
    "notepad_capture": {
        "mechanism": (
            "Sometimes note-taking captures thoughts but does not reduce the emotional load under those thoughts."
        ),
        "questions": (
            "When you jotted thoughts down, did your mind believe they were safely held, or keep rehearsing them anyway?",
            "Did the notes reduce mental noise, or leave the same fear in your body?",
            "What part felt unresolved even after writing things down?",
        ),
    },
    "sleep_hygiene": {
        "mechanism": (
            "Sometimes sleep routines are solid, but anxiety still stays active because the core fear has not been named yet."
        ),
        "questions": (
            "Even with good sleep habits, what fear seems to wake up at night?",
            "What does your nighttime mind keep trying to protect you from?",
            "When sleep does not come, what is the first future-thought your mind grabs?",
        ),
    },
    "default": {
        "mechanism": (
            "Sometimes a coping tool fails not because you did it wrong, but because the underlying fear is still running the system."
        ),
        "questions": (
            "What felt missing in that method when you tried it?",
            "What did your mind keep doing even while you were trying to help yourself?",
            "What part of the worry feels untouched so far?",
        ),
    },
}

STAGE_GOALS: dict[str, str] = {
    "rapport_and_safety": "Build emotional safety and trust before problem-solving.",
    "problem_exploration": "Clarify the concrete trigger, timing, and impact.",
    "emotion_identification": "Name the emotional layers underneath the surface issue.",
    "root_cause_exploration": "Explore deeper meaning, fear, and recurring pattern.",
    "goal_setting": "Align on one realistic and measurable near-term goal.",
    "intervention": "Offer one gentle intervention matched to the identified mechanism.",
    "session_closing": "Summarize and close with one collaborative next step.",
    "follow_up": "Review what happened since last step and adjust the plan.",
}

DEEP_DISCLOSURE_TOKENS = (
    "fear",
    "afraid",
    "scared",
    "hurt",
    "betray",
    "betrayed",
    "unsafe",
    "not safe",
    "safe at home",
    "not respected",
    "disrespected",
    "rejected",
    "abandoned",
    "ashamed",
    "guilty",
    "helpless",
    "worthless",
    "tense",
    "panic",
)

ACTION_REQUEST_TOKENS = (
    "what should i do",
    "what do i do",
    "what can i do",
    "how should i",
    "how do i",
    "any advice",
    "next step",
    "what's the best way",
    "怎么办",
    "怎么做",
    "我该怎么办",
    "该怎么做",
    "应该怎么做",
)

POSITIVE_SIGNAL_TOKENS = (
    "happy",
    "happier",
    "better",
    "calm",
    "relieved",
    "good now",
    "feel good",
    "feeling good",
    "great",
    "grateful",
    "hopeful",
)

POSITIVE_EMOTION_LABELS = {
    "happy",
    "happier",
    "calm",
    "relieved",
    "hopeful",
    "grateful",
    "content",
    "peaceful",
}

MEMORY_RECALL_CUES = (
    "do you remember",
    "remember me",
    "remind me",
    "do you remind",
    "还记得",
    "记得我",
    "记得吗",
    "提醒我",
)

MEMORY_DISCLAIMER_PATTERNS = (
    "i don't have the ability to remember",
    "i do not have the ability to remember",
    "i can't remember past conversations",
    "i cannot remember past conversations",
    "i don't remember past conversations",
    "i do not remember past conversations",
    "i don't have memory of past conversations",
    "i do not have memory of past conversations",
)


class MindBridgePipeline:
    def __init__(
        self,
        settings: Settings,
        run_config: PipelineRunConfig | None = None,
        profile_id: str = "default",
    ) -> None:
        self.settings = settings
        self.run_config = run_config or get_pipeline_run_config("full")
        self.profile_id = (profile_id or "default").strip() or "default"
        self.agents = build_agents(settings)
        self.retriever = (
            SupportKnowledgeRetriever(settings.support_kb_path)
            if self.run_config.use_retrieval
            else None
        )
        self._runtime_session_id = self._new_session_id()
        self._history: list[dict[str, Any]] = []
        self.user_state: dict[str, Any] = {}
        self._load_persistent_memory()

    def _new_session_id(self) -> str:
        return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")

    @property
    def runtime_session_id(self) -> str:
        return self._runtime_session_id

    @property
    def memory_store_path(self) -> Path:
        return Path(self.settings.persistent_memory_path)

    def _load_memory_store(self) -> dict[str, Any]:
        if not self.settings.persistent_memory_enabled:
            return {}
        path = self.memory_store_path
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    def _write_memory_store(self, payload: dict[str, Any]) -> None:
        if not self.settings.persistent_memory_enabled:
            return
        path = self.memory_store_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def _load_persistent_memory(self) -> None:
        if not self.settings.persistent_memory_enabled:
            return
        store = self._load_memory_store()
        profile_payload = store.get(self.profile_id, {})
        if not isinstance(profile_payload, dict):
            return

        history = profile_payload.get("history", [])
        user_state = profile_payload.get("user_state", {})
        if isinstance(history, list):
            cleaned_history = [item for item in history if isinstance(item, dict)]
            limit = self.settings.persistent_history_limit
            self._history = cleaned_history[-limit:] if limit > 0 else cleaned_history
        if isinstance(user_state, dict):
            self.user_state = dict(user_state)
        active_session_id = str(profile_payload.get("active_session_id", "")).strip()
        if active_session_id:
            self._runtime_session_id = active_session_id
        elif self._history:
            self._runtime_session_id = self._session_key_for_turn(self._history[-1])

    def _save_persistent_memory(self) -> None:
        if not self.settings.persistent_memory_enabled:
            return
        store = self._load_memory_store()
        store[self.profile_id] = {
            "profile_id": self.profile_id,
            "active_session_id": self._runtime_session_id,
            "updated_at": datetime.now(UTC).isoformat(),
            "turns": len(self._history),
            "user_state": self.user_state,
            "history": self._history,
        }
        self._write_memory_store(store)

    def clear_persistent_memory(self) -> None:
        self._history = []
        self.user_state = {}
        self._runtime_session_id = self._new_session_id()
        if not self.settings.persistent_memory_enabled:
            return
        store = self._load_memory_store()
        if self.profile_id in store:
            del store[self.profile_id]
            self._write_memory_store(store)

    def reset_runtime_memory(self) -> None:
        self._history = []
        self.user_state = {}
        self._runtime_session_id = self._new_session_id()

    def start_new_session(self) -> str:
        self._runtime_session_id = self._new_session_id()
        self._save_persistent_memory()
        return self._runtime_session_id

    def get_chat_log_snapshot(self, limit: int = 20) -> list[dict[str, Any]]:
        snapshot = self._history[-limit:] if limit > 0 else list(self._history)
        rows: list[dict[str, Any]] = []
        for turn in snapshot:
            rows.append(
                {
                    "user": str(turn.get("user_input", "")).strip(),
                    "assistant": str(turn.get("final_response", "")).strip(),
                    "route": str(turn.get("strategy_route", "general_support")).strip()
                    or "general_support",
                    "risk": str(turn.get("risk_level", "unknown")).strip() or "unknown",
                }
            )
        return rows

    def _session_key_for_turn(self, turn: dict[str, Any]) -> str:
        raw_key = str(turn.get("session_id", "")).strip()
        return raw_key or "legacy"

    def get_session_history_snapshot(
        self,
        limit_sessions: int = 20,
        exclude_session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        by_session: dict[str, dict[str, Any]] = {}
        for turn in self._history:
            if not isinstance(turn, dict):
                continue
            session_id = self._session_key_for_turn(turn)
            if exclude_session_id and session_id == exclude_session_id:
                continue
            if session_id not in by_session:
                summary = {
                    "session_id": session_id,
                    "title": str(turn.get("user_input", "")).strip() or "(empty message)",
                    "user": str(turn.get("user_input", "")).strip(),
                    "assistant": str(turn.get("final_response", "")).strip(),
                    "route": str(turn.get("strategy_route", "general_support")).strip()
                    or "general_support",
                    "risk": str(turn.get("risk_level", "unknown")).strip() or "unknown",
                    "started_at": str(turn.get("created_at", "")).strip(),
                    "turn_count": 0,
                }
                by_session[session_id] = summary
                summaries.append(summary)
            by_session[session_id]["turn_count"] = (
                int(by_session[session_id].get("turn_count", 0)) + 1
            )
        if limit_sessions > 0:
            return summaries[-limit_sessions:]
        return summaries

    def get_session_turns_snapshot(
        self,
        session_id: str,
        limit_turns: int = 50,
    ) -> list[dict[str, Any]]:
        wanted = (session_id or "").strip()
        if not wanted:
            return []
        rows: list[dict[str, Any]] = []
        for turn in self._history:
            if not isinstance(turn, dict):
                continue
            if self._session_key_for_turn(turn) != wanted:
                continue
            rows.append(
                {
                    "user": str(turn.get("user_input", "")).strip(),
                    "assistant": str(turn.get("final_response", "")).strip(),
                    "route": str(turn.get("strategy_route", "general_support")).strip()
                    or "general_support",
                    "risk": str(turn.get("risk_level", "unknown")).strip() or "unknown",
                }
            )
        if limit_turns > 0:
            return rows[-limit_turns:]
        return rows

    def get_active_session_turns_snapshot(self, limit_turns: int = 50) -> list[dict[str, Any]]:
        return self.get_session_turns_snapshot(self._runtime_session_id, limit_turns=limit_turns)

    def _skipped_output(self, reason: str, **payload: object) -> dict[str, object]:
        result = {"_skipped": True, "reason": reason}
        result.update(payload)
        return result

    def _memory_context(self) -> list[dict[str, Any]]:
        if self.settings.memory_window == 0:
            return []
        return self._history[-self.settings.memory_window :]

    def _keyword_safety_scan(self, user_input: str) -> dict[str, Any]:
        lowered = user_input.lower()
        hits = [pattern for pattern in HIGH_RISK_PATTERNS if pattern in lowered]
        if not hits:
            return {
                "risk_detected": False,
                "risk_level": "low",
                "flags": [],
                "constraints": [],
                "required_actions": [],
                "keyword_hits": [],
            }
        return {
            "risk_detected": True,
            "risk_level": "high",
            "flags": ["keyword_high_risk_signal"],
            "constraints": [
                "Avoid unsafe or dismissive advice.",
                "Prioritize immediate safety and trusted support contact.",
            ],
            "required_actions": [
                "Encourage immediate outreach to trusted people.",
                "Encourage local emergency/crisis resources if danger feels near.",
            ],
            "keyword_hits": hits,
        }

    def _risk_rank(self, level: str) -> int:
        return {"low": 1, "medium": 2, "high": 3}.get(level, 0)

    def _merge_safety_outputs(
        self,
        agent_output: dict[str, Any],
        keyword_scan: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        merged = dict(agent_output)
        agent_level = str(agent_output.get("risk_level", "low")).lower()
        keyword_level = str(keyword_scan.get("risk_level", "low")).lower()
        final_level = (
            agent_level
            if self._risk_rank(agent_level) >= self._risk_rank(keyword_level)
            else keyword_level
        )
        merged["risk_level"] = final_level
        merged["risk_detected"] = bool(agent_output.get("risk_detected")) or bool(
            keyword_scan.get("risk_detected")
        )

        merged_flags = list(agent_output.get("flags", []))
        for flag in keyword_scan.get("flags", []):
            if flag not in merged_flags:
                merged_flags.append(flag)
        merged["flags"] = merged_flags

        merged_constraints = list(agent_output.get("constraints", []))
        for item in keyword_scan.get("constraints", []):
            if item not in merged_constraints:
                merged_constraints.append(item)
        merged["constraints"] = merged_constraints

        merged_actions = list(agent_output.get("required_actions", []))
        for item in keyword_scan.get("required_actions", []):
            if item not in merged_actions:
                merged_actions.append(item)
        merged["required_actions"] = merged_actions

        trace = {
            "agent_risk_level": agent_level,
            "keyword_risk_level": keyword_level,
            "keyword_hits": keyword_scan.get("keyword_hits", []),
            "final_risk_level": final_level,
            "escalation_needed": self._risk_rank(final_level) >= 3,
            "reasoning_source": "agent_plus_keyword_scan",
        }
        return merged, trace

    def _build_strategy_route(
        self,
        user_input: str,
        analyzer: dict[str, Any],
        keyword_scan: dict[str, Any],
    ) -> dict[str, Any]:
        if str(keyword_scan.get("risk_level", "low")).lower() == "high":
            return {
                "scenario": "safety_escalation",
                "matched_keywords": list(keyword_scan.get("keyword_hits", [])),
                "recommended_focus": list(STRATEGY_PLAYBOOK[0]["focus"]),
                "route_reason": "high_risk_keyword_override",
                "avoid_generic_actions": list(GENERIC_ACTIONS_TO_AVOID),
            }

        routing_text = " ".join(
            [
                user_input.lower(),
                str(analyzer.get("topic", "")).lower(),
                str(analyzer.get("intent", "")).lower(),
                str(analyzer.get("notes", "")).lower(),
            ]
        )

        best: dict[str, Any] | None = None
        best_hits = 0
        best_priority = -1
        for entry in STRATEGY_PLAYBOOK:
            hits = [word for word in entry["keywords"] if word in routing_text]
            if not hits:
                continue
            priority = int(entry.get("priority", 0))
            if len(hits) > best_hits or (len(hits) == best_hits and priority > best_priority):
                best_hits = len(hits)
                best_priority = priority
                best = {
                    "scenario": entry["scenario"],
                    "matched_keywords": hits,
                    "recommended_focus": list(entry["focus"]),
                    "route_reason": "keyword_rule_match",
                }

        if not best:
            best = {
                "scenario": "general_support",
                "matched_keywords": [],
                "recommended_focus": [
                    "specific next-step planning",
                    "practical bottleneck diagnosis",
                    "one-step execution plan",
                ],
                "route_reason": "fallback_general_support",
            }

        best["avoid_generic_actions"] = list(GENERIC_ACTIONS_TO_AVOID)
        return best

    def _is_generic_suggestion(self, text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in GENERIC_ACTIONS_TO_AVOID)

    def _as_text_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        single = str(value).strip()
        return [single] if single else []

    def _has_failure_cue(self, text: str) -> bool:
        lowered = text.lower()
        return any(cue in lowered for cue in FAILURE_CUES)

    def _extract_methods_from_text(self, text: str) -> set[str]:
        lowered = text.lower()
        methods: set[str] = set()
        for method, keywords in METHOD_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                methods.add(method)
        return methods

    def _extract_methods_from_items(self, items: list[str]) -> set[str]:
        methods: set[str] = set()
        for item in items:
            methods.update(self._extract_methods_from_text(item))
        return methods

    def _method_label(self, method: str) -> str:
        return METHOD_LABELS.get(method, method.replace("_", " "))

    def _format_method_list(self, methods: set[str] | list[str]) -> str:
        labels = [self._method_label(method) for method in sorted(set(methods))]
        if not labels:
            return ""
        if len(labels) == 1:
            return labels[0]
        if len(labels) == 2:
            return f"{labels[0]} and {labels[1]}"
        return f"{', '.join(labels[:-1])}, and {labels[-1]}"

    def _humanize_reflection_text(self, text: str) -> str:
        cleaned = text.strip()
        replacements = (
            ("The user is ", "You are "),
            ("the user is ", "you are "),
            ("User is ", "You are "),
            ("The user needs ", "You need "),
            ("the user needs ", "you need "),
            ("User needs ", "You need "),
            ("The user feels ", "You feel "),
            ("the user feels ", "you feel "),
            ("User feels ", "You feel "),
            ("The user has ", "You have "),
            ("the user has ", "you have "),
            ("User has ", "You have "),
        )
        for source, target in replacements:
            cleaned = cleaned.replace(source, target)
        return cleaned

    def _soften_reflection_text(self, text: str) -> str:
        cleaned = text.strip()
        replacements = (
            ("You need to ", "It may help to "),
            ("you need to ", "it may help to "),
            ("You need ", "Maybe what would help right now is "),
            ("you need ", "maybe what would help right now is "),
            ("You are stuck in ", "It can feel like being stuck in "),
            ("you are stuck in ", "it can feel like being stuck in "),
        )
        for source, target in replacements:
            cleaned = cleaned.replace(source, target)
        return cleaned

    def _single_sentence(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        for marker in (". ", "! ", "? ", "\n"):
            if marker in cleaned:
                return cleaned.split(marker, 1)[0].strip() + (
                    "." if marker in {". ", "\n"} and not cleaned.endswith(".") else ""
                )
        return cleaned

    def _content_tokens(self, text: str) -> set[str]:
        tokens = {
            word.strip(".,!?;:'\"()[]{}").lower()
            for word in text.split()
            if word.strip(".,!?;:'\"()[]{}")
        }
        stop = {
            "the",
            "a",
            "an",
            "to",
            "of",
            "and",
            "or",
            "is",
            "are",
            "it",
            "that",
            "this",
            "you",
            "your",
            "for",
            "with",
            "in",
            "on",
            "at",
            "be",
            "as",
            "now",
            "really",
            "very",
            "feel",
            "feels",
            "feeling",
            "sounds",
            "like",
        }
        return {token for token in tokens if token and token not in stop}

    def _is_redundant_sentence(self, first: str, second: str) -> bool:
        first_tokens = self._content_tokens(first)
        second_tokens = self._content_tokens(second)
        if not first_tokens or not second_tokens:
            return False
        overlap = first_tokens & second_tokens
        overlap_ratio = len(overlap) / max(1, min(len(first_tokens), len(second_tokens)))
        return overlap_ratio >= 0.5 or len(overlap) >= 3

    def _is_memory_recall_question(self, user_input: str) -> bool:
        lowered = user_input.lower()
        return any(cue in lowered for cue in MEMORY_RECALL_CUES)

    def _contains_memory_reference(self, text: str) -> bool:
        lowered = (text or "").lower().strip()
        if not lowered:
            return False
        memory_tokens = (
            "remember",
            "remind",
            "memory",
            "earlier",
            "previous",
            "profile",
            "conversation",
            "chat",
            "thread",
        )
        return any(token in lowered for token in memory_tokens)

    def _memory_recall_line(self, state: DialogueState) -> str:
        if not state.memory_context:
            return ""
        last = state.memory_context[-1]
        topic = str(last.get("topic", "")).strip()
        user_text = str(last.get("user_input", "")).strip()
        if topic:
            return f"I do remember earlier in this profile: we were discussing {topic}."
        if user_text:
            snippet = user_text[:80].strip()
            if len(user_text) > 80:
                snippet += "..."
            return f"I do remember earlier in this profile: you shared \"{snippet}\"."
        return "I do remember our earlier conversation in this profile."

    def _memory_recall_relationship_response(self, state: DialogueState, variant: int) -> str:
        asked_recall = self._is_memory_recall_question(state.user_input)
        if not asked_recall:
            return ""

        parts: list[str] = []
        recall_line = self._memory_recall_line(state)
        if recall_line:
            parts.append(recall_line)
        else:
            parts.append(
                "I may not have enough earlier context loaded in this profile yet, and you can re-anchor me with one detail."
            )

        relationship_lines = (
            "Part of this question can be about whether this space is still continuous and whether I am still with you.",
            "It can make sense to check whether I still hold our thread and whether this still feels like a connected space.",
            "Questions like this are often about continuity and safety in the relationship, not only about memory facts.",
        )
        followups = (
            "What is it like for you to ask me that right now?",
            "As you ask that, what feels most important for me to hold with you?",
            "When you ask this, what are you hoping I will remember most?",
        )
        parts.append(relationship_lines[variant % len(relationship_lines)])
        parts.append(followups[variant % len(followups)])

        return "\n\n".join(part for part in parts if part).strip()

    def _remove_memory_disclaimer_paragraphs(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        kept: list[str] = []
        for part in paragraphs:
            lowered = part.lower()
            if any(pattern in lowered for pattern in MEMORY_DISCLAIMER_PATTERNS):
                continue
            kept.append(part)
        return kept

    def _apply_memory_recall_guard(self, state: DialogueState, response: str) -> str:
        text = (response or "").strip()
        if not text:
            return text
        lowered = text.lower()
        has_disclaimer = any(pattern in lowered for pattern in MEMORY_DISCLAIMER_PATTERNS)
        asked_recall = self._is_memory_recall_question(state.user_input)
        if not has_disclaimer and not asked_recall:
            return text

        cleaned_parts = self._remove_memory_disclaimer_paragraphs(text)

        if state.memory_context:
            recall_line = self._memory_recall_line(state)
            if recall_line and asked_recall:
                if cleaned_parts:
                    return "\n\n".join([recall_line] + cleaned_parts).strip()
                return recall_line
            if cleaned_parts:
                return "\n\n".join(cleaned_parts).strip()
            return recall_line or text

        fallback = (
            "I may not have enough earlier context loaded yet in this profile. "
            "If you share one detail from last time, I can continue from there."
        )
        if cleaned_parts:
            return "\n\n".join([fallback] + cleaned_parts).strip()
        return fallback

    def _has_deeper_disclosure_signal(self, user_input: str, state: DialogueState) -> bool:
        lowered = user_input.lower()
        if any(token in lowered for token in DEEP_DISCLOSURE_TOKENS):
            return True
        emotion = str(state.analyzer.get("emotion", "")).lower()
        if emotion in {"fear", "afraid", "scared", "angry", "frustrated", "sad", "distressed"}:
            return True
        notes = str(state.analyzer.get("notes", "")).lower()
        if any(token in notes for token in ("fear", "hurt", "unsafe", "worth", "rejected")):
            return True
        return False

    def _has_action_request_signal(self, user_input: str, state: DialogueState) -> bool:
        lowered = user_input.lower()
        if any(token in lowered for token in ACTION_REQUEST_TOKENS):
            return True
        intent = str(state.analyzer.get("intent", "")).lower()
        if any(token in intent for token in ("advice", "how_to", "next_step", "plan")):
            return True
        return False

    def _is_introduction_turn(self, user_input: str) -> bool:
        lowered = user_input.lower().strip()
        words = [w for w in lowered.replace(",", " ").split() if w]
        intro_patterns = (
            "my name is",
            "i am ",
            "i'm ",
            "call me ",
        )
        if any(pattern in lowered for pattern in intro_patterns):
            return True
        greeting_tokens = {"hi", "hello", "hey"}
        if words and words[0] in greeting_tokens and len(words) <= 8:
            return True
        return False

    def _is_positive_turn(self, user_input: str, state: DialogueState) -> bool:
        lowered = user_input.lower()
        emotion = str(state.analyzer.get("emotion", "")).strip().lower()
        if any(flag in lowered for flag in ("not happy", "not better", "not good", "still bad")):
            return False
        has_positive_signal = any(token in lowered for token in POSITIVE_SIGNAL_TOKENS)
        if not has_positive_signal and emotion not in POSITIVE_EMOTION_LABELS:
            return False
        distress_tokens = (
            "sad",
            "anxious",
            "anxiety",
            "depressed",
            "hopeless",
            "panic",
            "scared",
            "afraid",
            "hurt",
            "angry",
            "frustrated",
            "overwhelmed",
            "unsafe",
            "problem",
            "issue",
        )
        if any(token in lowered for token in distress_tokens):
            return False
        if emotion in {"sad", "anxious", "depressed", "frustrated", "distressed"}:
            return False
        return True

    def _is_brief_positive_checkin(self, user_input: str, state: DialogueState) -> bool:
        if not self._is_positive_turn(user_input, state):
            return False
        words = [w for w in user_input.lower().replace(",", " ").split() if w]
        return len(words) <= 8

    def _last_user_input(self, state: DialogueState) -> str:
        if not state.memory_context:
            return ""
        return str(state.memory_context[-1].get("user_input", "")).strip()

    def _select_positive_opening(
        self,
        empathy: str,
        reflection: str,
        fallback: str,
    ) -> str:
        blocked_tokens = (
            "remember",
            "remind",
            "profile",
            "earlier",
            "previous",
            "conversation",
            "thread",
            "note one",
            "momentum",
            "what helped",
            "repeat tonight",
        )
        candidates = [empathy.strip(), reflection.strip(), fallback.strip()]
        for candidate in candidates:
            if not candidate:
                continue
            lowered = candidate.lower()
            if any(token in lowered for token in blocked_tokens):
                continue
            return candidate
        return fallback

    def _strip_unprompted_memory_lead(self, state: DialogueState, response: str) -> str:
        text = (response or "").strip()
        if not text:
            return text
        if self._is_memory_recall_question(state.user_input):
            return text

        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        if not paragraphs:
            return text

        first = paragraphs[0]
        first_lower = first.lower()
        memory_lead_prefixes = (
            "i remember",
            "i do remember",
            "i still remember",
            "i can remember",
        )

        if any(first_lower.startswith(prefix) for prefix in memory_lead_prefixes):
            if len(paragraphs) > 1:
                return "\n\n".join(paragraphs[1:]).strip()
            return "I hear you."

        if self._is_positive_turn(state.user_input, state) and self._contains_memory_reference(first):
            clean_first = "I am glad to hear this moment feels a little lighter."
            if len(paragraphs) > 1:
                return "\n\n".join([clean_first] + paragraphs[1:]).strip()
            return clean_first

        return text

    def _action_step_allowed(self, user_input: str, state: DialogueState) -> bool:
        if self._is_positive_turn(user_input, state) and not self._has_action_request_signal(user_input, state):
            return False
        if self._has_action_request_signal(user_input, state):
            return True
        return self._session_turn(state) >= 4

    def _combine_empathy_text(
        self,
        acknowledgement: str,
        validation: str,
    ) -> str:
        ack = self._single_sentence(acknowledgement)
        val = self._single_sentence(validation)
        if not ack and not val:
            return ""
        if not ack:
            return val
        if not val:
            return ack

        ack_lower = ack.lower()
        val_lower = val.lower()
        repetitive_prefixes = (
            "it sounds",
            "i understand",
            "i'm sorry",
            "that sounds",
        )
        if any(ack_lower.startswith(prefix) and val_lower.startswith(prefix) for prefix in repetitive_prefixes):
            return ack
        if val_lower in ack_lower or self._is_redundant_sentence(ack, val):
            return ack
        return f"{ack} {val}"

    def _de_template_empathy(self, text: str, variant: int) -> str:
        cleaned = text.strip()
        lowered = cleaned.lower()
        if lowered.startswith("it sounds like") and variant == 1:
            return cleaned.replace("It sounds like", "That can feel like", 1)
        if lowered.startswith("it sounds") and variant == 2:
            return cleaned.replace("It sounds", "That sounds", 1)
        if lowered.startswith("i understand") and variant != 0:
            return cleaned.replace("I understand", "It makes sense", 1)
        if lowered.startswith("i'm sorry to hear") and variant == 2:
            return cleaned.replace("I'm sorry to hear that", "That's a lot to carry", 1)
        return cleaned

    def _validate_failed_effort(self, memory_ack: str) -> str:
        if memory_ack and "did not help enough" in memory_ack.lower():
            return (
                "You really gave that a try, and it is discouraging when something that should help does not bring relief."
            )
        return (
            "You put in real effort, and it makes sense to feel frustrated when it still feels this hard."
        )

    def _failed_method_for_turn(self, state: DialogueState) -> str:
        methods = sorted(self._extract_methods_from_text(state.user_input))
        if methods:
            return methods[0]
        last_method = str(state.user_state.get("last_strategy_method", "")).strip()
        if last_method:
            return last_method
        failed_methods = self._as_text_list(state.strategy.get("failed_methods_considered", []))
        return failed_methods[0] if failed_methods else "default"

    def _failed_coping_followup_parts(
        self,
        state: DialogueState,
        memory_ack: str,
        variant: int,
    ) -> list[str]:
        method = self._failed_method_for_turn(state)
        pattern = FAILED_COPING_FOLLOWUP.get(method, FAILED_COPING_FOLLOWUP["default"])

        mechanism = self._single_sentence(
            str(state.strategy.get("failure_mechanism_hypothesis", "")).strip()
        )
        if not mechanism:
            mechanism = pattern["mechanism"]

        question = str(state.strategy.get("exploration_question", "")).strip()
        if not question:
            question = pattern["questions"][variant % len(pattern["questions"])]
        if not question.endswith("?"):
            question = question.rstrip(".") + "?"

        return [
            self._validate_failed_effort(memory_ack),
            mechanism,
            "If we understand this part first, the next step usually becomes much clearer.",
            question,
        ]

    def _route_fallback_suggestions(
        self,
        scenario: str,
        recommended_focus: list[str],
    ) -> list[str]:
        fallback = list(ROUTE_FALLBACK_STEPS.get(scenario, []))
        if fallback:
            return fallback
        if recommended_focus:
            return [
                f"Would it help to focus on {recommended_focus[0]} first, and leave the rest for later?"
            ]
        return list(ROUTE_FALLBACK_STEPS["general_support"])

    def _gentle_action_invite(
        self,
        next_step: str,
        alternative: str,
        variant: int,
    ) -> str:
        step = next_step.strip()
        alt = alternative.strip()
        if not step:
            return ""
        step_is_question = step.endswith("?")
        alt_is_question = alt.endswith("?")
        if variant == 0:
            if step_is_question:
                if alt:
                    return f"{step} If that feels off, {alt}"
                return step
            if alt:
                bridge = "Or, if that feels off,"
                return f"We can keep this small: {step} {bridge} {alt}"
            return f"We can keep this small: {step}"
        if variant == 1:
            if alt:
                if step_is_question and alt_is_question:
                    return f"Would this fit better for tonight: {step} Or maybe this: {alt}"
                return f"Would you rather try this first: {step} Or does this feel easier: {alt}"
            return f"Would this feel manageable for tonight: {step}"
        if alt:
            if step_is_question:
                return f"{step} If that doesn't fit, we can switch to {alt}"
            return f"One gentle option is {step} If that doesn't fit, we can switch to {alt}"
        if step_is_question:
            return step
        return f"One gentle option is {step}"

    def _collaborative_close(self, state: DialogueState, variant: int) -> str:
        if self._is_positive_turn(state.user_input, state):
            prompts = (
                "We do not need to turn this into a task right now. We can stay with this lighter moment for a breath.",
                "No rush to solve the rest tonight; this small shift already matters.",
                "If you want, we can simply notice what feels different right now and keep it gentle.",
            )
            return prompts[variant % len(prompts)]
        lowered = state.user_input.lower()
        if "sleep" in lowered or "night" in lowered:
            prompts = (
                "When you wake up at night, what is the first thought that pulls you in?",
                "What does your mind start saying in those first few minutes at night?",
                "What would help your body feel 5% safer at bedtime tonight?",
            )
        elif "roommate" in lowered or "conflict" in lowered or "argue" in lowered:
            prompts = (
                "What feels hardest about going back home right now?",
                "Which part of that conflict hurt the most?",
                "What do you most need before opening that door tonight?",
            )
        else:
            prompts = (
                "What feels heaviest right now?",
                "What would make tonight even slightly easier?",
                "If we keep this very small, what feels doable?",
            )
        return prompts[variant % len(prompts)]

    def _session_turn(self, state: DialogueState) -> int:
        return len(state.memory_context) + 1

    def _determine_session_stage(self, user_input: str, state: DialogueState) -> tuple[str, str]:
        turn = self._session_turn(state)
        lowered = user_input.lower()
        risk_level = str(state.safety.get("risk_level", "low")).lower()
        deeper_signal = self._has_deeper_disclosure_signal(user_input, state)
        action_allowed = self._action_step_allowed(user_input, state)
        positive_turn = self._is_positive_turn(user_input, state)
        intro_turn = self._is_introduction_turn(user_input)
        has_last_strategy = bool(str(state.user_state.get("last_strategy", "")).strip())
        has_followup_cue = any(
            cue in lowered
            for cue in (
                "tried",
                "helped",
                "didn't help",
                "did not help",
                "worked",
                "not working",
            )
        )

        if risk_level == "high":
            stage = "rapport_and_safety"
        elif self._has_failure_cue(user_input):
            stage = "follow_up"
        elif has_last_strategy and has_followup_cue:
            stage = "follow_up"
        elif intro_turn or positive_turn:
            stage = "rapport_and_safety"
        elif turn <= 1:
            stage = "rapport_and_safety"
        elif turn == 2:
            stage = "problem_exploration"
        elif turn == 3:
            stage = "emotion_identification"
        elif turn == 4 and action_allowed:
            stage = "goal_setting"
        elif turn == 4:
            stage = "root_cause_exploration"
        elif turn >= 5 and not action_allowed:
            stage = "root_cause_exploration"
        elif turn >= 5 and not deeper_signal:
            stage = "root_cause_exploration"
        elif turn == 5:
            stage = "intervention"
        elif turn >= 9:
            stage = "session_closing"
        else:
            stage = "intervention"

        return stage, STAGE_GOALS.get(stage, "")

    def _stage_prompt_question(self, state: DialogueState, stage: str, variant: int) -> str:
        if self._is_introduction_turn(state.user_input):
            prompts = (
                "Nice to meet you. What would feel most helpful to talk about today?",
                "Glad you are here. What is one thing you want support with right now?",
                "Thanks for introducing yourself. What brought you in today?",
            )
            return prompts[variant % len(prompts)]

        if self._is_positive_turn(state.user_input, state):
            prompts = (
                "As you say that, what feels most different inside right now compared with earlier?",
                "If we stay with this lighter moment for a second, where do you notice it first?",
                "What feels a little easier in your body right now?",
            )
            return prompts[variant % len(prompts)]

        if stage == "rapport_and_safety":
            prompts = (
                "What feels most intense right now?",
                "Where do you feel this tension most in your body?",
                "What part feels hardest to carry at this moment?",
            )
            return prompts[variant % len(prompts)]
        if stage == "problem_exploration":
            return self._problem_exploration_question(state, variant)
        if stage == "emotion_identification":
            prompts = (
                "If you name the core feeling underneath this, what word comes up first?",
                "Under the stress, what feeling is most present for you right now?",
                "When you picture going back, what emotion hits first?",
            )
            return prompts[variant % len(prompts)]
        if stage == "root_cause_exploration":
            return self._root_cause_probe(state, variant)
        if stage == "goal_setting":
            prompts = (
                "If this week felt 10 percent better, what would be different first?",
                "What would count as a small win for tonight?",
                "What is one concrete change you want to notice in the next two days?",
            )
            return prompts[variant % len(prompts)]
        if stage == "intervention":
            prompts = (
                "Would you like one gentle step for tonight, then we review tomorrow?",
                "Do you want to pick one tiny action now and keep the rest for later?",
                "Would it help if we keep this to one step that feels doable in under 10 minutes?",
            )
            return prompts[variant % len(prompts)]
        if stage == "session_closing":
            prompts = (
                "Before we pause, what is the one line you want to remember from today?",
                "What should we check first when we continue next time?",
                "What would help you feel supported between now and our next check-in?",
            )
            return prompts[variant % len(prompts)]
        return self._collaborative_close(state, variant)

    def _positive_followup_question(self, state: DialogueState, variant: int) -> str:
        previous_user_input = self._last_user_input(state)
        if previous_user_input and self._is_memory_recall_question(previous_user_input):
            prompts = (
                "Does this lighter feeling seem connected to feeling remembered here?",
                "As you feel a bit lighter now, what changed after that memory-check moment?",
                "What is different inside right now compared with just before you asked about memory?",
            )
            return prompts[variant % len(prompts)]

        if self._is_brief_positive_checkin(state.user_input, state):
            prompts = (
                "If we stay with this for one breath, what feels different right now?",
                "As you say that, where do you notice the relief first, body or mind?",
                "What part of this lighter moment feels most real to you right now?",
            )
            return prompts[variant % len(prompts)]

        return self._stage_prompt_question(state, "rapport_and_safety", variant)

    def _stage_transition_line(self, stage: str) -> str:
        transitions = {
            "rapport_and_safety": "We can move at your pace here.",
            "problem_exploration": "Let's slow this down and make the picture clearer first.",
            "emotion_identification": "Naming the feeling often makes it less overwhelming.",
            "root_cause_exploration": "If we understand the pattern underneath, the next step usually becomes clearer.",
            "goal_setting": "We can turn this into one concrete target, not a huge task list.",
            "intervention": "Let's keep it practical and gentle for tonight.",
            "session_closing": "Let's land this with one clear takeaway.",
            "follow_up": "Let's first understand what happened with the last step before changing tools.",
        }
        return transitions.get(stage, "")

    def _emotion_identification_line(self, state: DialogueState) -> str:
        if self._is_positive_turn(state.user_input, state):
            return "I am glad to hear there is some relief in this moment."
        emotion = str(state.analyzer.get("emotion", "")).strip().lower()
        if not emotion:
            return "There may be more than one feeling layered together here."
        if emotion in {"anxious", "anxiety"}:
            return "This feels like anxiety mixed with pressure, not personal failure."
        if emotion in {"sad", "depressed", "low"}:
            return "This seems like sadness and exhaustion showing up together, not weakness."
        if emotion in {"angry", "frustrated"}:
            return "I hear frustration, and maybe some hurt underneath."
        if emotion in {"fear", "scared", "afraid"}:
            return "Fear may be taking up a lot of space right now."
        if emotion in {"distressed", "distress"}:
            return "I can hear a lot of distress in this."
        if emotion in POSITIVE_EMOTION_LABELS:
            return f"It sounds like {emotion} is more present right now."
        return f"It seems like {emotion} is very present right now."

    def _problem_exploration_question(self, state: DialogueState, variant: int) -> str:
        if self._is_positive_turn(state.user_input, state):
            prompts = (
                "What feels different in this moment compared with a little earlier?",
                "If we pause here, what part of you feels even 5% softer right now?",
                "What changed in your body or mind just before this lighter feeling showed up?",
            )
            return prompts[variant % len(prompts)]
        lowered = state.user_input.lower()
        if "sleep" in lowered or "night" in lowered:
            prompts = (
                "When this gets hardest at night, what is usually happening right before it starts?",
                "When does the worry usually spike most?",
                "What time window is usually the toughest for you?",
            )
        elif "roommate" in lowered or "conflict" in lowered:
            prompts = (
                "In that conflict, what moment felt the most painful for you?",
                "When things escalated, what did you need most but not get?",
                "What part of that interaction is still stuck in your mind?",
            )
        elif "job" in lowered or "internship" in lowered or "career" in lowered:
            prompts = (
                "When this pressure rises, what part hits your confidence most?",
                "What part feels heaviest lately?",
                "In this job-search stress, what is the part your mind keeps returning to?",
            )
        else:
            prompts = (
                "When this feeling shows up, what usually triggers it most?",
                "What makes this feel hardest lately?",
                "If we zoom in, what part hurts most right now?",
            )
        return prompts[variant % len(prompts)]

    def _root_cause_probe(self, state: DialogueState, variant: int) -> str:
        lowered = state.user_input.lower()
        if "sleep" in lowered or "night" in lowered:
            prompts = (
                "What do those nighttime thoughts usually say about you or your future?",
                "What meaning does your mind attach to that uncertainty at night?",
                "What is the worst meaning your mind attaches to not sleeping well?",
            )
            return prompts[variant % len(prompts)]
        if "internship" in lowered or "job" in lowered or "career" in lowered:
            prompts = (
                "When applications do not work out, what story does your mind tell about your value?",
                "What does this experience seem to say about you, in your own words?",
                "What are you most afraid this says about you?",
            )
            return prompts[variant % len(prompts)]
        prompts = (
            "When this pattern repeats, what core fear do you notice underneath it?",
            "If we look below the surface, what feels most at stake for you?",
            "What meaning does your mind attach to this situation?",
        )
        return prompts[variant % len(prompts)]

    def _reframe_line(self, state: DialogueState, variant: int) -> str:
        scenario = str(state.strategy_route.get("scenario", "general_support"))
        if scenario == "job_search_support":
            reframes = (
                "Not getting a response can reflect fit and process noise, not your worth.",
                "A setback here is data about strategy, not a verdict on who you are.",
                "This can be painful and still not define your value.",
            )
        elif scenario == "sleep_support":
            reframes = (
                "Your mind may be trying to protect you by over-solving at night, even though it is backfiring.",
                "This looks less like lack of effort and more like a stress loop that can be interrupted.",
                "The struggle here may be about nervous-system overload, not personal weakness.",
            )
        else:
            reframes = (
                "What you are feeling makes sense in context, even if it feels overwhelming.",
                "This may be a protective pattern that got too loud, not something broken in you.",
                "We can treat this as a pattern to understand, not a flaw to judge.",
            )
        return reframes[variant % len(reframes)]

    def _goal_or_action_line(
        self,
        state: DialogueState,
        primary_step: str,
        alternative_step: str,
        variant: int,
    ) -> str:
        session_turn = self._session_turn(state)
        if session_turn <= 4:
            return "We do not need to fix everything right now; understanding the pattern is already a good step."
        action = self._gentle_action_invite(primary_step, alternative_step, variant).strip()
        if not action:
            return "When you are ready, we can pick one tiny step for tonight."
        return action

    def _encouraging_close_line(self, state: DialogueState, variant: int) -> str:
        prompts = (
            "You do not have to carry the whole thing at once.",
            "Thanks for staying with this; that matters.",
            "We can keep this gentle and adjust together as you notice more.",
        )
        return prompts[variant % len(prompts)]

    def _render_standard_conversational_response(
        self,
        state: DialogueState,
        sections: dict[str, str],
    ) -> str:
        variant = self._conversation_variant(state)
        empathy = self._single_sentence(sections.get("Empathy", "").strip())
        empathy = self._de_template_empathy(empathy, variant)
        reflection = self._single_sentence(sections.get("Reflection", "").strip())
        suggestions = self._as_text_list(state.strategy.get("suggestions", []))
        next_step = str(state.strategy.get("next_step", "")).strip()
        memory_ack = self._single_sentence(str(state.strategy.get("memory_ack", "")).strip())
        high_emotion_turn = self._is_high_emotion_turn(state)
        risk_level = str(state.safety.get("risk_level", "unknown")).lower()
        failure_turn = self._has_failure_cue(state.user_input)
        action_allowed = self._action_step_allowed(state.user_input, state)
        positive_turn = self._is_positive_turn(state.user_input, state)

        parts: list[str] = []
        opening_bits: list[str] = []
        if empathy:
            opening_bits.append(empathy)
        if (
            reflection
            and reflection.lower() not in empathy.lower()
            and not self._is_redundant_sentence(empathy, reflection)
        ):
            opening_bits.append(reflection)
        if opening_bits:
            parts.append(" ".join(opening_bits))

        if risk_level == "high":
            parts.append(
                "Your safety comes first right now. Please contact a trusted person immediately, and use local emergency or crisis resources now if you might act on harmful thoughts."
            )
            return "\n\n".join(part for part in parts if part).strip()

        recall_response = self._memory_recall_relationship_response(state, variant)
        if recall_response:
            if risk_level == "medium":
                return (
                    f"{recall_response}\n\n"
                    "If things feel heavier tonight, please stay connected with someone you trust rather than carrying this alone."
                )
            return recall_response

        if positive_turn and risk_level == "low" and not failure_turn:
            if self._is_brief_positive_checkin(state.user_input, state):
                lead = "I am glad to hear this moment feels a little lighter."
            else:
                lead = self._select_positive_opening(
                    empathy,
                    reflection,
                    "I am glad to hear this moment feels a little lighter.",
                )
            parts = [lead, self._positive_followup_question(state, variant)]
            if not self._is_brief_positive_checkin(state.user_input, state):
                parts.append(self._collaborative_close(state, variant))
            return "\n\n".join(part for part in parts if part).strip()

        if failure_turn:
            followup_parts = self._failed_coping_followup_parts(
                state,
                memory_ack,
                variant,
            )
            parts.extend(followup_parts[:2])
            if followup_parts:
                parts.append(followup_parts[-1])
            if risk_level == "medium":
                parts.append(
                    "If things feel heavier tonight, please stay connected with someone you trust rather than carrying this alone."
                )
            return "\n\n".join(part for part in parts if part).strip()

        primary_step = next_step or (suggestions[0] if suggestions else "")
        alternative_step = suggestions[1] if len(suggestions) > 1 else ""
        if primary_step and action_allowed:
            if high_emotion_turn:
                parts.append("We do not need to fix everything tonight; one small step is enough for now.")
            parts.append(self._gentle_action_invite(primary_step, alternative_step, variant))

        if risk_level == "medium":
            parts.append(
                "If things feel heavier tonight, please stay connected with someone you trust rather than carrying this alone."
            )
        elif state.safety.get("flags"):
            parts.append("If this gets heavier later, reach out early for extra support.")

        parts.append(self._collaborative_close(state, variant))
        return "\n\n".join(part for part in parts if part).strip()

    def _conversation_variant(self, state: DialogueState) -> int:
        seed = sum(ord(ch) for ch in state.user_input) + len(state.memory_context) * 17
        return seed % 3

    def _is_high_emotion_turn(self, state: DialogueState) -> bool:
        user_text = state.user_input.lower()
        emotion_text = str(state.analyzer.get("emotion", "")).lower()
        high_emotion_tokens = (
            "overwhelmed",
            "panic",
            "scared",
            "afraid",
            "exhausted",
            "hopeless",
            "anxious",
            "frustrated",
            "drained",
        )
        return (
            self._has_failure_cue(user_text)
            or any(token in user_text for token in high_emotion_tokens)
            or any(token in emotion_text for token in high_emotion_tokens)
        )

    def _failed_methods_for_current_turn(
        self,
        user_input: str,
        user_state: dict[str, Any],
        memory_context: list[dict[str, Any]],
    ) -> set[str]:
        failed_methods = set(self._as_text_list(user_state.get("failed_methods", [])))
        if not self._has_failure_cue(user_input):
            return failed_methods

        mentioned = self._extract_methods_from_text(user_input)
        if mentioned:
            failed_methods.update(mentioned)
            return failed_methods

        last_strategy_method = str(user_state.get("last_strategy_method", "")).strip()
        if last_strategy_method:
            failed_methods.add(last_strategy_method)
            return failed_methods

        if memory_context:
            last_turn = memory_context[-1]
            failed_methods.update(
                self._as_text_list(last_turn.get("strategy_methods", []))
            )
        return failed_methods

    def _normalize_strategy_output(
        self,
        strategy_output: dict[str, Any],
        strategy_route: dict[str, Any],
        user_input: str,
        user_state: dict[str, Any],
        memory_context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized = dict(strategy_output)
        scenario = str(strategy_route.get("scenario", "general_support"))
        normalized["scenario"] = scenario

        failed_methods = self._failed_methods_for_current_turn(
            user_input,
            user_state,
            memory_context,
        )
        normalized["failed_methods_considered"] = sorted(failed_methods)

        suggestions = self._as_text_list(strategy_output.get("suggestions", []))
        route_focus = self._as_text_list(strategy_route.get("recommended_focus", []))
        recommended_focus = list(route_focus)

        if failed_methods:
            suggestions = [
                item
                for item in suggestions
                if not (self._extract_methods_from_text(item) & failed_methods)
            ]
            recommended_focus = [
                item
                for item in recommended_focus
                if not (self._extract_methods_from_text(item) & failed_methods)
            ]

        generic_count = sum(1 for item in suggestions if self._is_generic_suggestion(item))
        force_rule_based = (
            not suggestions
            or scenario != str(strategy_output.get("scenario", scenario))
            or generic_count == len(suggestions)
        )

        if force_rule_based:
            if not recommended_focus:
                recommended_focus = [
                    "specific next-step planning",
                    "practical bottleneck diagnosis",
                    "one-step execution plan",
                ]
            fallback_suggestions = self._route_fallback_suggestions(
                scenario,
                recommended_focus,
            )
            normalized["suggestions"] = fallback_suggestions[:2]
            if fallback_suggestions:
                normalized["next_step"] = fallback_suggestions[0]
            normalized["problem_frame"] = str(
                strategy_output.get(
                    "problem_frame",
                    f"Route selected: {scenario.replace('_', ' ')}",
                )
            ).strip()
            normalized["postprocess_note"] = "route_enforced_rule_based_fallback"
            if failed_methods and not str(normalized.get("memory_ack", "")).strip():
                labels = self._format_method_list(failed_methods)
                normalized["memory_ack"] = (
                    f"Thanks for telling me that {labels} did not help enough. Let's try a different angle tonight."
                )
            return normalized

        normalized["suggestions"] = suggestions
        next_step = str(normalized.get("next_step", "")).strip()
        if failed_methods and (self._extract_methods_from_text(next_step) & failed_methods):
            next_step = ""
        if not next_step and suggestions:
            next_step = suggestions[0]
        normalized["next_step"] = next_step

        if failed_methods and not str(normalized.get("memory_ack", "")).strip():
            labels = self._format_method_list(failed_methods)
            normalized["memory_ack"] = (
                f"Thanks for sharing that {labels} did not help enough. We can try a different angle."
            )

        return normalized

    def _run_support_agents(
        self,
        user_input: str,
        state: DialogueState,
        keyword_scan: dict[str, Any],
    ) -> None:
        shared_context = {
            "user_input": user_input,
            "analysis": state.analyzer,
            "session_stage": state.session_stage,
            "stage_goal": state.stage_goal,
            "retrieved_knowledge": state.retrieved_knowledge,
            "strategy_route": state.strategy_route,
            "memory_context": state.memory_context,
            "user_state": state.user_state,
        }
        max_workers = 3 if self.run_config.use_safety else 2
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            empathy_future = executor.submit(self.agents["empathy"].run, shared_context)
            strategy_future = executor.submit(self.agents["strategy"].run, shared_context)
            safety_future = None
            if self.run_config.use_safety:
                safety_future = executor.submit(self.agents["safety"].run, shared_context)

            state.empathy = empathy_future.result()
            state.strategy = self._normalize_strategy_output(
                strategy_future.result(),
                state.strategy_route,
                user_input,
                state.user_state,
                state.memory_context,
            )
            if safety_future is None:
                state.safety = self._skipped_output(
                    "Safety ablation mode disabled this stage.",
                    risk_detected=False,
                    risk_level="unknown",
                    flags=[],
                    constraints=[],
                    required_actions=[],
                )
                state.safety_trace = {
                    "mode": "no_safety",
                    "keyword_risk_level": keyword_scan["risk_level"],
                    "keyword_hits": keyword_scan["keyword_hits"],
                    "final_risk_level": "unknown",
                    "escalation_needed": False,
                }
            else:
                state.safety, state.safety_trace = self._merge_safety_outputs(
                    safety_future.result(),
                    keyword_scan,
                )

    def _run_revision_stage(self, user_input: str, state: DialogueState) -> str:
        if self.run_config.use_critic:
            state.critic = self.agents["critic"].run(
                {
                    "user_input": user_input,
                    "analysis": state.analyzer,
                    "session_stage": state.session_stage,
                    "stage_goal": state.stage_goal,
                    "draft_response": state.coordinator.get("draft_response", ""),
                    "safety": state.safety,
                    "retrieved_knowledge": state.retrieved_knowledge,
                    "memory_context": state.memory_context,
                    "user_state": state.user_state,
                }
            )
        else:
            state.critic = self._skipped_output(
                "Critic ablation mode disabled this stage.",
                issues=[],
                revision_goals=[],
            )

        if self.run_config.use_reviser:
            state.reviser = self.agents["reviser"].run(
                {
                    "user_input": user_input,
                    "analysis": state.analyzer,
                    "session_stage": state.session_stage,
                    "stage_goal": state.stage_goal,
                    "draft_response": state.coordinator.get("draft_response", ""),
                    "critic_feedback": state.critic,
                    "safety": state.safety,
                    "retrieved_knowledge": state.retrieved_knowledge,
                    "strategy_route": state.strategy_route,
                    "memory_context": state.memory_context,
                    "user_state": state.user_state,
                }
            )
            return state.reviser.get(
                "final_response",
                state.coordinator.get("draft_response", ""),
            )

        state.reviser = self._skipped_output(
            "Reviser ablation mode disabled this stage.",
            final_response="",
        )
        return state.coordinator.get("draft_response", "")

    def _run_retrieval_stage(self, user_input: str, state: DialogueState) -> None:
        if not self.run_config.use_retrieval:
            state.retrieval = self._skipped_output(
                "Retrieval ablation mode disabled this stage.",
                top_k=self.settings.retrieval_top_k,
                hits=0,
            )
            state.retrieved_knowledge = []
            return

        if self.retriever is None:
            self.retriever = SupportKnowledgeRetriever(self.settings.support_kb_path)

        state.retrieved_knowledge = self.retriever.retrieve(
            user_input=user_input,
            analysis=state.analyzer,
            top_k=self.settings.retrieval_top_k,
        )
        state.retrieval = {
            "enabled": True,
            "kb_path": str(self.retriever.kb_path),
            "top_k": self.settings.retrieval_top_k,
            "hits": len(state.retrieved_knowledge),
            "hit_ids": [item.get("id", "") for item in state.retrieved_knowledge],
        }

    def _build_structured_response(
        self,
        state: DialogueState,
        candidate_response: str,
    ) -> tuple[str, dict[str, str]]:
        empathy = self._combine_empathy_text(
            str(state.empathy.get("acknowledgement", "")).strip(),
            str(state.empathy.get("validation", "")).strip(),
        )
        if not empathy:
            empathy = "Thank you for sharing this. Your feelings make sense in this situation."

        reflection_candidates = [
            str(state.critic.get("reflection_line", "")).strip(),
            str(state.critic.get("reframe_statement", "")).strip(),
            str(state.critic.get("thinking_pattern", "")).strip(),
            str(state.critic.get("underlying_need", "")).strip(),
        ]
        reflection = next((item for item in reflection_candidates if item), "")
        if not reflection:
            reflection = str(state.analyzer.get("notes", "")).strip()
        if not reflection:
            reflection = "You are dealing with real pressure, not a personal failure."
        reflection = self._humanize_reflection_text(reflection)
        reflection = self._soften_reflection_text(reflection)
        reflection = self._single_sentence(reflection)

        scenario = str(state.strategy_route.get("scenario", "general_support")).replace("_", " ")
        suggestions = self._as_text_list(state.strategy.get("suggestions", []))
        next_step = str(state.strategy.get("next_step", "")).strip()
        action_lines: list[str] = []
        memory_ack = str(state.strategy.get("memory_ack", "")).strip()
        if memory_ack:
            action_lines.append(memory_ack)
        if suggestions:
            action_lines.extend([f"- {item}" for item in suggestions[:4]])
        if next_step:
            action_lines.append(f"- {next_step}")
        if not action_lines and candidate_response:
            action_lines.append(candidate_response.strip())
        if not action_lines:
            action_lines.append(f"- We can focus on {scenario} with one gentle next step.")
        action_step = "\n".join(action_lines)

        risk_level = str(state.safety.get("risk_level", "unknown"))
        safety_flags = state.safety.get("flags", [])
        if risk_level == "high":
            safety = (
                "Your safety matters most right now. Please contact a trusted person immediately and use local emergency or crisis resources now if you might act on harmful thoughts."
            )
        elif risk_level == "medium":
            safety = (
                "Please keep support close tonight and avoid being alone if possible; if risk rises, contact emergency or crisis resources immediately."
            )
        else:
            safety = "If things feel heavier later, reach out early for extra support."
        if safety_flags:
            safety += f" Noted concerns: {', '.join(str(flag) for flag in safety_flags)}."

        sections = {
            "Empathy": empathy,
            "Reflection": reflection,
            "Action Step": action_step,
            "Safety": safety,
        }
        structured = (
            f"Empathy:\n{sections['Empathy']}\n\n"
            f"Reflection:\n{sections['Reflection']}\n\n"
            f"Action Step:\n{sections['Action Step']}\n\n"
            f"Safety:\n{sections['Safety']}"
        )
        return structured, sections

    def _render_final_response(
        self,
        state: DialogueState,
        structured_text: str,
        sections: dict[str, str],
    ) -> str:
        if self.settings.response_style == "structured":
            return structured_text
        if not self.settings.therapist_flow_strict:
            return self._render_standard_conversational_response(state, sections)

        variant = self._conversation_variant(state)
        session_turn = self._session_turn(state)
        stage = str(state.session_stage or "").strip() or self._determine_session_stage(
            state.user_input,
            state,
        )[0]
        empathy = self._single_sentence(sections.get("Empathy", "").strip())
        empathy = self._de_template_empathy(empathy, variant)
        reflection = self._single_sentence(sections.get("Reflection", "").strip())
        suggestions = self._as_text_list(state.strategy.get("suggestions", []))
        next_step = str(state.strategy.get("next_step", "")).strip()
        memory_ack = self._single_sentence(str(state.strategy.get("memory_ack", "")).strip())
        high_emotion_turn = self._is_high_emotion_turn(state)
        risk_level = str(state.safety.get("risk_level", "unknown")).lower()
        failure_turn = self._has_failure_cue(state.user_input)
        action_allowed = self._action_step_allowed(state.user_input, state)
        positive_turn = self._is_positive_turn(state.user_input, state)
        intro_turn = self._is_introduction_turn(state.user_input)
        previous_stage = str(state.user_state.get("session_stage", "")).strip()

        parts: list[str] = []

        # 1) Rapport / listening (keep concise and avoid stacked template lines)
        opening_bits: list[str] = []
        if empathy:
            opening_bits.append(empathy)
        if (
            reflection
            and reflection.lower() not in empathy.lower()
            and not self._is_redundant_sentence(empathy, reflection)
        ):
            opening_bits.append(reflection)
        if opening_bits:
            parts.append(" ".join(opening_bits))

        if risk_level == "high":
            parts.append(
                "Your safety comes first right now. Please contact a trusted person immediately, and use local emergency or crisis resources now if you might act on harmful thoughts."
            )
            return "\n\n".join(part for part in parts if part).strip()

        recall_response = self._memory_recall_relationship_response(state, variant)
        if recall_response:
            if risk_level == "medium":
                return (
                    f"{recall_response}\n\n"
                    "If things feel heavier tonight, please stay connected with someone you trust rather than carrying this alone."
                )
            return recall_response

        if positive_turn and risk_level == "low" and not failure_turn:
            if self._is_brief_positive_checkin(state.user_input, state):
                lead = "I am glad to hear this moment feels a little lighter."
            else:
                lead = self._select_positive_opening(
                    empathy,
                    reflection,
                    "I am glad to hear this moment feels a little lighter.",
                )
            parts = [lead, self._positive_followup_question(state, variant)]
            if not self._is_brief_positive_checkin(state.user_input, state):
                parts.append(self._collaborative_close(state, variant))
            return "\n\n".join(part for part in parts if part).strip()

        # 2) Follow-up stage: first understand why prior attempt did not work.
        if stage == "follow_up" or failure_turn:
            followup_parts = self._failed_coping_followup_parts(
                state,
                memory_ack,
                variant,
            )
            # Keep only the most human pieces: validation + mechanism + one question.
            parts.extend(followup_parts[:2])
            parts.append(followup_parts[-1])
            if risk_level == "medium":
                parts.append(
                    "If things feel heavier tonight, please stay connected with someone you trust rather than carrying this alone."
                )
            return "\n\n".join(part for part in parts if part).strip()

        primary_step = next_step or (suggestions[0] if suggestions else "")
        alternative_step = suggestions[1] if len(suggestions) > 1 else ""

        # 3) Stage-driven flow
        stage_transition = self._stage_transition_line(stage)
        if stage_transition and previous_stage != stage:
            parts.append(stage_transition)

        if stage in {
            "rapport_and_safety",
            "problem_exploration",
            "emotion_identification",
            "root_cause_exploration",
        }:
            if stage in {"emotion_identification", "root_cause_exploration"}:
                parts.append(self._emotion_identification_line(state))
            if stage == "root_cause_exploration":
                parts.append(self._reframe_line(state, variant))
            parts.append(self._stage_prompt_question(state, stage, variant))
        elif stage == "goal_setting":
            parts.append(self._reframe_line(state, variant))
            parts.append(self._stage_prompt_question(state, stage, variant))
            if primary_step and action_allowed:
                parts.append(self._gentle_action_invite(primary_step, alternative_step, variant))
        elif stage == "intervention":
            if high_emotion_turn:
                parts.append("We can keep this to one very small step for tonight.")
            if action_allowed:
                parts.append(
                    self._goal_or_action_line(
                        state,
                        primary_step,
                        alternative_step,
                        variant,
                    )
                )
            else:
                parts.append("Before we pick a step, I want to understand one layer deeper.")
            parts.append(self._stage_prompt_question(state, stage, variant))
        elif stage == "session_closing":
            summary = self._single_sentence(str(state.analyzer.get("notes", "")).strip())
            if summary:
                parts.append(f"What I heard today: {summary}")
            if primary_step:
                parts.append(f"For now, one gentle next step: {primary_step}")
            parts.append(self._stage_prompt_question(state, stage, variant))
        else:
            parts.append(self._emotion_identification_line(state))
            parts.append(self._stage_prompt_question(state, stage, variant))

        # 4) Safety + close
        if risk_level == "medium":
            parts.append(
                "If things feel heavier tonight, please stay connected with someone you trust rather than carrying this alone."
            )
        elif state.safety.get("flags"):
            parts.append("If this gets heavier later, reach out early for extra support.")

        if stage in {"goal_setting", "intervention", "session_closing"}:
            parts.append(self._collaborative_close(state, variant))
        elif stage == "rapport_and_safety" and not positive_turn and not intro_turn:
            parts.append("If you want, we can look at one specific moment together.")

        return "\n\n".join(part for part in parts if part).strip()

    def _update_memory(self, state: DialogueState) -> None:
        previous_failed_methods = set(
            self._as_text_list(self.user_state.get("failed_methods", []))
        )
        current_failed_methods = set(
            self._as_text_list(state.strategy.get("failed_methods_considered", []))
        )
        if self._has_failure_cue(state.user_input):
            user_reported_methods = self._extract_methods_from_text(state.user_input)
            if user_reported_methods:
                current_failed_methods.update(user_reported_methods)
            else:
                last_strategy_method = str(
                    self.user_state.get("last_strategy_method", "")
                ).strip()
                if last_strategy_method:
                    current_failed_methods.add(last_strategy_method)

        strategy_suggestions = self._as_text_list(state.strategy.get("suggestions", []))
        strategy_next_step = str(state.strategy.get("next_step", "")).strip()
        strategy_methods = self._extract_methods_from_items(
            strategy_suggestions + ([strategy_next_step] if strategy_next_step else [])
        )
        next_step_methods = self._extract_methods_from_text(strategy_next_step)
        if next_step_methods:
            last_strategy_method = sorted(next_step_methods)[0]
        elif strategy_methods:
            last_strategy_method = sorted(strategy_methods)[0]
        else:
            last_strategy_method = ""

        self.user_state = {
            "main_stressor": state.analyzer.get("topic", ""),
            "emotion": state.analyzer.get("emotion", ""),
            "risk_level": state.safety.get("risk_level", ""),
            "session_stage": state.session_stage,
            "stage_goal": state.stage_goal,
            "last_strategy": strategy_next_step,
            "last_strategy_method": last_strategy_method,
            "strategy_methods": sorted(strategy_methods),
            "failed_methods": sorted(previous_failed_methods | current_failed_methods),
            "last_reflection": state.critic.get("thinking_pattern", ""),
        }
        if self.settings.memory_window == 0:
            self._history = []
            self._save_persistent_memory()
            return
        self._history.append(
            {
                "session_id": self._runtime_session_id,
                "created_at": datetime.now(UTC).isoformat(),
                "user_input": state.user_input,
                "emotion": state.analyzer.get("emotion", ""),
                "topic": state.analyzer.get("topic", ""),
                "risk_level": state.safety.get("risk_level", ""),
                "strategy_route": state.strategy_route.get("scenario", ""),
                "session_stage": state.session_stage,
                "strategy_next_step": strategy_next_step,
                "strategy_suggestions": strategy_suggestions[:4],
                "strategy_methods": sorted(strategy_methods),
                "failed_methods_snapshot": sorted(
                    previous_failed_methods | current_failed_methods
                ),
                "final_response": state.final_response,
            }
        )
        if (
            self.settings.persistent_history_limit > 0
            and len(self._history) > self.settings.persistent_history_limit
        ):
            self._history = self._history[-self.settings.persistent_history_limit :]
        self._save_persistent_memory()

    def run(self, user_input: str) -> DialogueState:
        state = DialogueState(
            user_input=user_input,
            system_name="pipeline",
            run_mode=self.run_config.name,
            profile_id=self.profile_id,
        )
        state.memory_context = self._memory_context()
        state.user_state = dict(self.user_state)

        state.analyzer = self.agents["analyzer"].run(
            {
                "user_input": user_input,
                "memory_context": state.memory_context,
                "user_state": state.user_state,
            }
        )
        state.session_stage, state.stage_goal = self._determine_session_stage(
            user_input,
            state,
        )
        keyword_scan = self._keyword_safety_scan(user_input)
        state.strategy_route = self._build_strategy_route(
            user_input,
            state.analyzer,
            keyword_scan,
        )
        self._run_retrieval_stage(user_input, state)
        self._run_support_agents(user_input, state, keyword_scan)
        state.session_stage, state.stage_goal = self._determine_session_stage(
            user_input,
            state,
        )

        print(f"[safety_trace] {json.dumps(state.safety_trace, ensure_ascii=False)}")

        state.coordinator = self.agents["coordinator"].run(
            {
                "user_input": user_input,
                "analysis": state.analyzer,
                "session_stage": state.session_stage,
                "stage_goal": state.stage_goal,
                "retrieval": state.retrieval,
                "retrieved_knowledge": state.retrieved_knowledge,
                "strategy_route": state.strategy_route,
                "memory_context": state.memory_context,
                "user_state": state.user_state,
                "empathy": state.empathy,
                "strategy": state.strategy,
                "safety": state.safety,
            }
        )

        draft_response = self._run_revision_stage(user_input, state)
        candidate_response, candidate_sections = self._build_structured_response(
            state,
            draft_response,
        )
        state.final_response_sections = candidate_sections

        if self.run_config.use_safety:
            state.final_check = self.agents["final_checker"].run(
                {
                    "user_input": user_input,
                    "candidate_response": candidate_response,
                    "safety": state.safety,
                    "safety_trace": state.safety_trace,
                }
            )
            checked = state.final_check.get("final_response", candidate_response)
            checked_structured, checked_sections = self._build_structured_response(
                state,
                checked,
            )
            state.final_response_sections = checked_sections
            state.final_response = self._render_final_response(
                state,
                checked_structured,
                checked_sections,
            )
        else:
            state.final_check = self._skipped_output(
                "Safety ablation mode skipped the final safety check.",
                approved=True,
                final_response=candidate_response,
                notes=[],
            )
            state.final_response = self._render_final_response(
                state,
                candidate_response,
                candidate_sections,
            )

        state.final_response = self._apply_memory_recall_guard(
            state,
            state.final_response,
        )
        state.final_response = self._strip_unprompted_memory_lead(
            state,
            state.final_response,
        )
        self._update_memory(state)
        return state
