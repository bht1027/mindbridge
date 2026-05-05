import time
import random

import streamlit as st

from config import get_settings
from pipeline import MindBridgePipeline


st.set_page_config(page_title="MindBridge Demo", page_icon="MB", layout="wide")


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --mb-ink: #2f3142;
            --mb-muted: #72768b;
            --mb-rose: #ff6f7d;
            --mb-coral: #ff9b73;
            --mb-sun: #ffd166;
            --mb-mint: #85dcb8;
            --mb-sky: #79b8ff;
            --mb-lavender: #b89cff;
            --mb-paper: rgba(255, 255, 255, 0.78);
            --mb-line: rgba(95, 99, 125, 0.14);
        }

        .stApp {
            color: var(--mb-ink);
            background:
                radial-gradient(circle at 12% 8%, rgba(255, 209, 102, 0.26), transparent 28%),
                radial-gradient(circle at 78% 6%, rgba(121, 184, 255, 0.24), transparent 26%),
                linear-gradient(135deg, #fff8ef 0%, #f7fbff 42%, #f7f2ff 100%);
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.5);
            backdrop-filter: blur(14px);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.82), rgba(245, 249, 255, 0.88));
            border-right: 1px solid var(--mb-line);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {
            color: var(--mb-muted);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 3.2rem;
            padding-bottom: 7rem;
        }

        .mb-hero {
            padding: 0.6rem 0 1.05rem;
        }

        .mb-kicker {
            color: #5b6f95;
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .mb-title {
            color: var(--mb-ink);
            font-size: clamp(2.05rem, 3.8vw, 3.45rem);
            font-weight: 850;
            line-height: 1.04;
            margin: 0;
        }

        .mb-subtitle {
            color: var(--mb-muted);
            font-size: 1.08rem;
            line-height: 1.65;
            max-width: 760px;
            margin-top: 1rem;
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid rgba(255, 255, 255, 0.72);
            border-radius: 18px;
            box-shadow: 0 16px 34px rgba(73, 82, 115, 0.08);
            padding: 0.72rem 0.85rem;
            margin: 0.85rem 0;
        }

        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: rgba(255, 255, 255, 0.74);
        }

        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.86), rgba(239, 249, 245, 0.84));
        }

        [data-testid="stChatInput"] {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(255, 255, 255, 0.9);
            border-radius: 22px;
            box-shadow: 0 20px 54px rgba(87, 91, 125, 0.16);
        }

        .stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(91, 111, 149, 0.18);
            background: rgba(255, 255, 255, 0.76);
            color: var(--mb-ink);
            box-shadow: 0 10px 24px rgba(93, 101, 134, 0.09);
            transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
        }

        .stButton > button:hover {
            border-color: rgba(255, 111, 125, 0.42);
            box-shadow: 0 14px 30px rgba(93, 101, 134, 0.13);
            transform: translateY(-1px);
        }

        div[role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.56);
            border: 1px solid rgba(95, 99, 125, 0.12);
            border-radius: 14px;
            margin-bottom: 0.45rem;
            padding: 0.42rem 0.58rem;
        }

        .stTextInput input {
            border-radius: 14px;
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.62);
            border: 1px solid rgba(95, 99, 125, 0.12);
            border-radius: 16px;
            padding: 0.7rem 0.8rem;
        }

        hr {
            border-color: rgba(95, 99, 125, 0.13);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_theme()

st.markdown(
    """
    <section class="mb-hero">
        <div class="mb-kicker">MindBridge</div>
        <h1 class="mb-title">A softer place to think things through.</h1>
        <div class="mb-subtitle">
            Gentle dialogue with memory, reflection, and safety checks for moments that feel heavy,
            tangled, or hard to name.
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

JOURNAL_PROMPTS = (
    "What felt heaviest today, and what would make tonight 5% easier?",
    "Which thought keeps looping, and what is one gentler reframe?",
    "What is one small action that would help future-you tomorrow morning?",
    "What do you need right now: rest, clarity, support, or boundaries?",
)

BREATH_PHASES = (
    ("Breathe in", 4),
    ("Hold", 2),
    ("Breathe out", 4),
)

RESPONSE_MODES = {
    "Light Assistant": {
        "engine": "multi_agent_pipeline",
        "strict": False,
        "hint": "Lighter support style with profile memory.",
    },
    "CBT-Informed": {
        "engine": "multi_agent_pipeline",
        "strict": True,
        "hint": "Multi-agent support with gentle CBT-informed reflection and planning.",
    },
}

LEGACY_MODE_ALIASES = {
    "Light Assistant (Baseline / Single-Agent)": "Light Assistant",
    "Strict Therapy (Deeper Thinking / Multi-Agent)": "CBT-Informed",
    "Strict Therapy": "CBT-Informed",
}


def _stream_text(text: str, chunk_size: int = 8, delay_s: float = 0.01):
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]
        time.sleep(delay_s)


def _one_line_preview(text: str, limit: int = 48) -> str:
    cleaned = " ".join(str(text).strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _boot_session(
    mode_label: str | None = None,
    profile_id: str | None = None,
    hydrate_from_persistent: bool = True,
) -> None:
    settings = get_settings()
    settings.validate()

    selected_mode = mode_label or st.session_state.get(
        "response_mode",
        "CBT-Informed",
    )
    selected_mode = LEGACY_MODE_ALIASES.get(selected_mode, selected_mode)
    if selected_mode not in RESPONSE_MODES:
        selected_mode = "CBT-Informed"
    selected_profile = (profile_id or st.session_state.get("profile_id", "default")).strip() or "default"
    mode_cfg = RESPONSE_MODES[selected_mode]
    settings.therapist_flow_strict = bool(mode_cfg["strict"])

    runner = MindBridgePipeline(settings, profile_id=selected_profile)
    if not hydrate_from_persistent:
        runner.reset_runtime_memory()
    chat_log = (
        runner.get_active_session_turns_snapshot(limit_turns=80)
        if hydrate_from_persistent
        else []
    )
    profile_history = runner.get_session_history_snapshot(
        limit_sessions=50,
        exclude_session_id=runner.runtime_session_id,
    )

    st.session_state.response_mode = selected_mode
    st.session_state.profile_id = selected_profile
    st.session_state.runner_kind = mode_cfg["engine"]
    st.session_state.runner = runner
    st.session_state.chat_log = chat_log
    st.session_state.profile_history = profile_history
    st.session_state.last_state = None
    st.session_state.journal_prompt = random.choice(JOURNAL_PROMPTS)
    st.session_state.selected_profile_session_id = None

if "runner" not in st.session_state:
    _boot_session()
if "profile_history" not in st.session_state:
    st.session_state.profile_history = []
if (
    st.session_state.get("runner_kind") == "multi_agent_pipeline"
    and st.session_state.get("profile_history")
):
    first_row = st.session_state.profile_history[0]
    if not isinstance(first_row, dict) or "session_id" not in first_row:
        runner = st.session_state.get("runner")
        if runner is not None and hasattr(runner, "get_session_history_snapshot"):
            st.session_state.profile_history = runner.get_session_history_snapshot(
                limit_sessions=50,
                exclude_session_id=getattr(runner, "runtime_session_id", ""),
            )
            st.session_state.selected_profile_session_id = None

def _run_turn(clean_input: str) -> None:
    mode_cfg = RESPONSE_MODES[st.session_state.response_mode]
    spinner_text = "Deeper thinking..." if mode_cfg["strict"] else "Thinking..."
    with st.chat_message("user"):
        st.write(clean_input)

    with st.chat_message("assistant"):
        with st.spinner(spinner_text):
            result = st.session_state.runner.run(clean_input)
        streamed_response = st.write_stream(_stream_text(result.final_response))

    strategy_route = getattr(result, "strategy_route", {}) or {}
    safety = getattr(result, "safety", {}) or {}
    route = strategy_route.get("scenario", "single_agent_baseline")
    risk = safety.get("risk_level", "n/a")

    st.session_state.chat_log.append(
        {
            "user": clean_input,
            "assistant": streamed_response,
            "route": route,
            "risk": risk,
        }
    )
    st.session_state.last_state = result

with st.sidebar:
    st.markdown("### Session")
    pending_input_override = ""
    profile_input = st.text_input(
        "Profile ID",
        value=st.session_state.get("profile_id", "default"),
        help="Use the same Profile ID to load previous memory next time.",
    ).strip() or "default"
    if profile_input != st.session_state.get("profile_id", "default"):
        _boot_session(
            st.session_state.response_mode,
            profile_input,
            hydrate_from_persistent=True,
        )
        st.rerun()

    mode_options = list(RESPONSE_MODES.keys())
    current_mode = LEGACY_MODE_ALIASES.get(
        st.session_state.get("response_mode", "CBT-Informed"),
        st.session_state.get("response_mode", "CBT-Informed"),
    )
    if current_mode not in mode_options:
        current_mode = "CBT-Informed"
        st.session_state.response_mode = current_mode
    selected_mode = st.radio(
        "Response Mode",
        options=mode_options,
        index=mode_options.index(current_mode),
    )
    st.caption(RESPONSE_MODES[selected_mode]["hint"])
    if selected_mode != st.session_state.response_mode:
        _boot_session(
            selected_mode,
            st.session_state.profile_id,
            hydrate_from_persistent=True,
        )
        st.rerun()

    if st.button("New Chat"):
        if st.session_state.runner_kind == "multi_agent_pipeline":
            runner = st.session_state.runner
            runner.start_new_session()
            st.session_state.chat_log = []
            st.session_state.last_state = None
            st.session_state.selected_profile_session_id = None
            st.session_state.profile_history = runner.get_session_history_snapshot(
                limit_sessions=50,
                exclude_session_id=runner.runtime_session_id,
            )
        else:
            _boot_session(
                st.session_state.response_mode,
                st.session_state.profile_id,
                hydrate_from_persistent=False,
            )
        st.rerun()
    if (
        st.session_state.runner_kind == "multi_agent_pipeline"
        and st.button("Clear Saved Profile Memory")
    ):
        st.session_state.runner.clear_persistent_memory()
        _boot_session(
            st.session_state.response_mode,
            st.session_state.profile_id,
            hydrate_from_persistent=False,
        )
        st.rerun()
    show_debug = st.checkbox("Show Debug Panel", value=False)

    st.divider()
    st.subheader("Previous Chats (Profile)")
    profile_history = st.session_state.get("profile_history", [])
    if not profile_history:
        st.caption("No previous chats for this profile yet.")
    else:
        visible_history = profile_history[-12:]
        for local_idx, session_item in enumerate(reversed(visible_history), start=1):
            session_id = str(session_item.get("session_id", "")).strip()
            user_text = _one_line_preview(session_item.get("title", ""), limit=44)
            label = f"{local_idx}. {user_text or '(empty message)'}"
            if st.button(
                label,
                key=f"profile_history_item_{session_id or local_idx}",
                use_container_width=True,
            ):
                st.session_state.selected_profile_session_id = session_id

    if st.session_state.chat_log and st.button("Retry Last User Message", use_container_width=True):
        pending_input_override = st.session_state.chat_log[-1]["user"]

selected_profile_turns: list[dict[str, str]] = []
if st.session_state.runner_kind == "multi_agent_pipeline":
    selected_session_id = str(
        st.session_state.get("selected_profile_session_id", "")
    ).strip()
    if selected_session_id:
        runner = st.session_state.get("runner")
        if runner is not None and hasattr(runner, "get_session_turns_snapshot"):
            selected_profile_turns = runner.get_session_turns_snapshot(
                selected_session_id,
                limit_turns=80,
            )

if selected_profile_turns:
    st.caption("Preview from previous chats")
    for turn in selected_profile_turns:
        with st.chat_message("user"):
            st.write(turn.get("user", ""))
        with st.chat_message("assistant"):
            st.markdown(turn.get("assistant", ""))
    route = str(selected_profile_turns[-1].get("route", "general_support")).strip()
    risk = str(selected_profile_turns[-1].get("risk", "unknown")).strip()
    st.caption(f"Saved chat route: {route} | risk: {risk}")
    st.divider()

if st.session_state.chat_log:
    for turn in st.session_state.chat_log:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.markdown(turn["assistant"])

user_input = st.chat_input("Share what is on your mind.")
pending_input = pending_input_override
if user_input and user_input.strip():
    pending_input = user_input.strip()

if pending_input:
    _run_turn(pending_input)

if show_debug and st.session_state.last_state is not None:
    last_state = st.session_state.last_state
    with st.expander("Debug Panel", expanded=False):
        if st.session_state.runner_kind == "multi_agent_pipeline":
            st.json(
                {
                    "engine": st.session_state.runner_kind,
                    "profile_id": getattr(last_state, "profile_id", st.session_state.get("profile_id", "default")),
                    "latest_input": last_state.user_input,
                    "session_stage": getattr(last_state, "session_stage", ""),
                    "stage_goal": getattr(last_state, "stage_goal", ""),
                    "intent": last_state.analyzer.get("intent", ""),
                    "topic": last_state.analyzer.get("topic", ""),
                    "strategy_route": last_state.strategy_route.get("scenario", ""),
                    "risk": last_state.safety.get("risk_level", ""),
                    "safety_trace": last_state.safety_trace,
                    "reflection_type": last_state.critic.get("thinking_pattern", ""),
                    "memory_ack": last_state.strategy.get("memory_ack", ""),
                    "failed_methods_considered": last_state.strategy.get(
                        "failed_methods_considered", []
                    ),
                    "memory_turns_used": len(last_state.memory_context),
                    "therapist_flow_strict": st.session_state.runner.settings.therapist_flow_strict,
                    "persistent_memory_enabled": st.session_state.runner.settings.persistent_memory_enabled,
                    "persistent_memory_path": st.session_state.runner.settings.persistent_memory_path,
                }
            )
        else:
            st.json(
                {
                    "engine": st.session_state.runner_kind,
                    "latest_input": last_state.user_input,
                    "mode": st.session_state.response_mode,
                    "therapist_flow_strict": st.session_state.runner.settings.therapist_flow_strict,
                }
            )
        st.caption("Raw state")
        if hasattr(last_state, "to_dict"):
            st.json(last_state.to_dict(), expanded=False)
        else:
            st.write(last_state)

st.divider()
st.markdown(
    "This demo is a project scaffold, not a production mental-health support system."
)
