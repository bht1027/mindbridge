import time
import random

import streamlit as st

from baseline import SingleAgentBaseline
from config import get_settings
from pipeline import MindBridgePipeline


st.set_page_config(page_title="MindBridge Demo", page_icon="MB", layout="wide")


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

        html, body, p, h1, h2, h3, h4, h5, h6,
        button, input, textarea, label,
        .stMarkdown, .stChatMessage p,
        [data-testid="stChatMessageContent"] {
            font-family: 'Nunito', sans-serif !important;
        }

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
            color: #171923;
        }

        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: rgba(255, 255, 255, 0.82);
            border-left: 3px solid rgba(255, 155, 115, 0.5);
            margin-left: 8%;
        }

        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background: linear-gradient(135deg, rgba(228, 248, 240, 0.9), rgba(255, 252, 246, 0.86));
            border-left: 3px solid rgba(133, 220, 184, 0.6);
            margin-right: 8%;
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] span,
        div[data-testid="stChatMessage"] div {
            color: #171923 !important;
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

        /* ── Floating orb animations ── */
        @keyframes floatA {
            0%   { transform: translate(0px, 0px) scale(1); }
            33%  { transform: translate(18px, -24px) scale(1.06); }
            66%  { transform: translate(-12px, 14px) scale(0.96); }
            100% { transform: translate(0px, 0px) scale(1); }
        }
        @keyframes floatB {
            0%   { transform: translate(0px, 0px) scale(1); }
            40%  { transform: translate(-22px, 18px) scale(1.08); }
            70%  { transform: translate(16px, -10px) scale(0.94); }
            100% { transform: translate(0px, 0px) scale(1); }
        }
        @keyframes floatC {
            0%   { transform: translate(0px, 0px) scale(1); }
            50%  { transform: translate(14px, 22px) scale(1.04); }
            100% { transform: translate(0px, 0px) scale(1); }
        }

        .mb-orb {
            position: fixed;
            border-radius: 50%;
            filter: blur(62px);
            pointer-events: none;
            z-index: 0;
            opacity: 0.45;
        }
        .mb-orb-1 {
            width: 340px; height: 340px;
            background: radial-gradient(circle, #ffd18880, #ff9b7340);
            top: -80px; left: -60px;
            animation: floatA 14s ease-in-out infinite;
        }
        .mb-orb-2 {
            width: 280px; height: 280px;
            background: radial-gradient(circle, #b89cff60, #79b8ff40);
            top: 5%; right: 2%;
            animation: floatB 18s ease-in-out infinite;
        }
        .mb-orb-3 {
            width: 220px; height: 220px;
            background: radial-gradient(circle, #85dcb870, #ffd16650);
            bottom: 12%; left: 8%;
            animation: floatC 16s ease-in-out infinite;
        }
        .mb-orb-4 {
            width: 180px; height: 180px;
            background: radial-gradient(circle, #ff6f7d50, #b89cff40);
            bottom: 5%; right: 10%;
            animation: floatA 20s ease-in-out infinite reverse;
        }

        /* ── Message fade-in ── */
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        div[data-testid="stChatMessage"] {
            animation: fadeUp 0.38s ease both;
        }

        /* ── Kicker shimmer ── */
        @keyframes shimmer {
            0%   { background-position: -200% center; }
            100% { background-position: 200% center; }
        }
        .mb-kicker {
            background: linear-gradient(90deg, #5b6f95 30%, #ff9b73 50%, #5b6f95 70%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: shimmer 5s linear infinite;
        }

        /* ── Starter card pulse ── */
        @keyframes softPulse {
            0%, 100% { box-shadow: 0 8px 22px rgba(93,101,134,0.07); }
            50%       { box-shadow: 0 8px 28px rgba(255,111,125,0.13); }
        }
        .mb-starter-card {
            animation: softPulse 4s ease-in-out infinite;
        }

        /* ── Journal card animated border ── */
        @keyframes borderGlow {
            0%, 100% { border-color: rgba(255,180,130,0.3); }
            50%       { border-color: rgba(255,140,80,0.55); }
        }
        .mb-journal-card {
            animation: borderGlow 5s ease-in-out infinite;
        }

        /* ── Floating dots in empty state ── */
        @keyframes floatDot {
            0%, 100% { transform: translateY(0px); opacity: 0.5; }
            50%       { transform: translateY(-8px); opacity: 1; }
        }
        .mb-dot {
            display: inline-block;
            width: 8px; height: 8px;
            border-radius: 50%;
            margin: 0 4px;
            animation: floatDot 2s ease-in-out infinite;
        }
        .mb-dot:nth-child(2) { animation-delay: 0.3s; }
        .mb-dot:nth-child(3) { animation-delay: 0.6s; }

        /* ── Thinking dots ── */
        @keyframes thinkDot {
            0%,80%,100% { transform: translateY(0); opacity: 0.35; }
            40%          { transform: translateY(-9px); opacity: 1; }
        }
        .mb-thinking {
            display: flex; gap: 7px; align-items: center; padding: 6px 4px;
        }
        .mb-t-dot {
            width: 9px; height: 9px; border-radius: 50%;
            animation: thinkDot 1.3s ease-in-out infinite;
        }
        .mb-t-dot:nth-child(1) { background: #ff9b73; }
        .mb-t-dot:nth-child(2) { background: #85dcb8; animation-delay: 0.2s; }
        .mb-t-dot:nth-child(3) { background: #b89cff; animation-delay: 0.4s; }

        /* ── Mochi the cat ── */
        @keyframes petBob {
            0%,100% { transform: translateY(0px); }
            50%      { transform: translateY(-7px); }
        }
        @keyframes newBounce {
            0%   { transform: translateY(0px) scale(1); }
            20%  { transform: translateY(-18px) scale(1.06); }
            40%  { transform: translateY(0px) scale(0.97); }
            60%  { transform: translateY(-8px) scale(1.02); }
            80%  { transform: translateY(0px) scale(1); }
            100% { transform: translateY(0px) scale(1); }
        }
        @keyframes heartPop {
            0%   { opacity:0; transform:translateY(0) scale(0.5); }
            30%  { opacity:1; transform:translateY(-14px) scale(1.2); }
            70%  { opacity:1; transform:translateY(-22px) scale(1); }
            100% { opacity:0; transform:translateY(-30px) scale(0.8); }
        }
        @keyframes tailWag {
            0%,100% { transform: rotate(-28deg); }
            50%      { transform: rotate(22deg); }
        }
        @keyframes blink {
            0%,90%,100% { transform: scaleY(1); }
            94%          { transform: scaleY(0.08); }
        }
        @keyframes blush {
            0%,100% { opacity: 0.45; }
            50%      { opacity: 0.85; }
        }
        @keyframes bubble1 {
            0%,12%       { opacity:0; transform:scale(0.85) translateY(6px); }
            16%,36%      { opacity:1; transform:scale(1) translateY(0); }
            42%,100%     { opacity:0; transform:scale(0.85) translateY(-4px); }
        }
        @keyframes bubble2 {
            0%,42%       { opacity:0; transform:scale(0.85) translateY(6px); }
            46%,66%      { opacity:1; transform:scale(1) translateY(0); }
            72%,100%     { opacity:0; transform:scale(0.85) translateY(-4px); }
        }
        @keyframes bubble3 {
            0%,72%       { opacity:0; transform:scale(0.85) translateY(6px); }
            76%,92%      { opacity:1; transform:scale(1) translateY(0); }
            97%,100%     { opacity:0; }
        }

        .mb-pet-wrap {
            position: fixed;
            bottom: 30px; right: 30px;
            z-index: 9999;
            width: 74px;
            animation: petBob 3.2s ease-in-out infinite;
            cursor: default;
            user-select: none;
        }
        .mb-pet-bounce {
            animation: newBounce 0.7s ease-out, petBob 3.2s ease-in-out 0.7s infinite !important;
        }
        /* worried expression */
        .mb-pet-worried .mb-eye-l { transform: rotate(15deg) scaleY(0.75); }
        .mb-pet-worried .mb-eye-r { transform: rotate(-15deg) scaleY(0.75); }
        .mb-pet-worried .mb-cheek-l,
        .mb-pet-worried .mb-cheek-r { opacity: 0.1 !important; }
        .mb-pet-worried .mb-nose { border-top-color: #e08060; }
        /* happy expression */
        .mb-pet-happy .mb-eye-l,
        .mb-pet-happy .mb-eye-r {
            height: 7px; border-radius: 50% 50% 0 0;
            background: #3d2b1f;
        }
        .mb-pet-happy .mb-cheek-l,
        .mb-pet-happy .mb-cheek-r { opacity: 0.9 !important; }
        /* floating heart */
        .mb-heart {
            position: absolute;
            top: -10px; left: 50%;
            transform: translateX(-50%);
            font-size: 1rem;
            animation: heartPop 1.2s ease-out forwards;
            pointer-events: none;
        }
        .mb-bubble {
            position: absolute;
            bottom: 115%; right: 0;
            background: rgba(255,255,255,0.95);
            border: 1.5px solid rgba(255,180,120,0.45);
            border-radius: 14px 14px 4px 14px;
            padding: 6px 11px;
            font-size: 0.7rem;
            color: #5a3820;
            white-space: nowrap;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            pointer-events: none;
        }
        .mb-bubble-1 { animation: bubble1 18s ease-in-out infinite; }
        .mb-bubble-2 { animation: bubble2 18s ease-in-out infinite; }
        .mb-bubble-3 { animation: bubble3 18s ease-in-out infinite; }

        /* head */
        .mb-cat-head {
            width: 58px; height: 52px;
            background: #FFF0DC;
            border-radius: 50% 50% 46% 46%;
            margin: 0 auto;
            position: relative;
            box-shadow: inset 0 -3px 7px rgba(200,140,70,0.14);
        }
        /* ears via pseudo */
        .mb-cat-head::before,.mb-cat-head::after {
            content:'';
            position:absolute; top:-14px;
            width:0; height:0;
            border-left:11px solid transparent;
            border-right:11px solid transparent;
            border-bottom:20px solid #FFF0DC;
        }
        .mb-cat-head::before { left:4px; }
        .mb-cat-head::after  { right:4px; }
        /* inner ear color overlay */
        .mb-ear-inner-l,.mb-ear-inner-r {
            position:absolute; top:-9px;
            width:0; height:0;
            border-left:6px solid transparent;
            border-right:6px solid transparent;
            border-bottom:12px solid rgba(255,170,140,0.5);
        }
        .mb-ear-inner-l { left:9px; }
        .mb-ear-inner-r { right:9px; }
        /* eyes */
        .mb-eye-l,.mb-eye-r {
            position:absolute; top:19px;
            width:9px; height:9px;
            background:#3d2b1f;
            border-radius:50%;
            animation: blink 4.5s ease-in-out infinite;
            transform-origin:center;
        }
        .mb-eye-l { left:11px; }
        .mb-eye-r { right:11px; }
        .mb-eye-l::after,.mb-eye-r::after {
            content:''; position:absolute;
            top:1px; left:2px;
            width:3px; height:3px;
            background:white; border-radius:50%;
        }
        /* nose */
        .mb-nose {
            position:absolute; top:31px; left:50%;
            transform:translateX(-50%);
            width:0; height:0;
            border-left:4px solid transparent;
            border-right:4px solid transparent;
            border-top:5px solid #FFA07A;
        }
        /* cheeks */
        .mb-cheek-l,.mb-cheek-r {
            position:absolute; top:30px;
            width:13px; height:7px;
            background:rgba(255,150,110,0.38);
            border-radius:50%;
            animation: blush 3s ease-in-out infinite;
        }
        .mb-cheek-l { left:2px; }
        .mb-cheek-r { right:2px; }
        /* body wrap (for tail positioning) */
        .mb-cat-body-wrap {
            position:relative;
            width:74px; margin:0 auto;
        }
        /* body */
        .mb-cat-body {
            width:50px; height:40px;
            background:#FFF0DC;
            border-radius:28% 28% 38% 38%;
            margin:-5px auto 0;
            position:relative;
            box-shadow:inset 0 -3px 8px rgba(200,140,70,0.12);
        }
        /* belly */
        .mb-cat-body::after {
            content:''; position:absolute;
            bottom:8px; left:50%;
            transform:translateX(-50%);
            width:24px; height:18px;
            background:rgba(255,242,224,0.75);
            border-radius:50%;
        }
        /* paws */
        .mb-paw-l,.mb-paw-r {
            position:absolute; bottom:-7px;
            width:14px; height:9px;
            background:#FFF0DC;
            border-radius:50% 50% 38% 38%;
        }
        .mb-paw-l { left:5px; }
        .mb-paw-r { right:5px; }
        /* tail */
        .mb-tail {
            position:absolute;
            bottom:4px; right:-4px;
            width:9px; height:34px;
            background:linear-gradient(to bottom, #FFD4A0, #FFBC80);
            border-radius:0 6px 10px 4px;
            transform-origin:top center;
            animation: tailWag 2s ease-in-out infinite;
        }

        /* Starter prompt cards */
        .mb-starters {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
            margin-top: 1.6rem;
        }

        .mb-starter-card {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(95, 99, 125, 0.13);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            cursor: pointer;
            transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
            box-shadow: 0 8px 22px rgba(93, 101, 134, 0.07);
        }

        .mb-starter-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 32px rgba(93, 101, 134, 0.13);
            border-color: rgba(255, 111, 125, 0.3);
        }

        .mb-starter-icon {
            font-size: 1.4rem;
            margin-bottom: 0.4rem;
        }

        .mb-starter-text {
            font-size: 0.9rem;
            color: var(--mb-ink);
            line-height: 1.45;
        }

        /* Response badge */
        .mb-badge {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 20px;
            font-size: 0.72rem;
            font-weight: 600;
            margin-right: 0.3rem;
        }

        .mb-badge-low    { background: rgba(133,220,184,0.28); color: #2a7a55; }
        .mb-badge-medium { background: rgba(255,209,102,0.32); color: #7a5a00; }
        .mb-badge-high   { background: rgba(255,111,125,0.22); color: #a02030; }
        .mb-badge-route  { background: rgba(121,184,255,0.24); color: #1a4a80; }

        /* Journal card */
        .mb-journal-card {
            background: linear-gradient(135deg, rgba(255,245,230,0.9), rgba(255,240,250,0.85));
            border: 1px solid rgba(255,180,130,0.3);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            margin-top: 1.4rem;
            max-width: 600px;
        }

        .mb-journal-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: #b07040;
            margin-bottom: 0.35rem;
        }

        .mb-journal-text {
            font-size: 1rem;
            color: var(--mb-ink);
            line-height: 1.55;
            font-style: italic;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_theme()

def _current_risk() -> str:
    last = st.session_state.get("last_state")
    if last and hasattr(last, "safety") and isinstance(last.safety, dict):
        return last.safety.get("risk_level", "low")
    return "low"


def _inject_pet() -> None:
    risk = _current_risk()
    has_chat = bool(st.session_state.get("chat_log"))
    mood_class = {"high": "mb-pet-worried", "medium": "", "low": "mb-pet-happy"}.get(risk, "")
    bounce_class = "mb-pet-bounce" if has_chat else ""
    heart = '<div class="mb-heart">💗</div>' if risk == "low" and has_chat else ""
    bubbles = {
        "high": ("I'm right here with you 🤍", "You're not alone in this ✨", "It's okay to reach out 🌿"),
        "medium": ("You've got this 🌸", "I'm here with you ✨", "One step at a time 🍃"),
        "low": ("You're doing great 🌟", "I'm here with you ✨", "Keep going 🌸"),
    }.get(risk, ("You've got this 🌸", "I'm here with you ✨", "One step at a time 🍃"))
    st.markdown(
        f"""
        <div class="mb-pet-wrap {mood_class} {bounce_class}">
          {heart}
          <div class="mb-bubble mb-bubble-1">{bubbles[0]}</div>
          <div class="mb-bubble mb-bubble-2">{bubbles[1]}</div>
          <div class="mb-bubble mb-bubble-3">{bubbles[2]}</div>
          <div class="mb-cat-head">
            <div class="mb-ear-inner-l"></div>
            <div class="mb-ear-inner-r"></div>
            <div class="mb-eye-l"></div>
            <div class="mb-eye-r"></div>
            <div class="mb-nose"></div>
            <div class="mb-cheek-l"></div>
            <div class="mb-cheek-r"></div>
          </div>
          <div class="mb-cat-body-wrap">
            <div class="mb-cat-body">
              <div class="mb-paw-l"></div>
              <div class="mb-paw-r"></div>
            </div>
            <div class="mb-tail"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _inject_orbs() -> None:
    risk = _current_risk()
    if risk == "high":
        orb1 = "radial-gradient(circle, #ffb08080, #ff6f7d50)"
        orb2 = "radial-gradient(circle, #ffb89c70, #ff9b7350)"
        orb3 = "radial-gradient(circle, #ffd18860, #ffb58440)"
        orb4 = "radial-gradient(circle, #ff9b7360, #ffb08050)"
    else:
        orb1 = "radial-gradient(circle, #ffd18880, #ff9b7340)"
        orb2 = "radial-gradient(circle, #b89cff60, #79b8ff40)"
        orb3 = "radial-gradient(circle, #85dcb870, #ffd16650)"
        orb4 = "radial-gradient(circle, #ff6f7d50, #b89cff40)"
    st.markdown(
        f"""
        <div class="mb-orb mb-orb-1" style="background:{orb1}"></div>
        <div class="mb-orb mb-orb-2" style="background:{orb2}"></div>
        <div class="mb-orb mb-orb-3" style="background:{orb3}"></div>
        <div class="mb-orb mb-orb-4" style="background:{orb4}"></div>
        """,
        unsafe_allow_html=True,
    )


_inject_pet()

_inject_orbs()

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

STARTER_PROMPTS = (
    ("😮‍💨", "I feel overwhelmed and don't know where to start."),
    ("😶", "I've been really hard on myself lately."),
    ("😴", "I can't sleep — my thoughts won't stop."),
    ("💬", "Something happened and I need to talk it through."),
)

RISK_COLORS = {"low": "low", "medium": "medium", "high": "high"}


RESPONSE_MODES = {
    "Single-Agent Baseline": {
        "engine": "single_agent_baseline",
        "strict": False,
        "hint": "One-step LLM baseline without retrieval, memory, reflection, or explicit safety trace.",
    },
    "MindBridge Multi-Agent": {
        "engine": "multi_agent_pipeline",
        "strict": True,
        "hint": "Full MindBridge pipeline with analysis, retrieval, memory, reflection, and safety checks.",
    },
}

LEGACY_MODE_ALIASES = {
    "Light Assistant (Baseline / Single-Agent)": "Single-Agent Baseline",
    "Light Assistant": "MindBridge Multi-Agent",
    "CBT-Informed": "MindBridge Multi-Agent",
    "Strict Therapy (Deeper Thinking / Multi-Agent)": "MindBridge Multi-Agent",
    "Strict Therapy": "MindBridge Multi-Agent",
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


def _save_visible_mode_session() -> None:
    current_mode = st.session_state.get("response_mode")
    if not current_mode or "runner" not in st.session_state:
        return
    mode_sessions = st.session_state.setdefault("mode_sessions", {})
    mode_sessions[current_mode] = {
        "runner": st.session_state.runner,
        "runner_kind": st.session_state.get("runner_kind", ""),
        "chat_log": list(st.session_state.get("chat_log", [])),
        "profile_history": list(st.session_state.get("profile_history", [])),
        "last_state": st.session_state.get("last_state"),
        "selected_profile_session_id": st.session_state.get(
            "selected_profile_session_id"
        ),
    }


def _resume_profile_session(session_id: str) -> None:
    runner = st.session_state.get("runner")
    if (
        not session_id
        or st.session_state.get("runner_kind") != "multi_agent_pipeline"
        or runner is None
        or not hasattr(runner, "get_session_turns_snapshot")
    ):
        return

    turns = runner.get_session_turns_snapshot(session_id, limit_turns=80)
    if not turns:
        return

    # Continue future turns inside the selected saved session.
    if hasattr(runner, "_runtime_session_id"):
        runner._runtime_session_id = session_id

    st.session_state.chat_log = turns
    st.session_state.last_state = None
    st.session_state.selected_profile_session_id = None
    if hasattr(runner, "get_session_history_snapshot"):
        st.session_state.profile_history = runner.get_session_history_snapshot(
            limit_sessions=50,
            exclude_session_id=session_id,
        )
    _save_visible_mode_session()


def _boot_session(
    mode_label: str | None = None,
    profile_id: str | None = None,
    hydrate_from_persistent: bool = True,
    fresh_chat: bool = False,
) -> None:
    settings = get_settings()
    settings.validate()

    current_mode = st.session_state.get("response_mode")
    current_profile = st.session_state.get("profile_id", "default")
    profile_changed = profile_id is not None and profile_id != current_profile
    if profile_changed:
        st.session_state.mode_sessions = {}
    elif current_mode and "runner" in st.session_state:
        _save_visible_mode_session()

    selected_mode = mode_label or st.session_state.get(
        "response_mode",
        "MindBridge Multi-Agent",
    )
    selected_mode = LEGACY_MODE_ALIASES.get(selected_mode, selected_mode)
    if selected_mode not in RESPONSE_MODES:
        selected_mode = "MindBridge Multi-Agent"
    selected_profile = (profile_id or st.session_state.get("profile_id", "default")).strip() or "default"
    mode_cfg = RESPONSE_MODES[selected_mode]
    settings.therapist_flow_strict = bool(mode_cfg["strict"])

    mode_sessions = st.session_state.setdefault("mode_sessions", {})
    cached_session = mode_sessions.get(selected_mode)
    if cached_session and not fresh_chat and not profile_changed:
        runner = cached_session["runner"]
        chat_log = list(cached_session.get("chat_log", []))
        profile_history = list(cached_session.get("profile_history", []))
        last_state = cached_session.get("last_state")
        selected_profile_session_id = cached_session.get("selected_profile_session_id")
    elif mode_cfg["engine"] == "single_agent_baseline":
        runner = SingleAgentBaseline(settings)
        chat_log = []
        profile_history = []
        last_state = None
        selected_profile_session_id = None
    else:
        runner = MindBridgePipeline(settings, profile_id=selected_profile)
        if hydrate_from_persistent:
            chat_log = runner.get_active_session_turns_snapshot(limit_turns=80)
        else:
            # Start a fresh visible chat while preserving saved profile history
            # so Previous Chats can still show prior sessions.
            runner.start_new_session()
            chat_log = []
        profile_history = runner.get_session_history_snapshot(
            limit_sessions=50,
            exclude_session_id=runner.runtime_session_id,
        )
        last_state = None
        selected_profile_session_id = None

    st.session_state.response_mode = selected_mode
    st.session_state.profile_id = selected_profile
    st.session_state.runner_kind = mode_cfg["engine"]
    st.session_state.runner = runner
    st.session_state.chat_log = chat_log
    st.session_state.profile_history = profile_history
    st.session_state.last_state = last_state
    st.session_state.journal_prompt = random.choice(JOURNAL_PROMPTS)
    st.session_state.selected_profile_session_id = selected_profile_session_id

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
    with st.chat_message("user"):
        st.write(clean_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            '<div class="mb-thinking">'
            '<span class="mb-t-dot"></span>'
            '<span class="mb-t-dot"></span>'
            '<span class="mb-t-dot"></span>'
            '</div>',
            unsafe_allow_html=True,
        )
        result = st.session_state.runner.run(clean_input)
        placeholder.empty()
        streamed_response = st.write_stream(_stream_text(result.final_response))

        strategy_route = getattr(result, "strategy_route", {}) or {}
        safety = getattr(result, "safety", {}) or {}
        route = strategy_route.get("scenario", "single_agent_baseline")
        risk = safety.get("risk_level", "n/a")
        badges = ""
        if risk and risk != "n/a":
            cls = RISK_COLORS.get(risk, "low")
            badges += f'<span class="mb-badge mb-badge-{cls}">risk: {risk}</span>'
        if route and route not in ("single_agent_baseline", "n/a", ""):
            badges += f'<span class="mb-badge mb-badge-route">{route}</span>'
        if badges:
            st.markdown(badges, unsafe_allow_html=True)

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
    profile_input = st.text_input(
        "Profile ID",
        value=st.session_state.get("profile_id", "default"),
        help="Use the same Profile ID to load previous memory next time.",
    ).strip() or "default"
    if profile_input != st.session_state.get("profile_id", "default"):
        _boot_session(
            st.session_state.response_mode,
            profile_input,
            hydrate_from_persistent=False,
            fresh_chat=True,
        )
        st.rerun()

    mode_options = list(RESPONSE_MODES.keys())
    current_mode = LEGACY_MODE_ALIASES.get(
        st.session_state.get("response_mode", "MindBridge Multi-Agent"),
        st.session_state.get("response_mode", "MindBridge Multi-Agent"),
    )
    if current_mode not in mode_options:
        current_mode = "MindBridge Multi-Agent"
        st.session_state.response_mode = current_mode
    selected_mode = st.radio(
        "Response Mode",
        options=mode_options,
        index=mode_options.index(current_mode),
    )
    st.caption(RESPONSE_MODES[selected_mode]["hint"])
    if selected_mode != st.session_state.response_mode:
        # Each system keeps its own visible chat, so responses from different
        # engines do not appear in the same thread.
        _boot_session(
            selected_mode,
            st.session_state.profile_id,
            hydrate_from_persistent=False,
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
                fresh_chat=True,
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
            fresh_chat=True,
        )
        st.rerun()
    show_debug = st.checkbox("Show Debug Panel", value=False)

    st.divider()
    st.markdown("**Previous Chats**")
    profile_history = st.session_state.get("profile_history", [])
    _HISTORY_ICONS = ["🌸", "🍃", "✨", "🌿", "💬", "🌙", "🌤️", "🫧", "🕊️", "🌱", "💛", "🫂"]
    if not profile_history:
        st.caption("No previous chats yet — start a new one 🌱")
    else:
        visible_history = profile_history[-12:]
        for local_idx, session_item in enumerate(reversed(visible_history), start=1):
            session_id = str(session_item.get("session_id", "")).strip()
            user_text = _one_line_preview(session_item.get("title", ""), limit=36)
            icon = _HISTORY_ICONS[(local_idx - 1) % len(_HISTORY_ICONS)]
            label = f"{icon}  {user_text or '(empty)'}"
            if st.button(
                label,
                key=f"profile_history_item_{session_id or local_idx}",
                use_container_width=True,
            ):
                _resume_profile_session(session_id)
                st.rerun()

if st.session_state.chat_log:
    for turn in st.session_state.chat_log:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.markdown(turn["assistant"])
            risk = turn.get("risk", "")
            route = turn.get("route", "")
            badges = ""
            if risk and risk != "n/a":
                cls = RISK_COLORS.get(risk, "low")
                badges += f'<span class="mb-badge mb-badge-{cls}">risk: {risk}</span>'
            if route and route not in ("single_agent_baseline", "n/a", ""):
                badges += f'<span class="mb-badge mb-badge-route">{route}</span>'
            if badges:
                st.markdown(badges, unsafe_allow_html=True)
else:
    # Empty state — show starter cards and journal prompt
    journal_prompt = st.session_state.get("journal_prompt", JOURNAL_PROMPTS[0])
    st.markdown(
        f'<div class="mb-journal-card">'
        f'<div class="mb-journal-label">✦ Reflection prompt</div>'
        f'<div class="mb-journal-text">{journal_prompt}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="margin-top:1.8rem;margin-bottom:0.5rem;font-size:0.88rem;'
        'color:#72768b;font-weight:600;letter-spacing:0.04em;">'
        'OR START WITH'
        '<span class="mb-dot" style="background:#ff9b73;margin-left:10px;"></span>'
        '<span class="mb-dot" style="background:#85dcb8;"></span>'
        '<span class="mb-dot" style="background:#b89cff;"></span>'
        '</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for idx, (icon, text) in enumerate(STARTER_PROMPTS):
        with cols[idx % 2]:
            if st.button(
                f"{icon}  {text}",
                key=f"starter_{idx}",
                use_container_width=True,
            ):
                _run_turn(text)
                st.rerun()

user_input = st.chat_input("Share what is on your mind.")
if user_input and user_input.strip():
    _run_turn(user_input.strip())

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
