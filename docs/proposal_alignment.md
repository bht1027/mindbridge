# Proposal Alignment Matrix

This document maps the implemented MindBridge system to the project proposal direction and the course rubric. It is intended as a grading aid for the final report, presentation, and demo.

## Implementation Alignment

| # | Proposal / Rubric Commitment | Current Implementation | Evidence File(s) | Demo / Report Evidence To Capture |
| --- | --- | --- | --- | --- |
| 1 | Build a GenAI system for supportive dialogue rather than a generic chatbot. | MindBridge focuses on emotional-support conversations with risk-aware response generation, memory, and strategy routing. | `README.md`, `pipeline.py`, `prompts.py`, `demo_streamlit.py` | Screenshot of the Streamlit chat flow. |
| 2 | Use CBT as a theoretical inspiration without claiming to provide therapy. | The system is framed as CBT-informed support: thought-emotion-action loops, evidence checking, balanced reframing, and tiny behavior steps. It explicitly avoids diagnosis or treatment claims. | `README.md`, `prompts.py`, `data/support_kb.json`, `demo_streamlit.py` | Report paragraph explaining "CBT-informed, non-clinical" scope. |
| 3 | Use a multi-agent architecture with interpretable intermediate stages. | The pipeline includes input analysis, retrieval, empathy, strategy, safety, coordination, reflection, revision, and final safety checking. | `pipeline.py`, `agents.py`, `prompts.py`, `schemas.py` | Debug panel screenshot showing analyzer, strategy route, safety trace, and final state. |
| 4 | Make safety explicit, not just a tone instruction. | The system has a dedicated safety agent, keyword override scan, merged `safety_trace`, high-risk handling, and final safety checker. | `pipeline.py`, `prompts.py`, `schemas.py` | High-risk demo case plus safety trace screenshot. |
| 5 | Adapt responses to the user's scenario. | Rule-based strategy routing maps inputs to scenarios such as academic overload, sleep stress, conflict, job search, loneliness, panic-like symptoms, and crisis signals. | `pipeline.py`, `data/support_kb.json` | Side-by-side demo examples with different routes. |
| 6 | Preserve short-term and profile-level continuity. | Runtime memory and optional persistent profile memory store session turns, previous strategies, failed methods, and active session history. | `pipeline.py`, `schemas.py`, `data/user_profiles.json`, `demo_streamlit.py` | Two-turn or restart demo showing remembered context. |
| 7 | Ground responses with lightweight support knowledge. | Hybrid retrieval combines lexical overlap, TF-IDF vector similarity, reciprocal-rank fusion, and interpretable retrieval metadata over CBT-informed support snippets. | `retriever.py`, `data/support_kb.json`, `pipeline.py` | Debug screenshot showing retrieved hit IDs and retrieval scores. |
| 8 | Compare against a simpler baseline. | `SingleAgentBaseline` provides a one-step baseline for CLI and batch evaluation. | `baseline.py`, `app.py`, `evaluate.py` | Evaluation table comparing baseline vs full pipeline. |
| 9 | Run ablation studies for major components. | Evaluation supports `full`, `no_critic`, `no_reviser`, `no_safety`, and `no_retrieval`. | `run_modes.py`, `evaluate.py`, `app.py` | Ablation table from `evaluation_report.md`. |
| 10 | Include quantitative metrics and statistical significance. | Evaluation records runtime, response length, LLM-judge scores, paired bootstrap confidence intervals, pairwise preference win rates, preference-pair artifacts, and high-risk safety recall. | `evaluate.py`, `metrics.py`, `judge.py` | Quality summary, 95% CI table, pairwise preference section, and safety recall from generated report. |
| 11 | Include qualitative analysis and error analysis. | Evaluation report generator selects top improvements/regressions and reports runtime errors, judge errors, and high-risk mismatches. | `evaluate.py`, `evaluation_pack/` | Case study and error analysis sections in final report. |
| 12 | Provide an interactive live demo. | Streamlit demo supports chat, mode switching, profile memory, previous chat preview, quick-start prompts, and optional debug panel. | `demo_streamlit.py`, `scripts/run_demo.sh` | Live demo recording or screenshots. |
| 13 | Make the repository reproducible. | Setup instructions, dependency file, `.env.example`, sample data, helper scripts, and local checks are included. | `README.md`, `requirements.txt`, `.env.example`, `scripts/run_checks.sh`, `tests/test_core.py` | Terminal screenshot of `scripts/run_checks.sh` passing. |

## Rubric Coverage Summary

| Rubric Area | Status | Notes |
| --- | --- | --- |
| Proposal relevance | Covered by implementation | Final report should explicitly quote the original proposal goal and connect it to this matrix. |
| Presentation: problem statement | Supported | Needs slides explaining supportive dialogue, safety, and target users. |
| Presentation: major contributions | Supported | Use pipeline architecture, safety trace, memory, retrieval, and ablation support as contributions. |
| Presentation: evaluation | Supported by code | Requires generated `evaluation_report.md` tables before final submission, preferably with `--pairwise-judge` enabled. |
| Final report | Not a code artifact | Report still needs literature review, methodology, results, limitations, and citations. |
| Project demo | Covered | Streamlit app is functional; prepare a controlled demo script with low/medium/high-risk examples. |
| GitHub repo quality | Mostly covered | `.env.example`, README, scripts, tests, and sample data are now present. |
| Reproducibility | Mostly covered | `scripts/run_checks.sh` is local/offline; full evaluation still requires OpenAI API access. |

## TA Feedback Traceability

| Feedback / Concern | Response in Current Project | Evidence |
| --- | --- | --- |
| Responses can sound too templated. | Prompt rules and final renderer were adjusted to reduce repeated therapy-style phrasing and make first-turn replies shorter. | `prompts.py`, `pipeline.py` |
| The system needs a clearer theoretical basis. | Added CBT-informed support framing and knowledge snippets while avoiding clinical treatment claims. | `README.md`, `prompts.py`, `data/support_kb.json`, `demo_streamlit.py` |
| Safety should be explicit and inspectable. | Safety is implemented as a separate stage and surfaced through `safety_trace` and debug output. | `pipeline.py`, `demo_streamlit.py`, `schemas.py` |
| Evaluation should include baseline and ablations. | Batch evaluation compares baseline, full pipeline, and four ablation modes. | `evaluate.py`, `run_modes.py` |
| Demo should be easy to run. | Streamlit script and `scripts/run_demo.sh` provide a one-command demo path. | `demo_streamlit.py`, `scripts/run_demo.sh`, `README.md` |

## Remaining Submission Tasks

- Generate final `evaluation_results.json`, `evaluation_report.md`, and `data/preference_pairs.json` using a full evaluation run with pairwise judging.
- Add screenshots under `docs/screenshots/` after the final UI and evaluation report are stable.
- Copy the final report and presentation evidence back into this matrix where useful.
