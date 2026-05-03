# Proposal Alignment Matrix

This document maps proposal commitments to implemented artifacts and evidence files for grading.

Important:
- Replace the text in the `Proposal Commitment` column with exact wording from your submitted proposal PDF.
- Add screenshots in `docs/screenshots/` and update the links in the last column.

| # | Proposal Commitment (paste exact sentence from proposal) | Final Implementation | Evidence File(s) | Screenshot / Demo Evidence |
| --- | --- | --- | --- | --- |
| 1 | Multi-agent architecture for supportive dialogue | Implemented analyzer + empathy + strategy + safety + coordinator + critic + reviser + final checker pipeline | `pipeline.py`, `agents.py`, `prompts.py`, `schemas.py` | Add architecture screenshot from demo/debug panel |
| 2 | Safety as explicit module, not implicit style | Dedicated `Safety Agent`, keyword high-risk override, merged `safety_trace`, final safety checker | `pipeline.py` (`_keyword_safety_scan`, `_merge_safety_outputs`), `prompts.py` (`SAFETY_PROMPT`, `FINAL_CHECKER_PROMPT`) | Add high-risk case screenshot + safety trace screenshot |
| 3 | Scenario-aware strategy routing | Rule-based scenario routing and focus areas (`sleep_support`, `conflict_resolution`, `job_search_support`, etc.) | `pipeline.py` (`STRATEGY_PLAYBOOK`, `_build_strategy_route`) | Add side-by-side routing examples for different inputs |
| 4 | Short-term memory and multi-turn continuity | Conversation memory with `memory_context` and `user_state`, updated after each turn | `pipeline.py` (`_memory_context`, `_update_memory`), `schemas.py` | Add two-turn conversation screenshot showing adaptation |
| 5 | Retrieval-grounded support knowledge | Retriever over support KB with top-k hits and retrieval metadata | `retriever.py`, `data/support_kb.json`, `pipeline.py` (`_run_retrieval_stage`) | Add debug screenshot showing retrieved `hit_ids` |
| 6 | Baseline comparison | Single-agent baseline and parallel evaluation against pipeline variants | `baseline.py`, `evaluate.py`, `README.md` | Add table screenshot from `evaluation_report.md` |
| 7 | Ablation studies | Supports `no_critic`, `no_reviser`, `no_safety`, `no_retrieval` modes | `run_modes.py`, `app.py`, `evaluate.py` | Add ablation comparison screenshot from report |
| 8 | Statistical significance analysis | Paired bootstrap confidence intervals for mean deltas vs baseline | `metrics.py`, `evaluate.py` (`_paired_quality_deltas`) | Add CI table screenshot from report |
| 9 | Qualitative analysis and error analysis | Auto-generated case studies (improvements/regressions) + runtime/judge error analysis + high-risk mismatch checks | `evaluate.py` (`_qualitative_case_studies`, `_error_analysis`), `evaluation_report.md` | Add case-study section screenshot |
| 10 | Interactive live demo | Streamlit chat UI with quick scenarios, breathing tool, journal prompts, and optional debug panel | `demo_streamlit.py`, `scripts/run_demo.sh` | Add demo UI screenshot + one live interaction screenshot |

## TA Feedback Traceability (Optional but Recommended)

| TA Feedback Item | What We Changed | Evidence |
| --- | --- | --- |
| Example: responses were too templated | Added conversational rewrite rules and failed-coping follow-up flow | `prompts.py`, `pipeline.py`, demo screenshots |
| Example: safety should be explicit | Exposed `safety_trace` in debug and hardened safety routing | `pipeline.py`, `demo_streamlit.py` |

## Submission Checklist

- [ ] Proposal commitments replaced with exact proposal wording
- [ ] Every row includes at least one concrete file path
- [ ] Screenshot links added and verified
- [ ] Alignment reviewed against final presentation slides
