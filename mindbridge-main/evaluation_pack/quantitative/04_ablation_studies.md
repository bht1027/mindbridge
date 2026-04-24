# 4.3.4 Ablation Studies

We perform ablation studies to understand which parts of the multi-agent pipeline contribute most to response quality. Starting from the full system, we remove one component at a time and compare the resulting performance.

## Ablation variants

- **no_critic**: removes reflective critique.
- **no_reviser**: removes the revision step after critique.
- **no_safety**: removes the dedicated safety agent.
- **no_retrieval**: removes support-knowledge retrieval.

## Why this matters

Ablation analysis helps move beyond the claim that “the full system works.” Instead, it tests which modules are actually useful. This is especially valuable in a multi-agent system, where architectural complexity should be justified by measurable gains.

## Fill-in table

| Variant | Main removed component | Expected impact |
|---|---|---|
| no_critic | reflection critic | lower response refinement |
| no_reviser | revision pass | weaker final phrasing |
| no_safety | safety agent | lower safety robustness |
| no_retrieval | support KB retrieval | less grounded practical advice |

## Fill-in results table

| Variant | Overall Delta vs Full | Empathy Delta | Helpfulness Delta | Safety Delta | Naturalness Delta |
|---|---:|---:|---:|---:|---:|
| no_critic | [ ] | [ ] | [ ] | [ ] | [ ] |
| no_reviser | [ ] | [ ] | [ ] | [ ] | [ ] |
| no_safety | [ ] | [ ] | [ ] | [ ] | [ ] |
| no_retrieval | [ ] | [ ] | [ ] | [ ] | [ ] |

## Interpretation template

Our ablation results help identify which modules are most important. If removing retrieval reduces helpfulness, it suggests that lightweight grounding improves practical relevance. If removing the safety module reduces safety scores, it confirms that safety should remain a dedicated component rather than an implicit side effect of prompting. If removing the critic or reviser lowers naturalness or overall quality, it supports the value of self-reflection in supportive response generation.
