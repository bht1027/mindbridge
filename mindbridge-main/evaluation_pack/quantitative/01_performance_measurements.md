# 4.3.1 Performance Measurements

This section reports quantitative performance measurements for the MindBridge supportive-dialogue system. We evaluate both the single-agent baseline and the multi-agent pipeline across the same evaluation cases. For each system, we report average runtime, response length, and judge-based quality metrics including empathy, helpfulness, safety, and naturalness.

## Metrics

- **Runtime**: average response generation time per case.
- **Response length**: average number of response characters.
- **Empathy**: judge score from 1 to 5.
- **Helpfulness**: judge score from 1 to 5.
- **Safety**: judge score from 1 to 5.
- **Naturalness**: judge score from 1 to 5.
- **Overall quality**: mean of the four judge dimensions.

## Suggested write-up

We measure both efficiency and response quality. Efficiency matters because supportive dialogue systems should be responsive in interactive settings, while quality matters because the system must remain empathetic, safe, and practically useful. We therefore report runtime and response length as lightweight system metrics, and use an LLM judge to score the content quality of generated responses.

Across the evaluation set, the full multi-agent pipeline is expected to be slower than the single-agent baseline because it includes multiple specialist stages, reflection, and safety checking. However, this extra computation is intended to improve response quality by producing more grounded, structured, and safer outputs.

## Fill-in table

| System | Avg Runtime (s) | Avg Response Chars | Empathy | Helpfulness | Safety | Naturalness | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single-agent baseline | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Pipeline full | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Pipeline no_critic | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Pipeline no_reviser | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Pipeline no_safety | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Pipeline no_retrieval | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

## Interpretation paragraph template

The results show that the full MindBridge pipeline achieves stronger response quality than the single-agent baseline on most content-focused metrics, especially empathy, helpfulness, and safety. Although the full pipeline requires more runtime than the baseline, the increase is expected because the architecture performs staged analysis, retrieval, coordination, reflection, and safety checking. In a course-project setting, this tradeoff is acceptable because the main objective is to improve response quality rather than minimize latency.
