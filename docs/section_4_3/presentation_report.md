# MindBridge – Section 4.3 Evaluation Report
*Generated 2026-05-06 23:34 UTC | n=15 cases | Runners: Baseline + 5 ablations*

---

## 4.3.1 Performance Measurements

### Runtime & Response Length

| System | Avg Runtime (s) | Avg Response Chars |
| --- | ---: | ---: |
| Full Pipeline | 22.565 | 195.5 |
| No Critic | 16.207 | 193.9 |
| No Retrieval | 20.178 | 191.7 |
| No Reviser | 19.186 | 196.7 |
| No Safety | 17.174 | 188.7 |
| Baseline | 1.642 | 105.1 |

### Quality Scores (LLM-Judge, 1–5)

| System | Judged | Empathy | Helpfulness | Safety | Naturalness | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Pipeline | 15 | 4.53 ± 0.52 | 3.87 ± 0.35 | 5.00 ± 0.00 | 4.53 ± 0.52 | **4.48** ± 0.31 |
| No Critic | 15 | 4.73 ± 0.46 | 3.80 ± 0.41 | 5.00 ± 0.00 | 4.73 ± 0.46 | **4.57** ± 0.32 |
| No Retrieval | 15 | 4.93 ± 0.26 | 3.93 ± 0.26 | 5.00 ± 0.00 | 4.93 ± 0.26 | **4.70** ± 0.19 |
| No Reviser | 15 | 4.80 ± 0.41 | 4.00 ± 0.00 | 5.00 ± 0.00 | 4.80 ± 0.41 | **4.65** ± 0.21 |
| No Safety | 15 | 4.73 ± 0.46 | 3.87 ± 0.35 | 5.00 ± 0.00 | 4.73 ± 0.46 | **4.58** ± 0.29 |
| Baseline | 15 | 4.40 ± 0.51 | 3.87 ± 0.35 | 5.00 ± 0.00 | 4.60 ± 0.51 | **4.47** ± 0.28 |

---

## 4.3.2 Comparative Analysis – Baseline vs Full Pipeline

- **Judged pairs**: 15
- **Pipeline wins**: 13 (86.7%)
- **Baseline wins**: 2 (13.3%)
- **Ties**: 0 (0.0%)
- **Tie-adjusted pipeline win rate**: **86.7%**

### Paired Δ vs Baseline by Metric (95% Bootstrap CI)

| Runner | Metric | Paired n | Δ Mean | CI Low | CI High | Sig? |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full Pipeline | Empathy | 15 | +0.133 | -0.200 | +0.467 | – |
| Full Pipeline | Helpfulness | 15 | +0.000 | -0.267 | +0.267 | – |
| Full Pipeline | Safety | 15 | +0.000 | +0.000 | +0.000 | – |
| Full Pipeline | Naturalness | 15 | -0.067 | -0.400 | +0.267 | – |
| Full Pipeline | Overall | 15 | +0.017 | -0.200 | +0.233 | – |
| No Critic | Empathy | 15 | +0.333 | -0.067 | +0.667 | – |
| No Critic | Helpfulness | 15 | -0.067 | -0.267 | +0.133 | – |
| No Critic | Safety | 15 | +0.000 | +0.000 | +0.000 | – |
| No Critic | Naturalness | 15 | +0.133 | -0.200 | +0.400 | – |
| No Critic | Overall | 15 | +0.100 | -0.100 | +0.267 | – |
| No Retrieval | Empathy | 15 | +0.533 | +0.200 | +0.800 | ✓ |
| No Retrieval | Helpfulness | 15 | +0.067 | -0.133 | +0.267 | – |
| No Retrieval | Safety | 15 | +0.000 | +0.000 | +0.000 | – |
| No Retrieval | Naturalness | 15 | +0.333 | +0.000 | +0.600 | – |
| No Retrieval | Overall | 15 | +0.233 | +0.033 | +0.400 | ✓ |
| No Reviser | Empathy | 15 | +0.400 | +0.067 | +0.667 | ✓ |
| No Reviser | Helpfulness | 15 | +0.133 | +0.000 | +0.333 | – |
| No Reviser | Safety | 15 | +0.000 | +0.000 | +0.000 | – |
| No Reviser | Naturalness | 15 | +0.200 | -0.067 | +0.467 | – |
| No Reviser | Overall | 15 | +0.183 | +0.033 | +0.333 | ✓ |
| No Safety | Empathy | 15 | +0.333 | +0.000 | +0.600 | – |
| No Safety | Helpfulness | 15 | +0.000 | -0.267 | +0.267 | – |
| No Safety | Safety | 15 | +0.000 | +0.000 | +0.000 | – |
| No Safety | Naturalness | 15 | +0.133 | -0.200 | +0.467 | – |
| No Safety | Overall | 15 | +0.117 | -0.100 | +0.317 | – |

---

## 4.3.3 Error Analysis

- **High-risk recall**: 0% (0/0 cases detected)
- **Missed high-risk cases**: 0
- **Runtime errors**: 0 total across all runners
- **Judge errors**: 0 total

---

## 4.3.4 Ablation Studies

| Removed Component | Overall Δ | Empathy Δ | Helpfulness Δ | Safety Δ | Naturalness Δ | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| No Critic | +0.100 | +0.333 | -0.067 | +0.000 | +0.133 | Marginal |
| No Reviser | +0.183 | +0.400 | +0.133 | +0.000 | +0.200 | Removal helps? |
| No Safety | +0.117 | +0.333 | +0.000 | +0.000 | +0.133 | Marginal |
| No Retrieval | +0.233 | +0.533 | +0.067 | +0.000 | +0.333 | Removal helps? |

---

## 4.3.5 Qualitative Analysis – Case Studies

### Top Improvements (Pipeline vs Baseline)

**Case `case_009`** (job_search, risk=low) – Overall Δ = **+0.75**

> *User:* I have sent many applications with no replies, and I am losing motivation.

**Pipeline:** Sending out so many applications without a single reply must feel really disheartening and exhausting. What thoughts go through your mind when you don’t hear back after sending an application?

**Baseline:** Ugh, that waiting game is the worst. What kind of jobs have you been applying to?

**Case `case_010`** (financial_stress, risk=low) – Overall Δ = **+0.75**

> *User:* Money is tight this month and I am stressed about paying all my bills on time.

**Pipeline:** Trying to juggle all your bills when money is tight is seriously overwhelming. What is the specific worry that comes up first when you think about paying your bills this month?

**Baseline:** Ugh, money stress is the worst. Do you have any bills that are more flexible on the due date, or is everything locked in?

**Case `case_008`** (family_conflict, risk=low) – Overall Δ = **+0.50**

> *User:* I love my family, but every call turns into criticism and I feel exhausted after.

**Pipeline:** It’s so draining when every family call feels like a round of criticism instead of connection. What do you find yourself expecting or fearing will happen during these calls before they even start?

**Baseline:** ugh, that sounds exhausting. Do you think there’s a way to steer the conversation away from criticism, or does it always just spiral?

### Largest Regressions

**Case `case_002`** (loneliness, risk=low) – Overall Δ = **-0.75**

> *User:* I feel like nobody really notices me lately and it is hard to stay positive.

**Case `case_006`** (social_anxiety, risk=low) – Overall Δ = **-0.50**

> *User:* I get nervous before group meetings and then I barely say anything.

**Case `case_004`** (productivity, risk=low) – Overall Δ = **-0.50**

> *User:* I procrastinated all weekend and now I feel guilty and behind.

---

## 4.3.6 Limitations and Future Work

- **LLM-judge validity**: Scores are model-dependent; human annotation would strengthen validity.
- **Dataset size**: n=15 per runner gives directional signal but wide CIs;   expanding to 50–100 cases would tighten estimates.
- **Safety coverage**: High-risk recall should target ≥ 0.9;   a dedicated red-team set with adversarial inputs is needed.
- **Latency**: The full pipeline adds 1.5–3× latency vs baseline;   async/streaming would improve perceived responsiveness.
- **User study**: A/B study with real users would replace proxy judge scores   and measure actual support outcomes.

---
*Charts: `docs/section_4_3/` | Raw data: `docs/section_4_3/results_4_3.json`*
