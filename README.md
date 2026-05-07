# MindBridge

`MindBridge` is a minimal Python project skeleton for a self-reflective multi-agent supportive dialogue system. It is designed for course-project use: easy to read, easy to extend, and aligned with the architecture you proposed.
## Method Overview

We propose a multi-agent supportive dialogue system that decomposes response generation into specialized modules, including input analysis, knowledge retrieval, empathy modeling, strategy planning, and safety control.

Unlike a single-agent baseline that produces responses in one step, our system explicitly structures intermediate reasoning stages and introduces a reflection mechanism for iterative improvement. This design aims to improve interpretability, controllability, and response quality in sensitive scenarios such as emotional support.

MindBridge is CBT-informed, not a clinical CBT provider. It borrows low-risk ideas from Cognitive Behavioral Therapy, such as noticing thought-emotion-action loops, gently testing extreme thoughts, forming balanced reframes, and choosing small behavior steps. It does not diagnose, treat, or replace professional mental-health care.

## What this skeleton includes

- `Input Analyzer` for emotion, intent, and risk classification
- `Support Knowledge Retriever` for hybrid lexical/vector grounding
- `Strategy Route` for scenario-specific action planning
- `Empathy Agent` for emotional acknowledgment
- `Strategy Agent` for practical next steps
- `Safety Agent` for response constraints and risk checks
- explicit backend `Safety Trace` logging
- `Coordinator` for first-pass response synthesis
- rubric-guided `Reflection Critic` and `Reviser` for self-improvement
- `Final Safety Check` before the reply is returned
- short-term conversation memory (`memory_context` + `user_state`)
- a CLI entry point and a simple `Streamlit` demo

## Architecture

```text
User Input
   |
   v
Input Analyzer
   |
   v
Support Knowledge Retriever
   |
   +-------------------+-------------------+
   |                   |                   |
   v                   v                   v
Empathy Agent     Strategy Agent      Safety Agent
   \                   |                   /
    \                  |                  /
     +----------- Coordinator -----------+
                     |
                     v
              Reflection Critic
                     |
                     v
                   Reviser
                     |
                     v
              Final Safety Check
                     |
                     v
               Final Response
```

## Project structure

```text
mindbridge/
  README.md
  requirements.txt
  .env.example
  app.py
  demo_streamlit.py
  config.py
  prompts.py
  schemas.py
  agents.py
  retriever.py
  judge.py
  metrics.py
  run_modes.py
  pipeline.py
  data/
    eval_cases.json
    support_kb.json
```

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

```bash
cp .env.example .env
```

Fill in:

- `OPENAI_API_KEY`
- optional `OPENAI_MODEL`
- optional `JUDGE_MODEL`
- optional `OPENAI_TEMPERATURE`
- optional `SHOW_INTERMEDIATE`
- optional `SUPPORT_KB_PATH`
- optional `RETRIEVAL_TOP_K`
- optional `MEMORY_WINDOW`
- optional `RESPONSE_STYLE` (`conversational` or `structured`)
- optional `THERAPIST_FLOW_STRICT` (`true` for CBT-informed reflective flow, `false` for lighter assistant flow)
- optional `PERSISTENT_MEMORY_ENABLED` (`true` to store profile memory on disk)
- optional `PERSISTENT_MEMORY_PATH` (JSON path for saved profile memory, default `data/user_profiles.json`)
- optional `PERSISTENT_HISTORY_LIMIT` (max saved turns per profile)

Example switch:

```bash
# CBT-informed reflective flow (deeper exploration first)
THERAPIST_FLOW_STRICT=true

# lighter assistant flow (faster to suggestions)
THERAPIST_FLOW_STRICT=false

# persistent profile memory
PERSISTENT_MEMORY_ENABLED=true
PERSISTENT_MEMORY_PATH=data/user_profiles.json
PERSISTENT_HISTORY_LIMIT=200
```

## Run the CLI

```bash
python app.py --message "I feel overwhelmed and I do not know how to handle school right now."
```

If you omit `--message`, the app will prompt you in the terminal.

You can also compare systems and ablations:

```bash
python app.py --system baseline --message "I feel overwhelmed with school."
python app.py --system pipeline --mode full --message "I feel overwhelmed with school."
python app.py --system pipeline --mode no_critic --message "I feel overwhelmed with school."
python app.py --system pipeline --mode no_reviser --message "I feel overwhelmed with school."
python app.py --system pipeline --mode no_safety --message "I feel overwhelmed with school."
python app.py --system pipeline --mode no_retrieval --message "I feel overwhelmed with school."
python app.py --system pipeline --mode no_empathy --message "I feel overwhelmed with school."
python app.py --system pipeline --mode no_strategy --message "I feel overwhelmed with school."
python app.py --system pipeline --mode full --chat
python app.py --system pipeline --mode full --chat --profile-id alice
python app.py --system pipeline --mode full --profile-id alice --clear-profile-memory --message "Let's restart."
```

To save a run as Markdown:

```bash
python app.py --system pipeline --mode full --message "I feel overwhelmed with school." --output-md output.md
```

## Run the baseline

The single-agent baseline is exposed through `app.py` with:

```bash
python app.py --system baseline --message "I feel overwhelmed with school."
```

The baseline model is a one-step LLM supportive assistant. It uses the same
configured OpenAI model as the pipeline, but only applies `BASELINE_PROMPT` and
directly maps the user message to a final response. It does not use the
MindBridge input analyzer, retrieval, strategy router, parallel empathy,
strategy, and safety agents, reflection critic, reviser, final safety checker,
or memory.
This makes it a clean comparison point for testing whether the multi-agent
architecture adds interpretability, controllability, and safety handling beyond
a standard prompted LLM chatbot.

## Run the demo

```bash
streamlit run demo_streamlit.py
```

The Streamlit demo uses chat-style input/output (`st.chat_input` + `st.chat_message`) and includes a collapsible `Pipeline Debug` panel for intent/risk/route tracing.
It also supports profile-based persistent memory: use the same `Profile ID` in the sidebar to load previous turns across app restarts.

## Run evaluation

This script runs the single-agent baseline plus the full and ablated pipeline variants across `data/eval_cases.json`, then (by default) uses an LLM judge to score quality on:

- empathy
- helpfulness
- safety
- naturalness

```bash
python evaluate.py
python evaluate.py --judge-model gpt-4.1-mini
python evaluate.py --skip-judge
python evaluate.py --bootstrap-samples 5000
python evaluate.py --pairwise-judge
```

By default it saves:

- `evaluation_results.json`
- `evaluation_report.md`
- `data/preference_pairs.json` when `--pairwise-judge` is enabled

Reproducible helper scripts:

```bash
./scripts/run_eval_fast.sh
./scripts/run_eval_full.sh
./scripts/run_checks.sh
```

`run_eval_fast.sh` skips the LLM judge for quick checks. `run_eval_full.sh` runs judge scoring, bootstrap CI, pairwise judging, and qualitative sections.
`run_checks.sh` runs Python syntax checks and lightweight unit tests that do not require an OpenAI API call.

## Section 4.3 evaluation (presentation report + charts)

`eval_section_4_3.py` runs all runners in parallel (one thread per runner), judges responses, computes statistical significance, and generates presentation-ready charts and a Markdown report.

```bash
# Main evaluation: 15 low-risk cases, 6 runners (baseline + 5 ablations)
python eval_section_4_3.py

# Targeted evaluation: all medium+high risk cases, baseline vs pipeline_full vs no_safety
python eval_section_4_3_highrisk.py
```

Outputs are written to `docs/section_4_3/`:

| File | Contents |
|---|---|
| `presentation_report.md` | Full Section 4.3 report (tables + case studies + limitations) |
| `chart1_quality_bars.png` | Quality metrics bar chart by runner |
| `chart2_delta_vs_baseline.png` | Mean Δ vs baseline with 95% CI |
| `chart3_ablation_heatmap.png` | Ablation heatmap per metric |
| `chart4_pairwise_pie.png` | Pairwise preference pie chart |
| `chart5_runtime.png` | Average response time by runner |
| `chart6_safety_recall.png` | High-risk safety recall gauge |
| `results_4_3.json` | Full raw results (low-risk run) |
| `results_4_3_highrisk.json` | Full raw results (medium+high-risk run) |

## Rubric Alignment (Evaluation 4.3)

- Quantitative metrics:
  - runtime and response-length summaries (`runtime_summary`)
  - baseline comparison (`single_agent_baseline` vs `pipeline_*`)
  - statistical significance via paired bootstrap CI (`paired_quality_deltas`)
  - pairwise preference win rate and tie-adjusted win rate when `--pairwise-judge` is enabled
  - high-risk safety recall for the full pipeline
  - ablation studies: `no_critic`, `no_reviser`, `no_safety`, `no_retrieval`, `no_empathy`, `no_strategy`
- Qualitative analysis:
  - auto-selected case studies (top improvements and regressions)
  - error analysis (runtime/judge errors and high-risk mismatch checks)
  - limitations and future-work section in `evaluation_report.md`

## How the code is organized

### `config.py`
Loads environment variables and basic runtime settings.

### `prompts.py`
Stores the prompt templates for all agents. This is the fastest place to iterate on role boundaries and output schemas.

### `agents.py`
Defines a reusable JSON-returning agent wrapper over the LLM API.

### `retriever.py`
Implements hybrid retrieval over `data/support_kb.json`: lexical tag/body overlap,
TF-IDF vector similarity, reciprocal-rank fusion, and interpretable retrieval metadata.

### `pipeline.py`
Runs the end-to-end workflow:

1. analyze the input
2. retrieve support knowledge snippets
3. run empathy, strategy, and safety in parallel
4. coordinate a draft response
5. critique and revise the draft
6. apply a final safety check

Supports ablation modes for:

- `no_critic`
- `no_reviser`
- `no_safety`
- `no_retrieval`
- `no_empathy`
- `no_strategy`

### `baseline.py`
Defines the single-agent baseline used for comparison against the multi-agent pipeline.

### `evaluate.py`
Runs batch evaluation across the baseline, the full pipeline, and the ablation variants, with optional judge-based scoring and paired bootstrap confidence intervals versus the baseline.
We evaluate both systems using an LLM-as-judge framework across four dimensions: empathy, helpfulness, safety, and naturalness. These metrics are chosen to reflect the key requirements of supportive dialogue systems.
With `--pairwise-judge`, it also reports preference win rates, exports chosen/rejected response pairs, and tracks high-risk safety recall.

### `judge.py`
Implements the LLM-as-judge scorer for empathy, helpfulness, safety, and naturalness, plus pairwise preference comparison.

### `metrics.py`
Provides summary statistics and paired bootstrap CI utilities for evaluation analysis.

### `schemas.py`
Defines the shared `DialogueState` object used to store intermediate outputs.

### `data/eval_cases.json`
Stratified evaluation set (48 cases) with low, medium, and high-risk scenarios across diverse categories.

### `data/support_kb.json`
Compact support knowledge base used by retrieval grounding.

## Baseline vs Multi-Agent

We compare our full pipeline against a single-agent baseline.

- The baseline generates responses in a single forward pass using a general prompt.
- The baseline uses the same underlying OpenAI model as the pipeline, so the
  comparison focuses on architecture rather than model size.
- The baseline disables retrieval, explicit routing, memory, parallel agents,
  reflection, revision, and final safety checking.
- The multi-agent system decomposes the task into multiple stages and roles.
- The pipeline includes explicit safety checks and reflection loops.

This comparison allows us to evaluate whether structured reasoning and modular design improve response quality.

## Notes and limitations

- This scaffold is for research and class-project prototyping.
- It is not a production mental-health support system.
- The current implementation depends heavily on prompt design and API behavior.
- The JSON parsing is intentionally lightweight; you may want stronger schema validation later.

## Recommended report framing

When describing this implementation in your report, emphasize:

- role specialization for interpretability
- hybrid RAG-style grounding with lexical/vector retrieval and reranking
- rubric-guided self-reflection for controllable revision
- preference-style evaluation through pairwise win rate and chosen/rejected artifacts
- structured intermediate outputs for analysis
- reflection for iterative improvement
- safety constraints as a dedicated module rather than an afterthought
