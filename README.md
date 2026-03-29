# MindBridge

`MindBridge` is a minimal Python project skeleton for a self-reflective multi-agent supportive dialogue system. It is designed for course-project use: easy to read, easy to extend, and aligned with the architecture you proposed.

## What this skeleton includes

- `Input Analyzer` for emotion, intent, and risk classification
- `Empathy Agent` for emotional acknowledgment
- `Strategy Agent` for practical next steps
- `Safety Agent` for response constraints and risk checks
- `Coordinator` for first-pass response synthesis
- `Reflection Critic` and `Reviser` for self-improvement
- `Final Safety Check` before the reply is returned
- a CLI entry point and a simple `Streamlit` demo

## Architecture

```text
User Input
   |
   v
Input Analyzer
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
  pipeline.py
  data/
    eval_cases.json
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
- optional `OPENAI_TEMPERATURE`
- optional `SHOW_INTERMEDIATE`

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

## Run the demo

```bash
streamlit run demo_streamlit.py
```

## Run evaluation

This script runs the single-agent baseline plus the full and ablated pipeline variants across `data/eval_cases.json`.

```bash
python evaluate.py
```

By default it saves:

- `evaluation_results.json`
- `evaluation_report.md`

## How the code is organized

### `config.py`
Loads environment variables and basic runtime settings.

### `prompts.py`
Stores the prompt templates for all agents. This is the fastest place to iterate on role boundaries and output schemas.

### `agents.py`
Defines a reusable JSON-returning agent wrapper over the LLM API.

### `pipeline.py`
Runs the end-to-end workflow:

1. analyze the input
2. run empathy, strategy, and safety in parallel
3. coordinate a draft response
4. critique and revise the draft
5. apply a final safety check

Supports ablation modes for:

- `no_critic`
- `no_reviser`
- `no_safety`

### `baseline.py`
Defines the single-agent baseline used for comparison against the multi-agent pipeline.

### `evaluate.py`
Runs batch evaluation across the baseline, the full pipeline, and the ablation variants.

### `schemas.py`
Defines the shared `DialogueState` object used to store intermediate outputs.

### `data/eval_cases.json`
Small starter evaluation set that you can expand for experiments.

## Suggested next development steps

### 1. Improve prompt quality

Right now the prompts are intentionally simple. Your next iteration should:

- sharpen each agent's role boundary
- define clearer output labels
- add one or two few-shot examples for difficult cases

### 2. Add a baseline

Create a single-agent baseline script so you can compare:

- single-agent baseline
- multi-agent without reflection
- full multi-agent with reflection

### 3. Add evaluation scripts

Useful scripts to add next:

- `run_baseline.py`
- `run_ablation.py`
- `judge.py`
- `metrics.py`

Then evaluate on:

- empathy
- helpfulness
- safety
- coherence

### 4. Add safety routing

For high-risk cases, add explicit routing logic such as:

- reduce normal advice
- prioritize supportive and immediate safety language
- provide crisis-oriented guidance templates

## Notes and limitations

- This scaffold is for research and class-project prototyping.
- It is not a production mental-health support system.
- The current implementation depends heavily on prompt design and API behavior.
- The JSON parsing is intentionally lightweight; you may want stronger schema validation later.

## Recommended report framing

When describing this implementation in your report, emphasize:

- role specialization for interpretability
- structured intermediate outputs for analysis
- reflection for iterative improvement
- safety constraints as a dedicated module rather than an afterthought

## Quick checklist

- [ ] Fill in `.env`
- [ ] Run the CLI once
- [ ] Test on the sample cases in `data/eval_cases.json`
- [ ] Tighten prompts
- [ ] Add baseline and evaluation scripts
- [ ] Prepare case-study examples for the final presentation
