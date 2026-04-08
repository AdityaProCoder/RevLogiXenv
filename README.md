---
title: RevLogiXenv
emoji: "📦"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - reverse-logistics
  - fraud-detection
---

# RevLogiXenv

Reverse logistics is one of the most expensive and least optimized parts of e-commerce.

Every returned item forces a decision:
- resell it?
- discount it?
- refurbish it?
- dispose it?
- flag it as fraud?

These decisions are made under uncertainty:
- condition scores are noisy
- customer signals are imperfect
- fraud is adversarial and evolving
- outcomes are often delayed

Most real-world systems rely on heuristics or manual review.
They work for simple cases, but fail when signals conflict or when fraud mimics legitimate behavior.

---

## What This Project Does

RevLogiXenv models returns processing as a **sequential decision problem**.

Instead of predicting a label, an agent must:
- observe partial, noisy signals
- choose an operational action
- handle delayed consequences
- optimize long-term reward

This turns reverse logistics into a **reinforcement learning environment**, not a classification task.

---

## Why This Matters

This environment captures real operational trade-offs:

- Profit vs fraud risk
- Speed vs inspection cost
- False positives vs missed fraud
- Short-term vs delayed outcomes

It provides a benchmark for evaluating agents that must reason under uncertainty in real-world systems.

---

## Key Idea

> Returns processing is not a prediction problem — it is a policy learning problem.

---

## What This Repo Provides

- A POMDP-based reverse logistics environment
- Realistic noise, fraud, and delayed resolution dynamics
- Deterministic and LLM-based policies
- Oracle-based scoring for measurable evaluation
- OpenEnv-compatible server and hackathon-ready inference pipeline

## Project Structure

- Root: `RevLogiXenv/` (this folder)
- Package: `RevLogiXenv_v0/` (Python environment + baselines)
- Spaces: `spaces/` (Hugging Face Space docs/config)

## How to Run

### 1. Setup

```bash
uv venv
.venv\\Scripts\\activate
uv pip install -e .
```

### 2. Run the API Server

```bash
python run_server.py
```

Useful endpoints:
- `GET http://127.0.0.1:8000/health`
- `GET http://127.0.0.1:8000/docs`

### 3. Run the Deterministic Baseline

```bash
python run_baseline.py --mode local --tasks easy,medium,hard --seeds 42 --pretty
```

## Benchmark Results

Performance was evaluated across three difficulty tiers comparing heuristic baseline against LLM agents (MiniMax-M2.7, Gemini 2.5 Flash, Gemini 2.5 Pro):

### Task Completion (Final Score)

| Difficulty | Heuristic | MiniMax-M2.7 | Gemini 2.5 Flash | Gemini 2.5 Pro |
|:-----------|:----------|:-------------|:------------------|:----------------|
| Easy       | 0.946     | 0.770        | 0.876             | **0.959**       |
| Medium     | 0.690     | 0.536        | 0.842             | **0.931**       |
| Hard       | 0.395     | 0.359        | 0.621             | **0.783**       |

### Fraud Detection (F1 Score)

| Difficulty | Heuristic | MiniMax-M2.7 | Gemini 2.5 Flash | Gemini 2.5 Pro |
|:-----------|:----------|:-------------|:------------------|:----------------|
| Easy       | 1.00      | 1.00         | 1.00              | 1.00            |
| Medium     | 0.706     | 0.00         | 0.857             | **0.923**       |
| Hard       | 0.545     | 0.00         | 0.800             | **0.923**       |

### Economic Efficiency (Hard Tier)

| Model             | Profit    | Efficiency |
|:------------------|:----------|:-----------|
| Optimal (Theoretical) | $9,363.39 | 100%       |
| Gemini 2.5 Pro    | $6,453.76 | 68.9%      |
| Gemini 2.5 Flash  | $4,697.00 | 50.1%      |
| Heuristic         | $2,762.76 | 29.5%      |

### Key Findings

- **Reasoning Gap**: High-tier reasoning models (Gemini 2.5 Pro) achieve 133% more profit than heuristics in complex environments
- **Safety Collapse**: MiniMax-M2.7 exhibits "fraud paranoia" in Hard mode, triggering flag_fraud 18x per 50 steps with near-zero rewards
- **Graceful Degradation**: Gemini models maintain robust F1 > 0.90 under uncertainty; heuristic and MiniMax collapse at higher difficulties
- **Diagnostic Inefficiency**: Lower-tier models enter inspection loops with diminishing returns rather than committing to decisive actions

Full technical analysis: `Technical Inference Analysis inference.md`  
Full benchmark report: `Technical Submission Baseline Agent.md`

## More Documentation

- Package/API details: `RevLogiXenv_v0/README.md`
