# AutonomousReturns-v0 (RevLogisticsRL)

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

## More Documentation

- Package/API details: `RevLogiXenv_v0/README.md`
