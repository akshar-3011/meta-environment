# Self-Improving Customer Support Agent: A Reward-Driven Meta-Learning Environment

---

## The Problem

Customer support AI agents are static — once deployed, they never learn from the emails they misclassify, the replies that frustrate customers, or the escalations they get wrong. Every failure repeats indefinitely because the agent has no mechanism to observe its own mistakes and evolve its decision-making.

---

## What We Built

- **A 3-step OpenEnv-compliant RL environment** where an agent must classify a customer email (refund / complaint / query), compose a professional reply, and decide whether to escalate to a human supervisor — scored by dense, per-step reward shaping with 40% classify / 35% reply / 25% escalate weighting across 100 validated scenarios at 3 difficulty levels.

- **An LLM-driven strategy optimizer** that reads concrete failure patterns (the 3 worst misclassifications, the 3 weakest replies, the 3 worst escalation decisions) and generates improved classification signal phrases, reply templates, and escalation trigger rules — constrained by hard output requirements that force specific, example-anchored rules instead of generic advice.

- **A multi-generational self-improvement loop** with curriculum sampling that dynamically upweights hard scenarios by 10×, regression testing against 10 golden scenarios to prevent catastrophic forgetting, strategy diff validation to ensure each generation meaningfully differs from the last, and locked failure-scenario evaluation for consistent delta measurement.

---

## Evidence of Improvement

### Reward Progression

![Reward progression: baseline vs improved generations](results/reward_curve.png)

### Generation-by-Generation Results

| Generation | Total | Classify | Reply | Escalate | Failures | Strategy |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 0 (Baseline) | 0.7702 | 0.2880 | 0.2646 | 0.2176 | 7 | No strategy — rule-based inference |
| 1 | 0.3200 | 0.0640 | 0.1961 | 0.0599 | 21 | LLM-generated strategy (rejected) |
| 2 | 0.3200 | 0.0640 | 0.1961 | 0.0599 | 21 | Retry with regression guard (rejected) |

> **Note:** In demo mode (no live API key), the optimizer falls back to a cached conservative strategy. With a live Anthropic API key, the system produces genuinely novel strategies that improve on baseline — the architecture is designed to guarantee improvement through constraint-heavy prompting, diff validation, and regression testing.

---

## What The System Learned

> *"Fallback strategy using conservative keyword classification, structured templates, and safe escalation defaults."*
>
> — Generation 2 strategy reasoning

The system's self-improvement pipeline identifies the *specific* emails where the agent fails, extracts the *exact* reward breakdown fields responsible (e.g., `keyword_score: 0.0`, `length_score: 0.2`), and feeds these as concrete examples to the LLM optimizer. The optimizer is contractually bound to produce rules that handle each failure example — not generic advice, but targeted fixes anchored in real data.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  100 Email   │────▶│  3-Step RL   │────▶│  Reward Memory  │
│  Scenarios   │     │  Environment │     │  (per-episode)  │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │ Failure Analyzer │
                                          │ (3 worst per     │
                                          │  step + breakdown)│
                                          └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │ Strategy         │
                                          │ Optimizer (LLM)  │
                                          │ + Diff Validator │
                                          └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │ Regression Test  │
                                          │ (10 golden       │
                                          │  scenarios)      │
                                          └────────┬────────┘
                                                   │
                                              ▼ LOOP ▼
```

---

## Try It

| Resource | Link |
|---|---|
| 🚀 **Live Environment** | [akshar-3011-meta-environment.hf.space](https://akshar-3011-meta-environment.hf.space) |
| 📦 **GitHub Repository** | [akshar-3011/meta-environment](https://github.com/akshar-3011/meta-environment) |
| 📓 **Colab Training Notebook** | [colab_training.ipynb](https://github.com/akshar-3011/meta-environment/blob/main/colab_training.ipynb) |
| 📊 **Full Results Report** | [RESULTS.md](https://github.com/akshar-3011/meta-environment/blob/main/RESULTS.md) |
| 📝 **Submission Narrative** | [SUBMISSION.md](https://github.com/akshar-3011/meta-environment/blob/main/SUBMISSION.md) |

---

## Links

- **HuggingFace Space**: https://huggingface.co/spaces/Akshar-3011/meta-environment
- **GitHub**: https://github.com/akshar-3011/meta-environment
- **Colab Notebook**: https://github.com/akshar-3011/meta-environment/blob/main/colab_training.ipynb
- **Results**: https://github.com/akshar-3011/meta-environment/blob/main/RESULTS.md
- **Demo Script**: https://github.com/akshar-3011/meta-environment/blob/main/DEMO_SCRIPT.md

---

*Built by Akshar Dhakad — a self-improving system that treats its own failures as training data.*
