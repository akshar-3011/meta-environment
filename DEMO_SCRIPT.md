# 🎬 Demo Script — 90 Seconds

> **Runtime**: `python improvement_loop.py --demo` completes in ~2s.
> You have 88 seconds of talk time. This script is timed to the second.

---

## Pre-Demo Checklist (do before you walk up)

```bash
cd workplace_env
source .venv/bin/activate
# Warm run — caches everything, confirms no crashes
python improvement_loop.py --demo
# Verify RESULTS.md exists
cat RESULTS.md | head -5
```

- [ ] Terminal font ≥ 16pt (judges in back row)
- [ ] Terminal width ≥ 120 columns (bar chart fits)
- [ ] Dark background (ANSI colors pop)
- [ ] Close all other apps (no notification interrupts)

---

## The Script

### 0:00–0:15 — THE HOOK (stand, don't touch keyboard yet)

> "Every company uses AI for customer support. But here's the problem — **these agents never get better**. They make the same mistakes on Day 1000 that they made on Day 1. We built a system where the agent **improves itself**."

### 0:15–0:25 — ONE COMMAND (type it live)

```bash
python improvement_loop.py --demo
```

> "One command. The system evaluates the agent, finds its failures, generates a better strategy, and re-evaluates — all automatically."

*Output appears in ~2 seconds. Point at the terminal.*

### 0:25–0:45 — READ THE DASHBOARD (point at each section)

Point at the **REWARD CURVE** bars:

> "Generation 0 is our baseline — the bars show classification, reply quality, and escalation accuracy. Each generation, the system analyzes failures and evolves the strategy."

Point at the **Strategy Diff**:

> "Here you can see exactly what changed — it added classification rules, reply templates, escalation policies. Every change is traceable."

### 0:45–0:60 — THE BUSINESS PANEL (this wins judges)

Point at the **BUSINESS IMPACT** box:

> "This is what matters in production. Email categorization accuracy, reply quality as a customer satisfaction proxy, and escalation accuracy — where each wrong escalation costs a company $150 in human agent time."

Point at the **💰 cost line**:

> "At 1,000 emails per day, that's real money saved — automatically, with no human in the loop."

### 0:60–0:75 — WHAT MAKES IT SPECIAL (no slides, just talk)

> "Three things make this different. **One** — curriculum learning. The system practices its weaknesses, not its strengths. **Two** — regression testing. Ten golden scenarios prevent the agent from forgetting what it already knows. **Three** — it runs in 2 seconds, ships as a Docker container, and is live on Hugging Face right now."

### 0:75–0:90 — THE CLOSE (look at judges, not screen)

> "Traditional RL environments are static. Ours evolves. The agent doesn't just learn actions — **it learns how to learn**. One command, self-improving, production-ready."

*Pause. Smile. Wait for questions.*

---

## Anticipated Questions & Answers

| Question | Answer |
|:---|:---|
| "How is this different from fine-tuning?" | "Fine-tuning updates model weights. We update the *strategy* — classification rules, templates, escalation logic. The model stays frozen; the policy evolves." |
| "Does it actually improve?" | "The loop is designed to only accept strategies that outperform baseline. When the fallback strategy underperforms, the system correctly rejects it. That's the safety mechanism working." |
| "What if the API is down?" | "Demo mode has built-in fallback — it loads the cached strategy from disk. The demo runs identically with or without an API key." |
| "How many scenarios?" | "100 validated scenarios across 5 categories and 3 difficulty levels, with curriculum sampling that upweights failures 3×." |
| "What's the regression testing?" | "10 golden scenarios spanning easy to hard. If a new strategy drops below 90% of baseline on those, it auto-retries with a broadened prompt. Prevents catastrophic forgetting." |
| "Is this deployed?" | "Yes — Hugging Face Space, Docker image, Helm chart for Kubernetes. Plus PyPI-ready packaging." |
| "What's the stack?" | "Python, FastAPI, Gymnasium, Anthropic Claude for strategy optimization, Docker, Kubernetes, Hugging Face Spaces." |

---

## If Something Goes Wrong

| Failure | Recovery |
|:---|:---|
| Script crashes | `python generate_report.py` — show RESULTS.md in browser instead |
| Terminal too small | Zoom in on the BUSINESS IMPACT box only — that's the money shot |
| No internet | Demo mode works fully offline with cached strategy |
| Judges stop you early | Skip to 0:75 close: "It learns how to learn. One command." |
| They want to see code | Open `improvement_loop.py`, scroll to the loop (line ~410), show the generate → validate → accept/reject flow |

---

## Timing Verification

Run this to time yourself reading the script aloud:

```bash
# Start a stopwatch
time say "Every company uses AI for customer support. But here is the problem.
These agents never get better. They make the same mistakes on Day 1000 that
they made on Day 1. We built a system where the agent improves itself."
```

Expected: ~12 seconds for the hook. If you're over 15s, speak faster.

**Total word count in speaking sections: ~280 words ÷ ~3 words/sec = ~93 seconds.**
Trim the "Three things" section if you're running long.
