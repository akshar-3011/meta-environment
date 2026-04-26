# Meta-Environment: A Self-Improving RL Platform for Customer Support Agents

## Problem

Enterprise customer support handles millions of emails daily. Current AI agents are static - they classify, reply, and escalate using fixed rules that never adapt. When they fail on edge cases (ambiguous complaints, multi-intent queries, false escalations), the same mistakes repeat indefinitely. Each incorrect escalation costs ~$150 in human agent time. At scale, this rigidity costs enterprises millions annually.

## What We Built

**Meta-Environment**is a reinforcement learning platform where the agent improves itself across generations. It operates in a 3-step action loop - classify the email, compose a reply, escalate or resolve - graded by a rule-based reward policy. What makes it different:

1. **Self-Improving Loop**: After each evaluation, a failure analyzer identifies systematic weaknesses. A strategy optimizer generates updated classification rules, reply templates, and escalation policies. The new strategy is re-evaluated - only accepted if it outperforms the baseline.

2. **Curriculum Learning**: Scenarios where the agent scores below 0.60 are automatically upweighted 3× in subsequent generations, forcing the system to practice its weaknesses rather than coasting on easy cases.

3. **Regression Testing**: 10 golden scenarios (spanning easy/medium/hard) act as a regression suite. If a new strategy drops golden-set performance below 90% of baseline, it triggers an automatic retry with a broadened optimization prompt - preventing catastrophic forgetting.

## Evidence

The system runs end-to-end in **2 seconds**(`python improvement_loop.py --demo`), producing a live ASCII dashboard with colored reward curves, strategy diffs, and business-impact summaries. All 232 tests pass. The platform ships with 100 validated scenarios, Kubernetes-ready Helm charts, and a Hugging Face Space deployment.

## Novelty

Unlike standard RL benchmarks that treat the environment as fixed, Meta-Environment treats *strategy evolution* as the optimization target - the agent doesn't just learn actions, it learns *how to learn*. The curriculum sampler, regression tester, and convergence detector form a closed-loop system that would be dangerous to deploy without safeguards, and we built those safeguards in.

---

**Stack**: Python · FastAPI · Gymnasium · Anthropic Claude · Docker · Kubernetes · Hugging Face Spaces  
**Repo**: [github.com/akshar-3011/meta-environment](https://github.com/akshar-3011/meta-environment)  
**Live Demo**: [huggingface.co/spaces/Akshar-3011/meta-environment](https://huggingface.co/spaces/Akshar-3011/meta-environment)
