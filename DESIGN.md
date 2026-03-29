# Design Overview

## Why Multi-Step?

Real-world AI systems do not operate in isolation. Decisions are sequential and interdependent.

This environment introduces:
- Credit assignment challenge
- Sequential reasoning
- Error propagation

## Reward Design

- Partial rewards encourage learning
- Deterministic scoring ensures reproducibility
- Each step contributes independently

## Difficulty

A naive agent fails because:
- Classification impacts reply quality
- Escalation requires understanding intent
- Multi-step dependency creates complexity

