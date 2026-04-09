"""02_custom_agent.py — Integrate a custom RL agent with the environment.

Demonstrates the agent ↔ environment loop with a simple rule-based agent
that reads the email text and makes decisions based on keywords.

Expected output:
    Episode 1/5: classify=0.400 reply=0.217 escalate=0.225 | total=0.842
    ...
    Mean reward over 5 episodes: 0.84 ± 0.02
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Dict, List

from core.models import WorkplaceAction
from environment.workplace_environment import WorkplaceEnvironment


class KeywordAgent:
    """Simple keyword-based agent for demonstration.

    A real agent would use a neural network or LLM here.
    """

    KEYWORDS: Dict[str, List[str]] = {
        "refund": ["refund", "money back", "return", "credit", "reimburse"],
        "complaint": ["angry", "unacceptable", "terrible", "worst", "furious"],
        "query": ["how", "when", "where", "what", "question", "wondering"],
    }

    def classify(self, email: str) -> str:
        """Classify email by keyword matching."""
        email_lower = email.lower()
        scores: Dict[str, int] = {label: 0 for label in self.KEYWORDS}
        for label, keywords in self.KEYWORDS.items():
            for kw in keywords:
                if kw in email_lower:
                    scores[label] += 1
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def reply(self, category: str) -> str:
        """Generate a templated reply based on category."""
        templates = {
            "refund": "Thank you for contacting us. We apologize for the inconvenience. Your refund will be processed within 3-5 business days.",
            "complaint": "We sincerely apologize for this experience. Your feedback is important and we will investigate immediately.",
            "query": "Thank you for your question. We'd be happy to help with the information you requested.",
        }
        return templates.get(category, templates["query"])

    def should_escalate(self, email: str, category: str) -> bool:
        """Decide whether to escalate to a senior team."""
        urgent_words = ["urgent", "legal", "lawyer", "sue", "deadline"]
        return any(w in email.lower() for w in urgent_words) and category == "complaint"

    def act(self, email: str) -> List[WorkplaceAction]:
        """Plan all 3 actions for an episode."""
        category = self.classify(email)
        return [
            WorkplaceAction(action_type="classify", content=category),
            WorkplaceAction(action_type="reply", content=self.reply(category)),
            WorkplaceAction(action_type="escalate",
                          content="yes" if self.should_escalate(email, category) else "no"),
        ]


def run_evaluation(n_episodes: int = 5) -> None:
    """Run the agent on multiple episodes and report statistics."""
    env = WorkplaceEnvironment()
    agent = KeywordAgent()
    rewards: List[float] = []

    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        actions = agent.act(obs.email)
        episode_reward = 0.0
        step_rewards: List[str] = []

        for action in actions:
            obs = env.step(action)
            r = obs.reward or 0.0
            episode_reward += r
            step_rewards.append(f"{action.action_type}={r:.3f}")

        rewards.append(episode_reward)
        print(f"Episode {ep}/{n_episodes}: {' '.join(step_rewards)} | total={episode_reward:.3f}")

    import statistics
    mean = statistics.mean(rewards)
    std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    print(f"\nMean reward over {n_episodes} episodes: {mean:.2f} ± {std:.2f}")


if __name__ == "__main__":
    run_evaluation()
