"""01_quickstart.py — Minimal working example.

Runs a single episode through the environment in under 10 lines.
No server required — uses the environment directly.

Expected output:
    Email: I want a refund for my order...
      classify: reward=0.400
      reply: reward=0.217
      escalate: reward=0.225
    Total episode reward: 0.842
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.models import WorkplaceAction
from environment.workplace_environment import WorkplaceEnvironment

env = WorkplaceEnvironment()
obs = env.reset()
print(f"Email: {obs.email[:75]}...")

total_reward = 0.0
actions = [
    WorkplaceAction(action_type="classify", content="refund"),
    WorkplaceAction(action_type="reply", content="Thank you for contacting us. We sincerely apologize for the inconvenience. We will process your refund within 3-5 business days."),
    WorkplaceAction(action_type="escalate", content="no"),
]

for action in actions:
    obs = env.step(action)
    reward = obs.reward or 0.0
    total_reward += reward
    print(f"  {action.action_type}: reward={reward:.3f}")

print(f"Total episode reward: {total_reward:.3f}")
