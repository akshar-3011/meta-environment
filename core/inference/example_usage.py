"""Example usage for strategy-based inference system.

Run directly:
    python -m core.inference.example_usage
"""

from __future__ import annotations

import asyncio

from .base import RetryConfig
from .strategies import AsyncInference, EnhancedInference, StandardInference


def standard_example():
    print("\n=== StandardInference (single episode) ===")
    strategy = StandardInference(timeout=8.0, retry=RetryConfig(max_attempts=2, backoff_seconds=0.25))
    strategy.run_episode()


def enhanced_example():
    print("\n=== EnhancedInference (single episode) ===")
    strategy = EnhancedInference(timeout=10.0, retry=RetryConfig(max_attempts=3, backoff_seconds=0.5))
    strategy.run_episode()


def batch_example():
    print("\n=== StandardInference (batch processing) ===")
    strategy = StandardInference(timeout=8.0, retry=RetryConfig(max_attempts=2, backoff_seconds=0.25))

    custom_batch = [
        [
            ("classify", "complaint"),
            ("reply", "We are sorry for the issue and will resolve it quickly."),
            ("escalate", "yes"),
        ],
        [
            ("classify", "query"),
            ("reply", "Happy to help. Please share your order ID for additional information."),
            ("escalate", "no"),
        ],
    ]
    strategy.run_batch(custom_batch)


async def async_example():
    print("\n=== AsyncInference (concurrent batch) ===")
    strategy = AsyncInference(timeout=10.0, retry=RetryConfig(max_attempts=3, backoff_seconds=0.4))

    async_batch = [
        None,
        [
            ("classify", "refund"),
            ("reply", "We can process your refund request within 3 business days."),
            ("escalate", "no"),
        ],
    ]
    await strategy.run_batch_async(async_batch)


class CustomEscalationFirstInference(StandardInference):
    """Example extension showing how easy strategy customization is."""

    @property
    def title(self) -> str:
        return "CUSTOM ESCALATION-FIRST INFERENCE"

    def build_actions(self, observation):
        return [
            ("classify", "complaint"),
            ("reply", "We are sorry for the inconvenience and will resolve this issue immediately."),
            ("escalate", "yes"),
        ]


def extensibility_example():
    print("\n=== Custom strategy extension ===")
    strategy = CustomEscalationFirstInference(timeout=9.0, retry=RetryConfig(max_attempts=2, backoff_seconds=0.2))
    strategy.run_episode()


def main():
    # Uncomment examples as needed when your local server is running.
    standard_example()
    enhanced_example()
    batch_example()
    asyncio.run(async_example())
    extensibility_example()


if __name__ == "__main__":
    main()
