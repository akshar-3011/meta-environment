"""Concrete inference strategies built on top of BaseInference."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .base import BaseInference


class StandardInference(BaseInference):
    """Scripted API-contract demo — always emits the same fixed actions.

    This is intentional: it exists to verify the API call sequence works
    end-to-end.  It does NOT read the email.  Use EmailAwareInference for
    an agent that actually makes decisions based on observation content.
    """

    @property
    def title(self) -> str:
        return "WORKPLACE ENVIRONMENT AGENT"

    def build_actions(self, observation: Dict[str, Any]) -> List[Tuple[str, str]]:
        return [
            ("classify", "complaint"),
            (
                "reply",
                "We sincerely apologize for the issue you experienced. We understand your frustration and "
                "will resolve this immediately. Our team will contact you within 24 hours with a solution.",
            ),
            ("escalate", "yes"),
        ]


class EnhancedInference(StandardInference):
    """Verbose scripted demo that reveals metadata labels for diagnostics."""

    @property
    def title(self) -> str:
        return "ENHANCED WORKPLACE ENVIRONMENT AGENT"

    @property
    def reveal_label(self) -> bool:
        return True


class EmailAwareInference(BaseInference):
    """Keyword-heuristic agent that reads the email before deciding.

    Bug 4 fix: replaces the pattern where all strategies hardcoded the same
    action sequence regardless of email content.  This is the reference
    implementation for a real agent — it inspects the observation, classifies
    from email text, then generates a category-appropriate reply and escalation
    decision.

    Not an RL agent (no learned weights), but demonstrates the correct pattern:
    read the observation → decide → act.
    """

    # Classification signal words
    _REFUND_SIGNALS = ["refund", "money back", "reimburs", "cancel", "return", "charged"]
    _COMPLAINT_SIGNALS = [
        "terrible", "awful", "unacceptable", "furious", "angry", "damaged",
        "charged twice", "not responding", "disappointed", "disgusted",
        "smashed", "broken", "great,",   # sarcasm detection ("Oh great,")
        "loving the experience",          # sarcasm detection
        "fourth time",                    # repeat contact
        "last time i contact",            # threat signal
        "reviews everywhere",             # reputation threat
    ]

    # Category-specific reply templates
    _REPLIES: Dict[str, str] = {
        "complaint": (
            "We sincerely apologize for the experience you have had. "
            "We understand your frustration and are fully committed to resolving "
            "this issue as quickly as possible. A member of our team will contact "
            "you within 24 hours with a concrete solution. "
            "Thank you for bringing this to our attention."
        ),
        "refund": (
            "Thank you for reaching out. We will process your refund right away. "
            "Please allow 3–5 business days for the amount to return to your account. "
            "If you have any further questions, please do not hesitate to contact us."
        ),
        "query": (
            "Thank you for your question — we are happy to help! "
            "Please contact our support team and we will provide you with "
            "the information you need as quickly as possible. "
            "Let us know if there is anything else we can assist with."
        ),
    }

    @property
    def title(self) -> str:
        return "EMAIL-AWARE WORKPLACE ENVIRONMENT AGENT"

    def _classify_email(self, email: str) -> str:
        """Classify email intent using keyword heuristics."""
        lower = email.lower()
        complaint_score = sum(1 for s in self._COMPLAINT_SIGNALS if s in lower)
        refund_score = sum(1 for s in self._REFUND_SIGNALS if s in lower)
        if complaint_score > refund_score:
            return "complaint"
        if refund_score > 0:
            return "refund"
        return "query"

    def build_actions(self, observation: Dict[str, Any]) -> List[Tuple[str, str]]:
        email = observation.get("email", "")
        category = self._classify_email(email)
        reply = self._REPLIES[category]
        escalate = "yes" if category == "complaint" else "no"
        return [
            ("classify", category),
            ("reply", reply),
            ("escalate", escalate),
        ]


class AsyncInference(StandardInference):
    """Async wrapper enabling concurrent batch episode execution.

    Uses ``asyncio.to_thread`` over the sync HTTP implementation so we keep
    dependencies minimal while still enabling parallel episode execution.
    """

    @property
    def title(self) -> str:
        return "ASYNC WORKPLACE ENVIRONMENT AGENT"

    async def run_episode_async(
        self,
        actions: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self.run_episode, actions)

    async def run_batch_async(
        self,
        batch_actions: Iterable[Optional[Sequence[Tuple[str, str]]]],
    ) -> List[Optional[Dict[str, Any]]]:
        tasks = [self.run_episode_async(actions=actions) for actions in batch_actions]
        return await asyncio.gather(*tasks)