"""Adaptive inference agent driven entirely by a strategy dictionary."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple


class AdaptiveAgent:
    """Drop-in inference agent that derives decisions from strategy rules."""

    _DEFAULT_CATEGORY = "query"
    _DEFAULT_REPLY = "Thank you for reaching out. We are here to assist you."
    _DEFAULT_ESCALATE = "no"

    def __init__(self, strategy: dict):
        self.strategy: Dict[str, Any] = strategy if isinstance(strategy, dict) else {}

    def build_actions(self, observation: dict) -> list[tuple[str, str]]:
        obs = observation if isinstance(observation, dict) else {}

        category = self._classify(obs)
        reply = self._generate_reply(obs, category)
        escalate = self._decide_escalation(obs, category)

        # Exact required structure
        return [
            ("classify", category),
            ("reply", reply),
            ("escalate", escalate),
        ]

    def _classify(self, observation: Dict[str, Any]) -> str:
        try:
            email = str(observation.get("email", "")).lower()
            urgency = str(observation.get("urgency", "")).lower()

            rules_raw = self.strategy.get("classification_rules", {})
            rules = rules_raw if isinstance(rules_raw, dict) else {}
            strategy_signals = rules

            categories = ["refund", "complaint", "query"]
            scores: Dict[str, int] = {c: 0 for c in categories}

            for category in categories:
                signals = self._safe_str_list(rules.get(category, []))
                scores[category] = sum(1 for phrase in signals if phrase and phrase.lower() in email)

            if urgency == "high":
                scores["complaint"] += 1

            # Only apply interrogative boost when scores are tied
            max_score = max(scores.values()) if scores else 0
            tied_categories = [c for c, s in scores.items() if s == max_score]

            if len(tied_categories) > 1:
                # Multiple categories tied — use interrogative signal as tiebreak
                email_lower = email.lower()
                is_interrogative = (
                    "?" in email or
                    any(email_lower.startswith(w) for w in
                        ["how", "what", "when", "where", "can", "do", "is", "are", "will"])
                )
                if is_interrogative and "query" in tied_categories:
                    scores["query"] += 1

            scores = {k: max(0, v) for k, v in scores.items()}

            if all(value == 0 for value in scores.values()):
                default_category = str(rules.get("default", self._DEFAULT_CATEGORY)).lower()
                return default_category if default_category in categories else self._DEFAULT_CATEGORY

            # Deterministic tie behavior by fixed category order.
            selected_category = categories[0]
            best_score = scores[selected_category]
            for category in categories[1:]:
                if scores[category] > best_score:
                    selected_category = category
                    best_score = scores[category]

            if os.environ.get("AGENT_DEBUG"):
                for cat, score in scores.items():
                    print(f"[AGENT DEBUG] {cat}: score={score}, signals={strategy_signals.get(cat, [])[:3]}")
                print(f"[AGENT DEBUG] Selected: {selected_category}")

            return selected_category
        except Exception:
            return self._DEFAULT_CATEGORY

    def _generate_reply(self, observation: Dict[str, Any], category: str) -> str:
        try:
            email_raw = str(observation.get("email", ""))
            excerpt = " ".join(email_raw.split()[:5]).strip()

            templates_raw = self.strategy.get("reply_templates", {})
            templates = templates_raw if isinstance(templates_raw, dict) else {}
            template = str(templates.get(category, self._DEFAULT_REPLY))

            reply_text = template.replace("{excerpt}", excerpt)

            req_raw = self.strategy.get("reply_requirements", {})
            req = req_raw if isinstance(req_raw, dict) else {}

            min_length = self._safe_int(req.get("min_length"), 0)
            must_greet = self._safe_bool(req.get("must_include_greeting", False))
            must_close = self._safe_bool(req.get("must_include_closing", False))
            forbidden = self._safe_str_list(req.get("forbidden_phrases", []))

            if must_greet and not self._has_greeting(reply_text):
                reply_text = f"Hello, {reply_text.strip()}"

            if len(reply_text) < min_length:
                if reply_text and not reply_text.endswith(" "):
                    reply_text += " "
                reply_text += "We appreciate your patience and are here to help."

            for phrase in forbidden:
                reply_text = self._remove_case_insensitive(reply_text, phrase)

            if must_close and not self._has_closing(reply_text):
                if reply_text and not reply_text.endswith(" "):
                    reply_text += " "
                reply_text += "Regards, Support Team."

            cleaned = " ".join(reply_text.split()).strip()
            return cleaned or self._DEFAULT_REPLY
        except Exception:
            return self._DEFAULT_REPLY

    def _decide_escalation(self, observation: Dict[str, Any], category: str) -> str:
        try:
            email = str(observation.get("email", "")).lower()
            urgency = str(observation.get("urgency", "")).lower()
            chosen_category = category

            rules_raw = self.strategy.get("escalation_rules", {})
            rules = rules_raw if isinstance(rules_raw, dict) else {}

            always_escalate = self._safe_str_list(rules.get("always_escalate", []))
            for trigger in always_escalate:
                if trigger and trigger.lower() in email:
                    return "yes"

            never_escalate = self._safe_str_list(rules.get("never_escalate", []))
            for pattern in never_escalate:
                if pattern and pattern.lower() in email:
                    return "no"

            if os.environ.get("AGENT_DEBUG"):
                print(
                    f"[ESCALATE DEBUG] chosen_category={chosen_category}, "
                    f"escalate_if_complaint={rules.get('escalate_if_complaint')}"
                )
            if self._safe_bool(rules.get("escalate_if_complaint", False)) and chosen_category == "complaint":
                return "yes"

            if self._safe_bool(rules.get("escalate_if_high_urgency", False)) and urgency == "high":
                return "yes"

            return "no"
        except Exception:
            return self._DEFAULT_ESCALATE

    @staticmethod
    def _safe_str_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if value is None:
            return []
        return [str(value)]

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _safe_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "1", "yes", "y"}:
                return True
            if low in {"false", "0", "no", "n"}:
                return False
        return bool(value)

    @staticmethod
    def _remove_case_insensitive(text: str, phrase: str) -> str:
        source = text or ""
        target = phrase or ""
        if not target:
            return source

        lower_source = source.lower()
        lower_target = target.lower()
        idx = lower_source.find(lower_target)
        while idx != -1:
            source = source[:idx] + source[idx + len(target):]
            lower_source = source.lower()
            idx = lower_source.find(lower_target)
        return source

    @staticmethod
    def _has_greeting(text: str) -> bool:
        low = (text or "").lower()
        greetings = ("hello", "hi", "dear", "greetings")
        return any(g in low for g in greetings)

    @staticmethod
    def _has_closing(text: str) -> bool:
        low = (text or "").lower()
        closings = ("regards", "sincerely", "support team", "best")
        return any(c in low for c in closings)


__all__ = ["AdaptiveAgent"]
