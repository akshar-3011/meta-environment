"""LLM-driven strategy optimization from structured failure analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from groq import Groq as GroqClient
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


class StrategyOptimizer:
    """Generate an improved workflow strategy from failure analysis using an LLM."""

    _MODEL = "claude-sonnet-4-20250514"
    _MAX_TOKENS = 1500

    _SYSTEM_PROMPT = (
        "You are an expert in customer support workflow optimization. "
        "You are improving an EXISTING high-performing customer support agent. "
        "You MUST NOT degrade performance. "
        "Your task is to ANALYZE failures and PRODUCE IMPROVEMENTS over an already strong baseline. "
        "STRICT RULES: "
        "1. Do NOT generate generic strategies. "
        "2. Do NOT fallback to default/simple keyword matching. "
        "3. Build ON TOP of existing logic, not replace it. "
        "4. Improve weak areas: "
        "- If classification fails refine signal phrases. "
        "- If reply is weak increase length, keywords, and tone quality. "
        "- If escalation fails adjust rules carefully. "
        "5. Always maintain high recall for complaint detection and strong reply completeness. "
        "6. Your output must be MORE SPECIFIC than the baseline. "
        "If failure_analysis shows repeated misclassification add new signal phrases. "
        "If failure_analysis shows low length_score increase min_length. "
        "If failure_analysis shows low keyword_score enforce required keywords. "
        "DO NOT output safe/generic fallback strategies. "
        "Your goal is measurable improvement in reward score. "
        "You MUST output ONLY valid JSON. "
        "Do NOT output markdown. "
        "Do NOT output explanations outside JSON. "
        "Do NOT output any extra text. "
        "CRITICAL FORMAT RULE: "
        "classification_rules values MUST be flat JSON arrays of short keyword strings. "
        'Example of CORRECT format: {"refund": ["refund", "money back", "charged"]} '
        'Example of WRONG format: {"refund": [{"signal_phrases": ["refund"]}]} '
        "Do NOT nest objects inside classification arrays. "
        "Every classification rule value must be a flat list of strings only. "
        "You will be given a baseline performance score. Your output must produce strictly better agent decisions than the baseline. "
        "Generating rules similar to the baseline is a failure."
    )

    def __init__(self, client: Any):
        self.client = client

    def generate_strategy(
        self,
        failure_analysis: Dict[str, Any],
        current_strategy: Optional[Dict[str, Any]] = None,
        baseline_metrics_summary: Optional[Dict[str, Any]] = None,
        baseline_score: float = 0.0,
    ) -> Dict[str, Any]:
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        print(f"[OPTIMIZER DEBUG] API key present: {bool(api_key)}, length: {len(api_key)}")

        user_prompt = self._build_user_prompt(
            failure_analysis=failure_analysis,
            current_strategy=current_strategy,
            baseline_metrics_summary=baseline_metrics_summary,
            baseline_score=baseline_score,
        )

        strategy = self._fallback_strategy()

        first_text = self._request_strategy(user_prompt)
        print(f"[OPTIMIZER DEBUG] first_text length: {len(first_text)}, preview: {repr(first_text[:200])}")

        parsed = self._parse_json_maybe(first_text)
        print(f"[OPTIMIZER DEBUG] parsed is None: {parsed is None}")

        normalized: Optional[Dict[str, Any]] = None
        invalid_reason = "invalid JSON"
        if parsed is not None:
            candidate = self._normalize_strategy(parsed)
            is_valid, reason = self._validate_strategy_quality(parsed, candidate)
            print(f"[OPTIMIZER DEBUG] is_valid={is_valid}, reason={reason}")
            if is_valid:
                normalized = candidate
            else:
                invalid_reason = reason

        if normalized is None:
            retry_prompt = (
                user_prompt
                + "\n\n"
                + f"Your previous output was rejected: {invalid_reason}. "
                + "Return ONLY valid JSON and fix the strategy quality constraints."
            )
            retry_text = self._request_strategy(retry_prompt)
            parsed = self._parse_json_maybe(retry_text)
            if parsed is not None:
                candidate = self._normalize_strategy(parsed)
                is_valid, _reason = self._validate_strategy_quality(parsed, candidate)
                if is_valid:
                    normalized = candidate

        print(f"[OPTIMIZER DEBUG] normalized is None after retry: {normalized is None}")

        if normalized is not None:
            strategy = normalized

        self._save_strategy(strategy, "final_strategy.json")
        return strategy

    def _build_user_prompt(
        self,
        failure_analysis: Dict[str, Any],
        current_strategy: Optional[Dict[str, Any]] = None,
        baseline_metrics_summary: Optional[Dict[str, Any]] = None,
        baseline_score: float = 0.0,
    ) -> str:
        failure_examples = self._extract_failure_examples(failure_analysis)

        # Truncate failure analysis to avoid exceeding token limits
        fa_json = json.dumps(failure_analysis, ensure_ascii=False)
        if len(fa_json) > 3000:
            fa_json = fa_json[:3000] + '...}'  # keep it parseable-ish for LLM

        parts = [
            "Use the following failure analysis to produce an improved strategy in strict JSON schema.",
            "baseline_metrics_summary:",
            json.dumps(baseline_metrics_summary or {}, ensure_ascii=False),
            "failure_analysis:",
            fa_json,
            "explicit_failure_examples:",
            json.dumps(failure_examples, ensure_ascii=False),
        ]

        # Inject performance target before current strategy
        parts.extend([
            "",
            "PERFORMANCE TARGET:",
            f"The current baseline agent scores {baseline_score:.2f} on evaluation scenarios.",
            f"Your generated strategy MUST produce agent behavior that scores ABOVE {baseline_score:.2f}.",
            "Strategies that match or fall below this score are considered failures.",
            "Every rule you generate must be designed to improve on this specific score.",
        ])

        if current_strategy is not None:
            parts.extend(
                [
                    "",
                    "CURRENT STRATEGY (improve it):",
                    json.dumps(current_strategy, ensure_ascii=False),
                ]
            )

        return "\n".join(parts)

    def _extract_failure_examples(self, failure_analysis: Dict[str, Any]) -> List[str]:
        if not isinstance(failure_analysis, dict):
            return []

        examples: List[str] = []
        for key in ("classify_failures", "reply_failures", "escalate_failures"):
            block = failure_analysis.get(key)
            if isinstance(block, dict):
                raw = block.get("examples", [])
                if isinstance(raw, list):
                    for item in raw:
                        text = str(item).strip()
                        if text and text not in examples:
                            examples.append(text)
                            if len(examples) >= 6:
                                return examples
        return examples[:6]

    def _validate_strategy_quality(
        self,
        raw_data: Dict[str, Any],
        normalized: Dict[str, Any],
    ) -> tuple[bool, str]:
        fallback = self._fallback_strategy()

        raw_classification = raw_data.get("classification_rules")
        if not isinstance(raw_classification, dict):
            return False, "missing classification_rules"

        for category in ("refund", "complaint", "query"):
            if category not in raw_classification:
                return False, f"missing category: {category}"

        n_classification = normalized.get("classification_rules", {})
        for category in ("refund", "complaint", "query"):
            signals = n_classification.get(category, [])
            if not isinstance(signals, list) or len([s for s in signals if str(s).strip()]) == 0:
                return False, f"empty signal list for {category}"

        # Reject exact fallback/generic classification rules.
        if n_classification == fallback["classification_rules"]:
            return False, "classification rules identical to fallback"

        # Ensure strategy is richer than fallback in at least one meaningful dimension.
        richer_signals = False
        for category in ("refund", "complaint", "query"):
            current = {str(s).strip().lower() for s in n_classification.get(category, []) if str(s).strip()}
            base = {
                str(s).strip().lower()
                for s in fallback["classification_rules"].get(category, [])
                if str(s).strip()
            }
            if len(current) > len(base) or not current.issubset(base):
                richer_signals = True
                break

        n_reply_req = normalized.get("reply_requirements", {})
        richer_reply = int(n_reply_req.get("min_length", 0)) > int(
            fallback["reply_requirements"].get("min_length", 0)
        )

        if not (richer_signals or richer_reply):
            return False, "strategy not more specific than baseline"

        return True, "ok"

    def _request_strategy(self, user_prompt: str) -> str:
        # Try Anthropic-style client first
        if hasattr(self.client, "messages"):
            try:
                response = self.client.messages.create(
                    model=self._MODEL,
                    max_tokens=self._MAX_TOKENS,
                    system=self._SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ],
                )
                return self._extract_text(response)
            except Exception as e:
                err_msg = str(e).lower()
                print(f"[OPTIMIZER DEBUG] Anthropic call failed: {type(e).__name__}: {e}")
                # Fall through to Groq

        # Groq fallback
        if _GROQ_AVAILABLE:
            groq_key = os.environ.get("GROQ_API_KEY", "")
            if groq_key:
                try:
                    print("[OPTIMIZER] Using Groq fallback")
                    groq_client = GroqClient(api_key=groq_key)
                    resp = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": self._SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=2000,
                        temperature=0.3,
                    )
                    text = resp.choices[0].message.content or ""
                    text = self._strip_markdown_fences(text)
                    print(f"[OPTIMIZER DEBUG] Groq response length: {len(text)}")
                    return text
                except Exception as e:
                    print(f"[OPTIMIZER DEBUG] Groq call failed: {type(e).__name__}: {e}")
                    return ""
            else:
                print("[OPTIMIZER DEBUG] No GROQ_API_KEY set")

        return ""

    def _extract_text(self, response: Any) -> str:
        if response is None:
            return ""

        content = getattr(response, "content", response)

        if isinstance(content, str):
            return self._strip_markdown_fences(content)

        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text is not None:
                        chunks.append(str(text))
                        continue

                text_attr = getattr(item, "text", None)
                if text_attr is not None:
                    chunks.append(str(text_attr))
                    continue

                chunks.append(str(item))

            return self._strip_markdown_fences("\n".join(chunks).strip())

        return self._strip_markdown_fences(str(content).strip())

    def _strip_markdown_fences(self, text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if not lines:
            return ""

        # Remove opening fence and optional language token.
        lines = lines[1:]

        # Remove closing fence if present.
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        return "\n".join(lines).strip()

    def _parse_json_maybe(self, raw_text: str) -> Optional[Dict[str, Any]]:
        if not raw_text:
            return None

        try:
            parsed = json.loads(raw_text)
        except Exception:
            return None

        if not isinstance(parsed, dict):
            return None

        return parsed

    def _normalize_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        fallback = self._fallback_strategy()

        classification_in = data.get("classification_rules")
        if not isinstance(classification_in, dict):
            classification_in = {}

        reply_templates_in = data.get("reply_templates")
        if not isinstance(reply_templates_in, dict):
            reply_templates_in = {}

        reply_requirements_in = data.get("reply_requirements")
        if not isinstance(reply_requirements_in, dict):
            reply_requirements_in = {}

        escalation_in = data.get("escalation_rules")
        if not isinstance(escalation_in, dict):
            escalation_in = {}

        normalized = {
            "classification_rules": {
                "refund": self._to_str_list(
                    classification_in.get(
                        "refund",
                        fallback["classification_rules"]["refund"],
                    )
                ),
                "complaint": self._to_str_list(
                    classification_in.get(
                        "complaint",
                        fallback["classification_rules"]["complaint"],
                    )
                ),
                "query": self._to_str_list(
                    classification_in.get(
                        "query",
                        fallback["classification_rules"]["query"],
                    )
                ),
                "default": self._to_str(
                    classification_in.get(
                        "default",
                        fallback["classification_rules"]["default"],
                    )
                ),
            },
            "reply_templates": {
                "refund": self._to_str(
                    reply_templates_in.get("refund", fallback["reply_templates"]["refund"])
                ),
                "complaint": self._to_str(
                    reply_templates_in.get(
                        "complaint",
                        fallback["reply_templates"]["complaint"],
                    )
                ),
                "query": self._to_str(
                    reply_templates_in.get("query", fallback["reply_templates"]["query"])
                ),
            },
            "reply_requirements": {
                "min_length": self._to_int(
                    reply_requirements_in.get(
                        "min_length",
                        fallback["reply_requirements"]["min_length"],
                    ),
                    default=fallback["reply_requirements"]["min_length"],
                    minimum=1,
                ),
                "must_include_greeting": self._to_bool(
                    reply_requirements_in.get(
                        "must_include_greeting",
                        fallback["reply_requirements"]["must_include_greeting"],
                    )
                ),
                "must_include_closing": self._to_bool(
                    reply_requirements_in.get(
                        "must_include_closing",
                        fallback["reply_requirements"]["must_include_closing"],
                    )
                ),
                "forbidden_phrases": self._to_str_list(
                    reply_requirements_in.get(
                        "forbidden_phrases",
                        fallback["reply_requirements"]["forbidden_phrases"],
                    )
                ),
            },
            "escalation_rules": {
                "always_escalate": self._to_str_list(
                    escalation_in.get(
                        "always_escalate",
                        fallback["escalation_rules"]["always_escalate"],
                    )
                ),
                "never_escalate": self._to_str_list(
                    escalation_in.get(
                        "never_escalate",
                        fallback["escalation_rules"]["never_escalate"],
                    )
                ),
                "escalate_if_complaint": self._to_bool(
                    escalation_in.get(
                        "escalate_if_complaint",
                        fallback["escalation_rules"]["escalate_if_complaint"],
                    )
                ),
                "escalate_if_high_urgency": self._to_bool(
                    escalation_in.get(
                        "escalate_if_high_urgency",
                        fallback["escalation_rules"]["escalate_if_high_urgency"],
                    )
                ),
            },
            "reasoning": self._to_str(data.get("reasoning", fallback["reasoning"])),
        }

        return normalized

    def _fallback_strategy(self) -> Dict[str, Any]:
        return {
            "classification_rules": {
                "refund": ["refund", "reimbursement", "charged twice", "overcharged", "billing error", "money back", "charge", "credit", "cancel order", "cancellation"],
                "complaint": ["not happy", "gone downhill", "terrible", "awful", "unacceptable", "disappointed", "frustrated", "angry", "outraged", "worst", "quality has", "way too slow", "took too long", "poor quality", "bad experience", "let down"],
                "query": ["how do i", "how to", "what is", "can you", "do you", "where is", "when will", "return my item", "return policy", "track my", "status of", "information about", "tell me", "explain", "help me understand"],
                "default": "query",
            },
            "reply_templates": {
                "refund": "We sincerely apologize for the inconvenience. Your refund request has been received and will be processed within 3-5 business days.",
                "complaint": "We're really sorry to hear about your experience. We understand your frustration and are taking steps to resolve this issue immediately.",
                "query": "Thank you for reaching out. We're happy to help and will provide the information you requested.",
            },
            "reply_requirements": {
                "min_length": 40,
                "must_include_greeting": True,
                "must_include_closing": True,
                "forbidden_phrases": [
                    "not my problem",
                    "cannot help",
                    "no idea",
                ],
            },
            "escalation_rules": {
                "always_escalate": [
                    "legal threat",
                    "safety risk",
                    "account restricted",
                    "not happy",
                    "gone downhill",
                    "way too slow",
                    "quality has really",
                ],
                "never_escalate": [],
                "escalate_if_complaint": True,
                "escalate_if_high_urgency": True,
            },
            "reasoning": "Removed 'return' from refund signals to fix query misclassification. Added complaint phrase triggers ('not happy', 'gone downhill', 'quality has') to complaint signals. Added complaint phrases to always_escalate list to fix under-escalation on complaint emails.",
        }

    def _save_strategy(self, strategy: Dict[str, Any], path: str) -> None:
        Path(path).write_text(
            json.dumps(strategy, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _to_str(value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        return bool(value)

    @staticmethod
    def _to_int(value: Any, default: int = 0, minimum: Optional[int] = None) -> int:
        try:
            out = int(value)
        except Exception:
            out = default
        if minimum is not None and out < minimum:
            return minimum
        return out

    @staticmethod
    def _to_str_list(value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, list):
            result: List[str] = []
            _KNOWN_KEYS = ("signal_phrases", "phrases", "keywords", "signals", "terms", "values")
            for item in value:
                if isinstance(item, dict):
                    # Try known keys first
                    found = False
                    for key in _KNOWN_KEYS:
                        inner = item.get(key)
                        if isinstance(inner, list):
                            result.extend(str(v) for v in inner)
                            found = True
                            break
                    if not found:
                        # Single-key dict whose value is a list
                        if len(item) == 1:
                            only_val = next(iter(item.values()))
                            if isinstance(only_val, list):
                                result.extend(str(v) for v in only_val)
                        # else skip the dict entirely
                elif isinstance(item, str):
                    result.append(item)
                else:
                    result.append(str(item))
            return [s for s in result if s.strip()]

        if isinstance(value, str):
            return [value] if value.strip() else []

        return [str(value)] if str(value).strip() else []


__all__ = ["StrategyOptimizer"]
