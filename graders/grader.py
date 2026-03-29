"""
Enhanced grading engine with weighted reward components,
penalty system, deterministic scoring, and explanations.
"""

from typing import Optional, Dict, Tuple

CATEGORY_OPTIONS = ["refund", "complaint", "query"]

RELATED_LABELS = {
    "refund": ["complaint"],
    "complaint": ["refund"],
    "query": []
}

REQUIRED_KEYWORDS = {
    "refund": ["refund", "return", "process", "business days"],
    "complaint": ["sorry", "apologize", "understand", "resolve"],
    "query": ["happy to help", "please", "contact", "let us know", "information"],
}

ESCALATION_REQUIRED = {
    "complaint": True,
    "refund": False,
    "query": False,
}

HARSH_PHRASES = ["not my problem", "figure it out", "stop emailing", "nothing we can do"]


def grade_classification(
    predicted_category: str,
    actual_category: str,
    scenario_difficulty: str = "easy",
) -> Tuple[float, str]:
    pred = predicted_category.lower().strip()

    if pred == actual_category:
        return 1.0, f"Correct classification: {actual_category}"

    related = RELATED_LABELS.get(actual_category, [])
    if pred in related:
        if scenario_difficulty == "easy":
            score = 0.3
        elif scenario_difficulty == "medium":
            score = 0.4
        else:
            score = 0.2
        return score, f"Partially correct: chose {pred}, should be {actual_category}"

    return 0.0, f"Wrong classification: {pred} (actual: {actual_category})"


def grade_reply(
    response: str,
    actual_category: str,
    min_length: int = 30,
) -> Tuple[float, str]:
    text = response.lower().strip()
    score = 0.0
    components = []

    if len(text) >= min_length:
        score += 0.35
        components.append(f"length_ok({len(text)})")
    elif len(text) > 10:
        score += 0.1
        components.append(f"too_short({len(text)}/{min_length})")

    if len(text) < 500:
        score += 0.15
        components.append("concise")
    else:
        score -= 0.1
        components.append("too_verbose")

    keywords = REQUIRED_KEYWORDS.get(actual_category, [])
    matched = sum(1 for kw in keywords if kw in text)
    if matched > 0:
        keyword_score = min(0.4, 0.15 * matched)
        score += keyword_score
        components.append(f"keywords({matched}/{len(keywords)})")
    else:
        score -= 0.2
        components.append("no_keywords")

    if any(phrase in text for phrase in HARSH_PHRASES):
        score -= 0.15
        components.append("harsh_tone")

    solution_words = ["help", "assist", "resolve", "fix", "solution", "refund", "process"]
    if any(word in text for word in solution_words):
        score += 0.1
        components.append("solution_oriented")

    score = max(0.0, min(1.0, score))
    return score, f"Reply scoring: {', '.join(components)}"


def grade_escalation(
    escalation_decision: str,
    actual_category: str,
    step_count: int,
) -> Tuple[float, str]:
    decision = escalation_decision.lower().strip()
    did_escalate = decision in ["yes", "true", "urgent", "1", "escalate"]
    should_escalate = ESCALATION_REQUIRED.get(actual_category, False)

    score = 0.0
    reason = ""

    if should_escalate and did_escalate:
        score = 1.0
        reason = f"Correctly escalated {actual_category}"
    elif not should_escalate and not did_escalate:
        score = 0.9
        reason = f"Correctly handled {actual_category} without escalation"
    elif should_escalate and not did_escalate:
        score = 0.1
        reason = f"Should escalate {actual_category}, but did not"
    elif not should_escalate and did_escalate:
        score = 0.3
        reason = f"Over-escalated {actual_category}"

    if did_escalate and step_count < 2:
        score *= 0.7
        reason += " (early escalation penalty)"

    if did_escalate and step_count == 2:
        score = min(1.0, score + 0.1)
        reason += " (good timing)"

    return score, reason


def calculate_step_reward(
    action_type: str,
    content: str,
    actual_category: str,
    step_count: int,
    scenario_difficulty: str = "easy",
    min_reply_length: int = 30,
    previous_actions: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict]:
    previous_actions = previous_actions or {}
    breakdown = {
        "step_count": step_count,
        "action_type": action_type,
        "category": actual_category,
    }

    if action_type == "classify":
        score, explanation = grade_classification(content, actual_category, scenario_difficulty)
        breakdown["raw_score"] = score
        breakdown["explanation"] = explanation
        breakdown["weight"] = 0.4
        reward = score * 0.4

    elif action_type == "reply":
        classification_reward = previous_actions.get("classify", 0.0)
        consistency_penalty = 0.0 if classification_reward > 0.5 else 0.2

        score, explanation = grade_reply(content, actual_category, min_reply_length)
        breakdown["raw_score"] = score
        breakdown["explanation"] = explanation
        breakdown["consistency_penalty"] = consistency_penalty
        breakdown["weight"] = 0.35
        reward = (score - consistency_penalty) * 0.35

    elif action_type == "escalate":
        score, explanation = grade_escalation(content, actual_category, step_count)
        breakdown["raw_score"] = score
        breakdown["explanation"] = explanation
        breakdown["weight"] = 0.25
        reward = score * 0.25

    else:
        reward = 0.0
        breakdown["error"] = f"Unknown action type: {action_type}"

    reward = max(0.0, min(1.0, reward))
    breakdown["final_reward"] = reward

    return reward, breakdown
