"""
Enhanced grading engine with:
  - Weighted reward components
  - Penalty system for poor decisions
  - Deterministic, interpretable scoring
  - Explanations for each grade
"""

from typing import Optional, Dict, Tuple


# ============================================================================
# Knowledge bases for deterministic grading
# ============================================================================

CATEGORY_OPTIONS = ["refund", "complaint", "query"]

RELATED_LABELS = {
    "refund": ["complaint"],  # Refund is related to complaint (damaged product)
    "complaint": ["refund"],   # Complaint is related to refund (price issues)
    "query": []                # Query stands alone
}

REQUIRED_KEYWORDS = {
    "refund": ["refund", "return", "money"],
    "complaint": ["sorry", "apologize", "issue", "problem", "wrong"],
    "query": ["help", "information", "how", "what", "where"]
}

ESCALATION_REQUIRED = {
    "complaint": True,
    "refund": False,
    "query": False,
}

# Sentiment indicators for enhanced grading
NEGATIVE_INDICATORS = ["terrible", "unhappy", "unacceptable", "disappointed", "angry", "poor"]
URGENT_INDICATORS = ["immediately", "urgent", "urgent", "asap", "emergency", "critical"]


# ============================================================================
# Grading functions with rich feedback
# ============================================================================

def grade_classification(
    predicted_category: str,
    actual_category: str,
    scenario_difficulty: str = "easy",
) -> Tuple[float, str]:
    """
    Grade the classification task.
    
    Returns:
        (score, explanation) tuple
    
    Scoring logic:
      - Perfect match (easy/medium/hard): 1.0
      - Related category (only medium+): 0.4
      - Wrong: 0.0
    """
    pred = predicted_category.lower().strip()
    
    # Perfect match - always best
    if pred == actual_category:
        return 1.0, f"✓ Correct classification: {actual_category}"
    
    # Related categories - partial credit (harder scenarios penalize more)
    related = RELATED_LABELS.get(actual_category, [])
    if pred in related:
        if scenario_difficulty == "easy":
            score = 0.3  # Stricter on easy
        elif scenario_difficulty == "medium":
            score = 0.4
        else:
            score = 0.2  # Stricter on hard (should know better)
        return score, f"~ Partially correct: chose {pred}, should be {actual_category}"
    
    # Wrong
    return 0.0, f"✗ Wrong classification: {pred} (actual: {actual_category})"


def grade_reply(
    response: str,
    actual_category: str,
    min_length: int = 30,
) -> Tuple[float, str]:
    """
    Grade the reply generation task.
    
    Components:
      - Coherence: Has minimum length, good structure
      - Empathy: Contains required keywords for category
      - Appropriateness: Matches category tone
    
    Returns:
        (score, explanation) tuple
    """
    text = response.lower().strip()
    score = 0.0
    components = []
    
    # Length check (most important - shows effort)
    if len(text) >= min_length:
        score += 0.35
        components.append(f"length_ok({len(text)})")
    elif len(text) > 10:
        score += 0.1
        components.append(f"too_short({len(text)}/{min_length})")
    
    # Length upper bound - avoid verbosity
    if len(text) < 500:
        score += 0.15
        components.append("concise")
    else:
        score -= 0.1  # Penalty for excessive length
        components.append("too_verbose")
    
    # Keyword matching (shows understanding of category)
    keywords = REQUIRED_KEYWORDS.get(actual_category, [])
    matched = sum(1 for kw in keywords if kw in text)
    if matched > 0:
        keyword_score = min(0.4, 0.15 * matched)
        score += keyword_score
        components.append(f"keywords({matched}/{len(keywords)})")
    else:
        score -= 0.2  # Penalty for missing category-specific language
        components.append("no_keywords")
    
    # Tone check - penalize harsh/dismissive responses
    if any(neg in text for neg in ["no", "won't", "can't", "never"]):
        score -= 0.15
        components.append("harsh_tone")
    
    # Encourage problem-solving language
    solution_words = ["help", "assist", "resolve", "fix", "solution", "refund"]
    if any(word in text for word in solution_words):
        score += 0.1
        components.append("solution_oriented")
    
    score = max(0.0, min(1.0, score))
    explanation = f"Reply scoring: {', '.join(components)}"
    
    return score, explanation


def grade_escalation(
    escalation_decision: str,
    actual_category: str,
    step_count: int,
) -> Tuple[float, str]:
    """
    Grade the escalation decision.
    
    Rules:
      - Complaints MUST be escalated (high reward if yes, low if no)
      - Query/Refund should NOT be escalated unless critical
      - Early escalation (step < 2) is discouraged
    
    Returns:
        (score, explanation) tuple
    """
    decision = escalation_decision.lower().strip()
    did_escalate = decision in ["yes", "true", "urgent", "1", "escalate"]
    should_escalate = ESCALATION_REQUIRED.get(actual_category, False)
    
    score = 0.0
    reason = ""
    
    # Correct decision for category
    if should_escalate and did_escalate:
        score = 1.0
        reason = f"✓ Correctly escalated {actual_category}"
    elif not should_escalate and not did_escalate:
        score = 0.9
        reason = f"✓ Correctly handled {actual_category} without escalation"
    elif should_escalate and not did_escalate:
        score = 0.1
        reason = f"✗ Should escalate {actual_category}, but didn't"
    elif not should_escalate and did_escalate:
        score = 0.3
        reason = f"⚠ Over-escalated {actual_category}"
    
    # Penalty for early escalation (bypass the workflow)
    if did_escalate and step_count < 2:
        score *= 0.7
        reason += " (early escalation penalty)"
    
    # Bonus for escalating at the right time
    if did_escalate and step_count == 2:
        score = min(1.0, score + 0.1)
        reason += " (good timing)"
    
    return score, reason


# ============================================================================
# Composite reward calculation
# ============================================================================

def calculate_step_reward(
    action_type: str,
    content: str,
    actual_category: str,
    step_count: int,
    scenario_difficulty: str = "easy",
    min_reply_length: int = 30,
    previous_actions: Optional[Dict[str, str]] = None,
) -> Tuple[float, Dict]:
    """
    Calculate total reward for a step with component breakdown.
    
    Applies weights based on task importance:
      - Classification: 40% (foundational)
      - Reply: 35% (main task)
      - Escalation: 25% (decision making)
    
    Also applies penalties for:
      - Inconsistency (e.g., escalate without classifying)
      - Repeated mistakes
    
    Returns:
        (total_reward, breakdown_dict)
    """
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
        # Penalize if classification wasn't done or was wrong
        classification_reward = previous_actions.get("classify_reward", 0.0)
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
    
    # Clamp to [0, 1]
    reward = max(0.0, min(1.0, reward))
    breakdown["final_reward"] = reward
    
    return reward, breakdown

