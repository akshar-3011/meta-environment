#!/usr/bin/env python3
"""LLM-augmented scenario generation pipeline.

Takes the 39 seed scenarios from data.py and generates 60+ variations using
OpenAI's API with controlled mutations (tone, product, complexity). All
generated scenarios are validated against the Scenario Pydantic model and
de-duplicated using token-level Jaccard similarity.

Usage:
    # Generate with OpenAI (requires OPENAI_API_KEY):
    python tools/generate_scenarios.py --count 70

    # Generate without LLM (rule-based augmentation only):
    python tools/generate_scenarios.py --count 70 --no-llm

    # Dry-run (preview without saving):
    python tools/generate_scenarios.py --count 20 --dry-run

Output:
    data/generated_scenarios.py  — Python file with GENERATED_SCENARIOS list
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.models import Scenario
from data.scenario_repository import SCENARIOS

# ─── Constants ───────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.85  # Jaccard threshold for duplicate detection

# Mutation axes for rule-based generation
TONE_MUTATIONS = {
    "angry_to_polite": {
        "replacements": [
            ("terrible", "not ideal"), ("awful", "disappointing"),
            ("disgusting", "concerning"), ("worst", "not the best"),
            ("furious", "concerned"), ("demand", "would appreciate"),
            ("unacceptable", "needs improvement"), ("outraged", "worried"),
            ("never", "rarely"), ("I want", "I would like"),
            ("!!!", "."), ("!!", "."), ("!", "."),
        ],
        "sentiment_map": {"negative": "neutral", "mixed": "positive"},
    },
    "polite_to_urgent": {
        "replacements": [
            ("would like", "NEED"), ("please", "IMMEDIATELY"),
            ("when you can", "RIGHT NOW"), ("appreciate", "demand"),
            ("thank you", "this is URGENT"), ("could you", "you MUST"),
        ],
        "sentiment_map": {"neutral": "negative", "positive": "mixed"},
        "urgency_map": {"low": "high", "medium": "high"},
    },
    "formal_to_casual": {
        "replacements": [
            ("I would like to", "Can I"), ("regarding", "about"),
            ("inquiry", "question"), ("purchase", "thing I bought"),
            ("assistance", "help"), ("Dear", "Hey"), ("Sincerely", "Thanks"),
            ("furthermore", "also"), ("subsequently", "then"),
        ],
        "sentiment_map": {},
    },
}

PRODUCT_CATEGORIES = [
    ("order", "subscription"), ("product", "software license"),
    ("item", "service plan"), ("purchase", "membership"),
    ("refund", "account credit"), ("delivery", "access"),
    ("package", "digital download"), ("shipment", "cloud storage"),
]

COMPLEXITY_MUTATIONS = {
    "add_multi_issue": [
        " Additionally, I also noticed {issue2}.",
        " On top of that, {issue2}.",
        " And another thing — {issue2}.",
    ],
    "secondary_issues": [
        "my account settings are wrong",
        "I was charged twice for the same item",
        "the promotional code didn't apply",
        "my loyalty points disappeared",
        "I never received a confirmation email",
        "the tracking number doesn't work",
        "my password reset isn't going through",
        "the mobile app keeps crashing",
    ],
}

# LLM prompt template
GENERATION_PROMPT = """You are generating customer support email scenarios for an RL training environment.

Given these seed scenarios as examples of the format:
{seed_examples}

Generate {count} NEW, UNIQUE customer support email scenarios. Each must be a valid JSON object with these exact fields:

- "email": string (customer's email, 20-200 chars, realistic and varied)
- "label": one of "refund", "complaint", "query"
- "difficulty": one of "easy", "medium", "hard"
- "sentiment": one of "negative", "neutral", "positive", "mixed"
- "urgency": one of "low", "medium", "high"
- "complexity": integer 1-5
- "requires_escalation": boolean
- "min_reply_length": integer >= 10

Rules:
1. Generate {easy_count} easy (short emails, single issue, complexity 1-2)
2. Generate {medium_count} medium (moderate emails, some context, complexity 2-3)
3. Generate {hard_count} hard (long emails, multiple issues, complexity 4-5)
4. Mix labels roughly equally across difficulties
5. Vary customer tone: angry, polite, confused, sarcastic, desperate
6. Vary product domains: electronics, clothing, software, food, travel, subscription
7. requires_escalation=true only for complaints with high urgency OR complexity >= 4
8. min_reply_length should scale with difficulty: easy=20-40, medium=40-60, hard=60-100
9. Do NOT copy seed emails verbatim — create genuinely new scenarios

Output ONLY a JSON array of objects. No markdown, no commentary."""


# ─── Similarity Engine ──────────────────────────────────────────────────────

def _tokenize(text: str) -> Set[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return set(
        t for t in re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split() if len(t) > 2
    )


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    a_set = _tokenize(a)
    b_set = _tokenize(b)
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def is_duplicate(email: str, existing_emails: List[str], threshold: float = SIMILARITY_THRESHOLD) -> bool:
    """Check if email is too similar to any existing email."""
    for existing in existing_emails:
        if jaccard_similarity(email, existing) >= threshold:
            return True
    return False


# ─── Rule-Based Generation ──────────────────────────────────────────────────

def _apply_tone_mutation(scenario: Dict[str, Any], mutation_name: str) -> Optional[Dict[str, Any]]:
    """Apply a tone mutation to a scenario."""
    mutation = TONE_MUTATIONS.get(mutation_name)
    if not mutation:
        return None

    new = dict(scenario)
    email = new["email"]
    for old_word, new_word in mutation["replacements"]:
        email = email.replace(old_word, new_word)

    if email == new["email"]:
        return None  # No changes made

    new["email"] = email
    sent_map = mutation.get("sentiment_map", {})
    if new["sentiment"] in sent_map:
        new["sentiment"] = sent_map[new["sentiment"]]
    urg_map = mutation.get("urgency_map", {})
    if new.get("urgency") in urg_map:
        new["urgency"] = urg_map[new["urgency"]]
    return new


def _apply_product_mutation(scenario: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Replace product references with different categories."""
    new = dict(scenario)
    email = new["email"]
    applied = False
    for old_term, new_term in random.sample(PRODUCT_CATEGORIES, min(3, len(PRODUCT_CATEGORIES))):
        if old_term.lower() in email.lower():
            email = re.sub(re.escape(old_term), new_term, email, flags=re.IGNORECASE)
            applied = True
    if not applied:
        return None
    new["email"] = email
    return new


def _apply_complexity_mutation(scenario: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Add secondary issues to make scenario more complex."""
    if scenario["complexity"] >= 4:
        return None
    new = dict(scenario)
    template = random.choice(COMPLEXITY_MUTATIONS["add_multi_issue"])
    issue = random.choice(COMPLEXITY_MUTATIONS["secondary_issues"])
    new["email"] = new["email"].rstrip(".!? ") + template.format(issue2=issue)
    new["complexity"] = min(5, new["complexity"] + 1)
    if new["difficulty"] == "easy":
        new["difficulty"] = "medium"
        new["min_reply_length"] = max(new["min_reply_length"], 40)
    elif new["difficulty"] == "medium":
        new["difficulty"] = "hard"
        new["min_reply_length"] = max(new["min_reply_length"], 60)
    return new


def generate_rule_based(
    seeds: List[Dict[str, Any]],
    target_count: int = 70,
    target_distribution: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Generate scenario variations using deterministic rule-based mutations."""
    if target_distribution is None:
        target_distribution = {"easy": 20, "medium": 25, "hard": 25}

    generated: List[Dict[str, Any]] = []
    existing_emails = [s["email"] for s in seeds]
    mutations = [
        ("angry_to_polite", _apply_tone_mutation),
        ("polite_to_urgent", _apply_tone_mutation),
        ("formal_to_casual", _apply_tone_mutation),
        ("product", _apply_product_mutation),
        ("complexity", _apply_complexity_mutation),
    ]

    for seed in seeds:
        for mut_name, mut_fn in mutations:
            if len(generated) >= target_count:
                break

            if mut_name in ("product", "complexity"):
                result = mut_fn(seed)
            else:
                result = mut_fn(seed, mut_name)

            if result is None:
                continue

            # Validate
            try:
                validated = Scenario(**result)
                new_dict = validated.model_dump()
            except Exception:
                continue

            # De-duplicate
            if is_duplicate(new_dict["email"], existing_emails):
                continue

            generated.append(new_dict)
            existing_emails.append(new_dict["email"])

    # Balance difficulties
    return _balance_difficulties(generated, target_distribution)


def _balance_difficulties(
    scenarios: List[Dict[str, Any]],
    targets: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Trim over-represented difficulties to hit targets."""
    by_diff: Dict[str, List] = {"easy": [], "medium": [], "hard": []}
    for s in scenarios:
        by_diff[s["difficulty"]].append(s)

    balanced = []
    for diff, target in targets.items():
        pool = by_diff.get(diff, [])
        balanced.extend(pool[:target])

    return balanced


# ─── LLM-Based Generation ───────────────────────────────────────────────────

def generate_with_llm(
    seeds: List[Dict[str, Any]],
    target_count: int = 70,
    target_distribution: Optional[Dict[str, int]] = None,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """Generate scenarios using OpenAI API."""
    if target_distribution is None:
        target_distribution = {"easy": 20, "medium": 25, "hard": 25}

    try:
        from openai import OpenAI
    except ImportError:
        print("⚠️  openai package not installed. Falling back to rule-based generation.")
        return generate_rule_based(seeds, target_count, target_distribution)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Falling back to rule-based generation.")
        return generate_rule_based(seeds, target_count, target_distribution)

    client = OpenAI(api_key=api_key)
    existing_emails = [s["email"] for s in seeds]
    all_generated: List[Dict[str, Any]] = []

    # Generate in batches to stay within token limits
    batch_size = min(20, target_count)
    batches_needed = (target_count + batch_size - 1) // batch_size

    for batch_idx in range(batches_needed):
        remaining = target_count - len(all_generated)
        if remaining <= 0:
            break

        count = min(batch_size, remaining)
        easy_c = max(1, int(count * target_distribution["easy"] / target_count))
        medium_c = max(1, int(count * target_distribution["medium"] / target_count))
        hard_c = count - easy_c - medium_c

        # Pick diverse seed examples
        seed_examples = json.dumps(random.sample(seeds, min(5, len(seeds))), indent=2)

        prompt = GENERATION_PROMPT.format(
            seed_examples=seed_examples,
            count=count,
            easy_count=easy_c,
            medium_count=medium_c,
            hard_count=hard_c,
        )

        print(f"  📡 LLM batch {batch_idx + 1}/{batches_needed} ({count} scenarios)...", end="", flush=True)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=4000,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            # Parse JSON array from response
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                # Some models wrap in {"scenarios": [...]}
                parsed = parsed.get("scenarios", parsed.get("data", [parsed]))
            if not isinstance(parsed, list):
                parsed = [parsed]

            valid_count = 0
            for raw in parsed:
                try:
                    validated = Scenario(**raw)
                    new_dict = validated.model_dump()

                    if not is_duplicate(new_dict["email"], existing_emails):
                        all_generated.append(new_dict)
                        existing_emails.append(new_dict["email"])
                        valid_count += 1
                except Exception:
                    continue

            print(f" ✅ {valid_count} valid")

        except Exception as exc:
            print(f" ❌ Error: {exc}")
            continue

        # Rate limit
        time.sleep(1)

    return _balance_difficulties(all_generated, target_distribution)


# ─── Synthetic Scenario Templates ───────────────────────────────────────────

# Fallback: hand-crafted template-based generation for when LLM is unavailable
TEMPLATES = {
    "refund": {
        "easy": [
            "I need a refund for order #{order}. The {product} arrived {issue}.",
            "Please process a return for my recent {product} purchase.",
            "Hi, I'd like to get a refund on my {product}. {reason}.",
            "Can I return my {product}? It doesn't match the description.",
            "I want my money back for the {product} I ordered last {time}.",
        ],
        "medium": [
            "I've been trying to get a refund for my {product} for {time} now. It arrived {issue} and I've already contacted support once. {extra}",
            "My order #{order} has multiple problems: the {product} was {issue} and the {product2} was missing entirely. I need a full refund.",
            "I requested a refund {time} ago for a defective {product} but haven't heard back. This is getting frustrating. Please escalate this.",
        ],
        "hard": [
            "I've spent over ${amount} with your company this year alone. My last three orders have all had issues: the {product} was {issue}, the {product2} never arrived, and now I'm being charged for a {product3} I returned {time} ago. I want all three refunds processed immediately AND compensation for my time. If this isn't resolved today, I will be contacting my bank for chargebacks.",
            "This is my FOURTH attempt to get a refund for order #{order}. Every time I call, I get transferred to someone new who doesn't know my case. The {product} was defective, I returned it with tracking #{tracking}, it was received {time} ago, and STILL no refund. I want to speak to a manager and I want this resolved TODAY.",
        ],
    },
    "complaint": {
        "easy": [
            "Your {product} quality has really gone downhill. Not happy.",
            "I'm disappointed with the customer service I received about my {product}.",
            "The {product} I received looks nothing like the photos on your website.",
            "Your shipping is way too slow. My {product} took {time} to arrive.",
        ],
        "medium": [
            "Your {product} broke after just {time} of use. This is unacceptable for the price I paid. Customer service was unhelpful when I called — the rep was rude and dismissive.",
            "I've been a loyal customer for {time} and the quality of your {product} has been declining steadily. The last {product2} I bought was defective out of the box. {extra}",
            "Your website crashed TWICE while I was trying to place an order for a {product}. Then when it finally worked, you charged me double. {extra}",
        ],
        "hard": [
            "I am absolutely FURIOUS. I ordered a {product} for my {person}'s {event} — it arrived {issue}, {time} LATE, and wrapped in newspaper instead of proper packaging. When I called customer service, the first rep hung up on me, the second one put me on hold for 45 minutes, and the third one told me there's nothing they can do. This is the worst customer experience I've ever had. I want an immediate refund, a replacement {product}, AND a formal apology from management. If I don't hear back within 24 hours, I'm going to the BBB and posting reviews everywhere.",
            "Let me describe my NIGHTMARE experience with your company. I've been a premium member paying ${amount}/month for {time}. Your app crashed and corrupted my {product} data — {time2} of work, gone. Your support team said it's 'not their problem' because I should have backed up. ARE YOU SERIOUS? I'm a paying customer! I want my data recovered, a full refund of {time} of membership fees, and an explanation of how this happened.",
        ],
    },
    "query": {
        "easy": [
            "What are your return policies for {product}?",
            "How long does shipping take for {product}? I need it by {date}.",
            "Do you offer student discounts on {product}?",
            "Can I change the color of my {product} order before it ships?",
            "What warranty comes with the {product}?",
        ],
        "medium": [
            "I'm trying to decide between your {product} and {product2}. Could you explain the key differences? Also, do either qualify for the current promotion?",
            "I'd like to upgrade my {product} subscription from Basic to Premium. What features do I get? Is the upgrade prorated or do I pay the full difference?",
            "My {product} seems to be running slow lately. Is there a known issue? I've tried the basic troubleshooting steps on your FAQ already.",
        ],
        "hard": [
            "I'm evaluating your {product} for our company ({size} employees). We need to understand: 1) Volume licensing options beyond {amount} seats, 2) SAML/SSO integration with our existing {product2} setup, 3) Data residency compliance for EU customers under GDPR, 4) SLA guarantees for uptime. Can you provide detailed answers and connect me with an enterprise sales rep?",
            "I'm comparing your {product} against {competitor1} and {competitor2} for an enterprise deployment. Specific questions: Does your API support {protocol}? What's your rate limiting policy? Can we get a dedicated instance? What's the migration path from {product2}? We need answers before our evaluation deadline on {date}.",
        ],
    },
}

FILL_VALUES = {
    "product": ["laptop", "headphones", "smart watch", "wireless charger", "tablet", "monitor", "keyboard", "camera", "printer", "router", "fitness tracker", "earbuds"],
    "product2": ["phone case", "USB cable", "screen protector", "carrying bag", "extended warranty", "software subscription"],
    "product3": ["docking station", "external drive", "webcam"],
    "issue": ["damaged", "scratched", "with a cracked screen", "missing parts", "in the wrong color", "not working", "already opened"],
    "reason": ["It doesn't meet my needs", "I found a better price elsewhere", "It was a gift and they already have one"],
    "time": ["2 weeks", "3 days", "a month", "over a week", "5 days"],
    "time2": ["6 months", "2 years", "a year"],
    "order": ["38472", "91025", "67341", "10294", "55682", "82916"],
    "tracking": ["1Z999AA10123456784", "9400111899223100012", "TRK-20240301-5521"],
    "amount": ["500", "1,200", "2,000", "800", "350", "5,000"],
    "person": ["daughter", "wife", "partner", "mother", "father"],
    "event": ["birthday", "anniversary", "graduation", "wedding"],
    "date": ["next Friday", "March 15th", "end of the month", "this weekend"],
    "extra": ["I need this resolved urgently.", "Please don't just give me a canned response.", "This is really affecting my trust in your company."],
    "size": ["200", "500", "1,500", "50"],
    "competitor1": ["Competitor X", "AlternativePro", "RivalCorp"],
    "competitor2": ["BenchmarkInc", "IndustryLeader", "OpenSource Option"],
    "protocol": ["GraphQL", "REST v3", "gRPC", "WebSockets"],
}


def _fill_template(template: str) -> str:
    """Replace {placeholders} with random values."""
    def replacer(match):
        key = match.group(1)
        values = FILL_VALUES.get(key, [key])
        return random.choice(values)
    return re.sub(r"\{(\w+)\}", replacer, template)


def generate_from_templates(
    target_count: int = 70,
    target_distribution: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Generate scenarios from curated templates with random fill values."""
    if target_distribution is None:
        target_distribution = {"easy": 20, "medium": 25, "hard": 25}

    generated: List[Dict[str, Any]] = []
    existing_emails: List[str] = [s["email"] for s in SCENARIOS]
    n_labels = len(TEMPLATES)  # 3: refund, complaint, query

    for label, difficulties in TEMPLATES.items():
        for difficulty, templates in difficulties.items():
            # Each label gets ~1/3 of the difficulty target
            tier_target = target_distribution.get(difficulty, 20)
            per_label_target = max(2, (tier_target + n_labels - 1) // n_labels)

            attempts = 0
            label_diff_count = 0

            while label_diff_count < per_label_target and attempts < per_label_target * 8:
                attempts += 1
                template = random.choice(templates)
                email = _fill_template(template)

                if is_duplicate(email, existing_emails):
                    continue

                # Derive metadata from difficulty + label
                sentiment = random.choice(
                    ["negative", "mixed"] if label == "complaint"
                    else ["neutral", "positive"] if label == "query"
                    else ["negative", "neutral", "mixed"]
                )
                urgency = (
                    "high" if difficulty == "hard"
                    else "medium" if difficulty == "medium"
                    else "low"
                )
                complexity = (
                    random.randint(4, 5) if difficulty == "hard"
                    else random.randint(2, 3) if difficulty == "medium"
                    else random.randint(1, 2)
                )
                requires_escalation = (
                    label == "complaint"
                    and (urgency in ("high", "medium") or complexity >= 4)
                )
                min_reply_length = (
                    random.randint(60, 100) if difficulty == "hard"
                    else random.randint(40, 60) if difficulty == "medium"
                    else random.randint(20, 40)
                )

                raw = {
                    "email": email,
                    "label": label,
                    "difficulty": difficulty,
                    "sentiment": sentiment,
                    "urgency": urgency,
                    "complexity": complexity,
                    "requires_escalation": requires_escalation,
                    "min_reply_length": min_reply_length,
                }

                try:
                    validated = Scenario(**raw)
                    new_dict = validated.model_dump()
                    generated.append(new_dict)
                    existing_emails.append(new_dict["email"])
                    label_diff_count += 1
                except Exception:
                    continue

    return generated


# ─── Output ─────────────────────────────────────────────────────────────────

def save_generated_scenarios(scenarios: List[Dict[str, Any]], output_path: str) -> None:
    """Save generated scenarios as a Python module."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        '"""Auto-generated scenario data — DO NOT EDIT MANUALLY.',
        '',
        f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}',
        f'Count: {len(scenarios)}',
        '"""',
        '',
        '',
        'GENERATED_SCENARIOS = [',
    ]
    for s in scenarios:
        # Use repr() for Python-valid output (True/False, not true/false)
        lines.append(f"    {repr(s)},")
    lines.append("]")
    lines.append("")

    path.write_text("\n".join(lines))
    print(f"\n💾 Saved {len(scenarios)} scenarios to {path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate scenario variations")
    parser.add_argument("--count", type=int, default=70, help="Target number of scenarios")
    parser.add_argument("--output", type=str, default="data/generated_scenarios.py")
    parser.add_argument("--no-llm", action="store_true", help="Use rule-based + template generation only")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    target_dist = {"easy": 20, "medium": 25, "hard": 25}

    print(f"\n{'═' * 60}")
    print(f"  SCENARIO GENERATION PIPELINE")
    print(f"  Seeds: {len(SCENARIOS)} existing scenarios")
    print(f"  Target: {args.count} new scenarios")
    print(f"  Mode: {'Rule-based + Templates' if args.no_llm else 'LLM-augmented'}")
    print(f"{'═' * 60}\n")

    # Phase 1: Template-based generation (always run — reliable baseline)
    print("📝 Phase 1: Template-based generation...")
    template_scenarios = generate_from_templates(
        target_count=args.count,
        target_distribution=target_dist,
    )
    print(f"   Generated {len(template_scenarios)} from templates")

    # Phase 2: Rule-based mutations
    print("🔄 Phase 2: Rule-based mutations...")
    rule_scenarios = generate_rule_based(
        SCENARIOS,
        target_count=args.count - len(template_scenarios),
        target_distribution=target_dist,
    )
    print(f"   Generated {len(rule_scenarios)} from mutations")

    # Combine and de-duplicate
    all_emails = [s["email"] for s in SCENARIOS]
    combined: List[Dict[str, Any]] = []
    for s in template_scenarios + rule_scenarios:
        if not is_duplicate(s["email"], all_emails, SIMILARITY_THRESHOLD):
            combined.append(s)
            all_emails.append(s["email"])

    # Phase 3: LLM augmentation (if needed and enabled)
    if not args.no_llm and len(combined) < args.count:
        shortfall = args.count - len(combined)
        print(f"\n🤖 Phase 3: LLM augmentation ({shortfall} more needed)...")
        llm_scenarios = generate_with_llm(
            SCENARIOS, target_count=shortfall, model=args.model,
        )
        for s in llm_scenarios:
            if not is_duplicate(s["email"], all_emails, SIMILARITY_THRESHOLD):
                combined.append(s)
                all_emails.append(s["email"])

    # Stats
    by_diff = {"easy": 0, "medium": 0, "hard": 0}
    by_label = {"refund": 0, "complaint": 0, "query": 0}
    for s in combined:
        by_diff[s["difficulty"]] = by_diff.get(s["difficulty"], 0) + 1
        by_label[s["label"]] = by_label.get(s["label"], 0) + 1

    print(f"\n{'─' * 40}")
    print(f"  Total generated: {len(combined)}")
    print(f"  By difficulty: {by_diff}")
    print(f"  By label: {by_label}")
    print(f"  Escalation: {sum(1 for s in combined if s['requires_escalation'])}/{len(combined)}")
    print(f"{'─' * 40}")

    if not args.dry_run:
        save_generated_scenarios(combined, args.output)
    else:
        print("\n  [DRY RUN] Scenarios not saved.")
        print(f"\n  Sample scenario:\n{json.dumps(combined[0] if combined else {}, indent=2)}")

    return combined


if __name__ == "__main__":
    main()
