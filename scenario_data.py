"""
Enhanced scenario data with difficulty levels, sentiment, and metadata.
Each scenario includes:
  - email: Customer message
  - label: True category (refund/complaint/query)
  - difficulty: easy/medium/hard
  - sentiment: negative/neutral/positive
  - urgency: low/medium/high
  - complexity: 1-5 scale
  - requires_escalation: bool
  - min_reply_length: Minimum response length expectation
"""

SCENARIOS = [
    # Easy scenarios - clear intent, straightforward handling
    {
        "email": "I want a refund for my order",
        "label": "refund",
        "difficulty": "easy",
        "sentiment": "neutral",
        "urgency": "low",
        "complexity": 1,
        "requires_escalation": False,
        "min_reply_length": 30,
    },
    {
        "email": "Can you tell me delivery status?",
        "label": "query",
        "difficulty": "easy",
        "sentiment": "neutral",
        "urgency": "low",
        "complexity": 1,
        "requires_escalation": False,
        "min_reply_length": 20,
    },
    {
        "email": "Where is my package?",
        "label": "query",
        "difficulty": "easy",
        "sentiment": "neutral",
        "urgency": "low",
        "complexity": 1,
        "requires_escalation": False,
        "min_reply_length": 20,
    },
    {
        "email": "How long does delivery take?",
        "label": "query",
        "difficulty": "easy",
        "sentiment": "neutral",
        "urgency": "low",
        "complexity": 1,
        "requires_escalation": False,
        "min_reply_length": 25,
    },
    # Medium scenarios - mixed signals, requires judgment
    {
        "email": "Your service is terrible, I'm very unhappy",
        "label": "complaint",
        "difficulty": "medium",
        "sentiment": "negative",
        "urgency": "medium",
        "complexity": 2,
        "requires_escalation": True,
        "min_reply_length": 40,
    },
    {
        "email": "I need to cancel and get refund urgently",
        "label": "refund",
        "difficulty": "medium",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 2,
        "requires_escalation": False,
        "min_reply_length": 50,
    },
    {
        "email": "Product arrived damaged, I need help",
        "label": "complaint",
        "difficulty": "medium",
        "sentiment": "negative",
        "urgency": "medium",
        "complexity": 2,
        "requires_escalation": True,
        "min_reply_length": 45,
    },
    {
        "email": "Refund my money immediately",
        "label": "refund",
        "difficulty": "medium",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 2,
        "requires_escalation": False,
        "min_reply_length": 40,
    },
    {
        "email": "This is unacceptable service",
        "label": "complaint",
        "difficulty": "medium",
        "sentiment": "negative",
        "urgency": "medium",
        "complexity": 2,
        "requires_escalation": True,
        "min_reply_length": 50,
    },
    {
        "email": "I was charged twice!",
        "label": "complaint",
        "difficulty": "medium",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 2,
        "requires_escalation": True,
        "min_reply_length": 45,
    },
    # -----------------------------------------------------------------------
    # Additional easy scenarios
    # -----------------------------------------------------------------------
    {
        "email": "Do you offer express shipping?",
        "label": "query",
        "difficulty": "easy",
        "sentiment": "neutral",
        "urgency": "low",
        "complexity": 1,
        "requires_escalation": False,
        "min_reply_length": 25,
    },
    {
        "email": "What are your working hours?",
        "label": "query",
        "difficulty": "easy",
        "sentiment": "neutral",
        "urgency": "low",
        "complexity": 1,
        "requires_escalation": False,
        "min_reply_length": 20,
    },
    {
        "email": "Can I change my delivery address?",
        "label": "query",
        "difficulty": "easy",
        "sentiment": "neutral",
        "urgency": "low",
        "complexity": 1,
        "requires_escalation": False,
        "min_reply_length": 25,
    },
    {
        "email": "How do I return my item?",
        "label": "query",
        "difficulty": "easy",
        "sentiment": "neutral",
        "urgency": "low",
        "complexity": 1,
        "requires_escalation": False,
        "min_reply_length": 20,
    },
    {
        "email": "Please process a refund for my last order.",
        "label": "refund",
        "difficulty": "easy",
        "sentiment": "neutral",
        "urgency": "low",
        "complexity": 1,
        "requires_escalation": False,
        "min_reply_length": 25,
    },
    {
        "email": "Are you open on weekends?",
        "label": "query",
        "difficulty": "easy",
        "sentiment": "neutral",
        "urgency": "low",
        "complexity": 1,
        "requires_escalation": False,
        "min_reply_length": 20,
    },
    {
        "email": "I need a refund for the shoes.",
        "label": "refund",
        "difficulty": "easy",
        "sentiment": "neutral",
        "urgency": "low",
        "complexity": 1,
        "requires_escalation": False,
        "min_reply_length": 25,
    },

    # -----------------------------------------------------------------------
    # Additional medium scenarios
    # -----------------------------------------------------------------------
    {
        "email": "Your support team is not responding",
        "label": "complaint",
        "difficulty": "medium",
        "sentiment": "negative",
        "urgency": "medium",
        "complexity": 2,
        "requires_escalation": True,
        "min_reply_length": 45,
    },
    {
        "email": "Hi there, I hope you're having a lovely day! I was wondering if it might at all be possible to arrange a refund for order #4521 when you get a chance? No rush at all!",
        "label": "refund",
        "difficulty": "medium",
        "sentiment": "positive",
        "urgency": "low",
        "complexity": 2,
        "requires_escalation": False,
        "min_reply_length": 35,
    },
    {
        "email": "I wish to formally request a full reimbursement for order number 88821 in accordance with your stated returns policy.",
        "label": "refund",
        "difficulty": "medium",
        "sentiment": "neutral",
        "urgency": "low",
        "complexity": 2,
        "requires_escalation": False,
        "min_reply_length": 40,
    },
    {
        "email": "The shirt is okay, but it fits weird. Not sure if I want to keep it or what.",
        "label": "query",
        "difficulty": "medium",
        "sentiment": "mixed",
        "urgency": "low",
        "complexity": 2,
        "requires_escalation": False,
        "min_reply_length": 35,
    },
    {
        "email": "I liked the product but delivery was a bit frustrating. Is there a way to return just part of it?",
        "label": "refund",
        "difficulty": "medium",
        "sentiment": "mixed",
        "urgency": "medium",
        "complexity": 3,
        "requires_escalation": False,
        "min_reply_length": 40,
    },
    {
        "email": "Well that was somewhat disappointing, though not entirely your fault. How do we fix this?",
        "label": "query",
        "difficulty": "medium",
        "sentiment": "mixed",
        "urgency": "medium",
        "complexity": 2,
        "requires_escalation": False,
        "min_reply_length": 40,
    },
    {
        "email": "I mean, it's not the worst thing ever, but I kind of want my money back.",
        "label": "refund",
        "difficulty": "medium",
        "sentiment": "mixed",
        "urgency": "low",
        "complexity": 2,
        "requires_escalation": False,
        "min_reply_length": 35,
    },

    # ===================================================================
    # HARD SCENARIOS — 15 genuinely adversarial, designed to challenge
    # frontier LLMs with sarcasm, multi-intent, threats, cross-lingual
    # cues, and subtle category distinctions
    # ===================================================================

    # H1: Unauthorized charge + loyalty + compensation demand → refund
    {
        "email": "My account shows a charge I never authorized. I've been a loyal customer for 6 years and this is how you treat me? I want this reversed AND compensation for my time.",
        "label": "refund",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 5,
        "requires_escalation": True,
        "min_reply_length": 80,
    },

    # H2: Mild dissatisfaction, explicitly not asking for refund → complaint
    {
        "email": "The product works fine I guess but it's nothing like what was advertised. Not asking for a refund but wanted someone to know.",
        "label": "complaint",
        "difficulty": "hard",
        "sentiment": "mixed",
        "urgency": "low",
        "complexity": 4,
        "requires_escalation": False,
        "min_reply_length": 50,
    },

    # H3: Purely informational question about return policy → query
    {
        "email": "Quick question — does your return policy apply to items purchased during the sale event last November? Asking for a friend.",
        "label": "query",
        "difficulty": "hard",
        "sentiment": "neutral",
        "urgency": "low",
        "complexity": 3,
        "requires_escalation": False,
        "min_reply_length": 40,
    },

    # H4: Repeat disappointment, no explicit ask → complaint
    {
        "email": "I'm not angry. I'm just incredibly disappointed. Every single time I order from you something goes wrong. I keep giving second chances and this is the result.",
        "label": "complaint",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "medium",
        "complexity": 5,
        "requires_escalation": True,
        "min_reply_length": 70,
    },

    # H5: Exchange + stock question (multi-intent informational) → query
    {
        "email": "Can I exchange an item rather than get a refund? Also can you tell me if the replacement is even in stock before I ship back?",
        "label": "query",
        "difficulty": "hard",
        "sentiment": "neutral",
        "urgency": "medium",
        "complexity": 4,
        "requires_escalation": False,
        "min_reply_length": 60,
    },

    # H6: Broken promise on refund timeline → refund
    {
        "email": "Your customer service rep told me to email here. He said you'd process the refund in 3-5 days. It's been 11 days.",
        "label": "refund",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 5,
        "requires_escalation": True,
        "min_reply_length": 70,
    },

    # H7: Empty box delivery (sarcastic + enraged) → complaint
    {
        "email": "Wow, just wow. Package arrived with a note saying 'sorry for the delay' and nothing inside. The box was empty. EMPTY.",
        "label": "complaint",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 5,
        "requires_escalation": True,
        "min_reply_length": 80,
    },

    # H8: GDPR data question (technical, informational) → query
    {
        "email": "I need to understand your GDPR data deletion policy and also want to know how long you store purchase history.",
        "label": "query",
        "difficulty": "hard",
        "sentiment": "neutral",
        "urgency": "medium",
        "complexity": 4,
        "requires_escalation": False,
        "min_reply_length": 60,
    },

    # H9: Inferior quality, goodwill request from loyal customer → complaint
    {
        "email": "Technically the item isn't broken. But it's clearly inferior quality to what I paid for. Is there any goodwill gesture you can offer a long-term customer?",
        "label": "complaint",
        "difficulty": "hard",
        "sentiment": "mixed",
        "urgency": "low",
        "complexity": 4,
        "requires_escalation": False,
        "min_reply_length": 60,
    },

    # H10: Competitor threat + refund ultimatum → refund
    {
        "email": "Your competitor just offered me a full refund no questions asked. I'm giving you one chance to match that before I switch.",
        "label": "refund",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 5,
        "requires_escalation": True,
        "min_reply_length": 80,
    },

    # H11: Cancelled but still charged → refund
    {
        "email": "I placed an order, cancelled it within 10 minutes, got a cancellation email but was still charged. The money left my account.",
        "label": "refund",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 5,
        "requires_escalation": True,
        "min_reply_length": 70,
    },

    # H12: Warranty coverage question (informational) → query
    {
        "email": "Does the warranty cover accidental damage? I dropped my item and I'm not sure if it's a manufacturing defect or my fault.",
        "label": "query",
        "difficulty": "hard",
        "sentiment": "neutral",
        "urgency": "medium",
        "complexity": 3,
        "requires_escalation": False,
        "min_reply_length": 50,
    },

    # H13: Account retaliation accusation → complaint
    {
        "email": "I left a 1-star review and within hours my account was restricted. I want an explanation and restoration of my account.",
        "label": "complaint",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 5,
        "requires_escalation": True,
        "min_reply_length": 80,
    },

    # H14: Misdelivered order, needs reship or refund today → refund
    {
        "email": "My order was delivered to the wrong address according to the tracking. The photo shows someone else's door. I need this reshipped or refunded today.",
        "label": "refund",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 5,
        "requires_escalation": True,
        "min_reply_length": 80,
    },

    # H15: Gift return logistics question → query
    {
        "email": "I bought a gift for someone. They didn't like it. Can they return it directly using my order number or does it need to come back through me?",
        "label": "query",
        "difficulty": "hard",
        "sentiment": "neutral",
        "urgency": "low",
        "complexity": 3,
        "requires_escalation": False,
        "min_reply_length": 50,
    },
]