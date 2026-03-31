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
    # Hard scenarios - ambiguous, edge cases, multi-intent
    {
        "email": "I need to cancel and get refund urgently. Your support team is not responding!",
        "label": "complaint",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 4,
        "requires_escalation": True,
        "min_reply_length": 60,
    },
    {
        "email": "Product quality is poor but I'm willing to try again if you help",
        "label": "complaint",
        "difficulty": "hard",
        "sentiment": "mixed",
        "urgency": "medium",
        "complexity": 4,
        "requires_escalation": True,
        "min_reply_length": 70,
    },
    {
        "email": "Do you offer express shipping? I had a bad experience last time",
        "label": "query",
        "difficulty": "hard",
        "sentiment": "mixed",
        "urgency": "medium",
        "complexity": 3,
        "requires_escalation": False,
        "min_reply_length": 50, 
    },
    {
        "email": "Extremely disappointed. Refund not received. What's happening?",
        "label": "complaint",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 4,
        "requires_escalation": True,
        "min_reply_length": 60,
    },
    # Additional balanced set
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
        "email": "Your support team is not responding",
        "label": "complaint",
        "difficulty": "medium",
        "sentiment": "negative",
        "urgency": "medium",
        "complexity": 2,
        "requires_escalation": True,
        "min_reply_length": 45,
    },

    # -----------------------------------------------------------------------
    # Edge-case scenarios — designed to challenge naive agents
    # -----------------------------------------------------------------------

    # 1. Sarcasm masking a complaint — reads like mild feedback, is actually angry
    {
        "email": "Oh great, another delayed package. Really loving the experience so far.",
        "label": "complaint",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "medium",
        "complexity": 4,
        "requires_escalation": True,
        "min_reply_length": 50,
    },

    # 2. Refund request buried inside genuine praise — easy to misread as a query
    {
        "email": "Your team has always been so helpful and I love your products! Quick question — would it be possible to get a refund on my most recent order?",
        "label": "refund",
        "difficulty": "hard",
        "sentiment": "positive",
        "urgency": "low",
        "complexity": 3,
        "requires_escalation": False,
        "min_reply_length": 35,
    },

    # 3. Query phrased with frustration — sounds like a complaint but is asking for info
    {
        "email": "I genuinely don't understand why my order still hasn't arrived. Can someone please explain what's happening?",
        "label": "query",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "medium",
        "complexity": 3,
        "requires_escalation": False,
        "min_reply_length": 40,
    },

    # 4. Passive-aggressive query — frustrated tone but genuinely asking a question
    {
        "email": "I was just wondering... is it totally normal for orders to take three weeks to arrive? Just curious.",
        "label": "query",
        "difficulty": "hard",
        "sentiment": "mixed",
        "urgency": "low",
        "complexity": 3,
        "requires_escalation": False,
        "min_reply_length": 35,
    },

    # 5. Reputation threat — strong escalation signal regardless of underlying issue
    {
        "email": "This is the last time I contact you nicely. Fix this immediately or I will be leaving reviews everywhere and disputing the charge.",
        "label": "complaint",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 5,
        "requires_escalation": True,
        "min_reply_length": 60,
    },

    # 6. Broken product + refund demand — primary intent is complaint, not refund
    {
        "email": "I am absolutely furious. The item arrived completely smashed and I want my money back immediately. This is unacceptable.",
        "label": "complaint",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 4,
        "requires_escalation": True,
        "min_reply_length": 55,
    },

    # 7. Hyper-polite refund — politeness hides urgency; agent might downgrade to query
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

    # 8. Multi-issue overload — refund + service failure + unresponsive team; escalate required
    {
        "email": "Can I get a refund? Also your website has been broken for two days and nobody has replied to any of my emails.",
        "label": "complaint",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 5,
        "requires_escalation": True,
        "min_reply_length": 65,
    },

    # 9. Implicit repeat contact — signals escalation needed without saying so explicitly
    {
        "email": "This is the fourth time I am writing about the same issue. I have not received any response and my problem is still not resolved.",
        "label": "complaint",
        "difficulty": "hard",
        "sentiment": "negative",
        "urgency": "high",
        "complexity": 4,
        "requires_escalation": True,
        "min_reply_length": 55,
    },

    # 10. Overly formal / corporate tone — unusual register; agent may misread as internal
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
]