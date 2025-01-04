"""
Mock responses for testing LLM interactions.
"""
from typing import Dict, Any, List

# Mock responses for content analysis
MOCK_CONTENT_ANALYSIS = {
    "positive_tweet": {
        "sentiment": "positive",
        "topics": ["technology", "ai", "innovation"],
        "engagement_potential": 0.85,
        "key_points": [
            "Exciting developments in AI",
            "New technological breakthroughs",
            "Positive impact on society"
        ],
        "tone": "enthusiastic",
        "audience_fit": 0.9
    },
    "negative_tweet": {
        "sentiment": "negative",
        "topics": ["complaints", "issues", "problems"],
        "engagement_potential": 0.3,
        "key_points": [
            "Technical difficulties",
            "User frustration",
            "Service issues"
        ],
        "tone": "frustrated",
        "audience_fit": 0.2
    },
    "neutral_tweet": {
        "sentiment": "neutral",
        "topics": ["updates", "announcements"],
        "engagement_potential": 0.5,
        "key_points": [
            "Status update",
            "General information",
            "Routine announcement"
        ],
        "tone": "informative",
        "audience_fit": 0.6
    }
}

# Mock responses for user analysis
MOCK_USER_ANALYSIS = {
    "active_user": {
        "engagement_score": 0.85,
        "topic_alignment": 0.9,
        "influence_score": 0.75,
        "interaction_quality": 0.8,
        "risk_assessment": "low",
        "recommendation": "follow",
        "confidence": 0.9
    },
    "inactive_user": {
        "engagement_score": 0.2,
        "topic_alignment": 0.4,
        "influence_score": 0.1,
        "interaction_quality": 0.3,
        "risk_assessment": "medium",
        "recommendation": "ignore",
        "confidence": 0.7
    },
    "risky_user": {
        "engagement_score": 0.6,
        "topic_alignment": 0.3,
        "influence_score": 0.8,
        "interaction_quality": 0.4,
        "risk_assessment": "high",
        "recommendation": "avoid",
        "confidence": 0.85
    }
}

# Mock responses for conversation analysis
MOCK_CONVERSATION_ANALYSIS = {
    "friendly_chat": {
        "tone": "friendly",
        "engagement_level": "high",
        "topics": ["casual", "personal", "interests"],
        "suggested_responses": [
            "Thanks for sharing! That's really interesting.",
            "I'd love to hear more about that.",
            "Great point! I agree completely."
        ],
        "response_tone": "warm",
        "confidence": 0.9
    },
    "technical_discussion": {
        "tone": "professional",
        "engagement_level": "medium",
        "topics": ["technical", "development", "features"],
        "suggested_responses": [
            "That's a good technical approach.",
            "Have you considered alternative solutions?",
            "Let's explore this further."
        ],
        "response_tone": "analytical",
        "confidence": 0.85
    },
    "complaint_handling": {
        "tone": "frustrated",
        "engagement_level": "high",
        "topics": ["issues", "complaints", "support"],
        "suggested_responses": [
            "I understand your frustration.",
            "Let me help resolve this issue.",
            "Here's what we can do."
        ],
        "response_tone": "empathetic",
        "confidence": 0.8
    }
}

def get_mock_llm_response(response_type: str, scenario: str) -> Dict[str, Any]:
    """Get a mock LLM response for testing"""
    responses = {
        "content": MOCK_CONTENT_ANALYSIS,
        "user": MOCK_USER_ANALYSIS,
        "conversation": MOCK_CONVERSATION_ANALYSIS
    }
    
    if response_type not in responses:
        raise ValueError(f"Unknown response type: {response_type}")
    
    response_set = responses[response_type]
    if scenario not in response_set:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return response_set[scenario]

def get_mock_llm_error() -> Dict[str, Any]:
    """Get a mock LLM error response"""
    return {
        "error": {
            "type": "server_error",
            "message": "The LLM service is temporarily unavailable",
            "code": 503,
            "should_retry": True,
            "retry_after": 30
        }
    }

def get_mock_llm_timeout() -> Dict[str, Any]:
    """Get a mock LLM timeout response"""
    return {
        "error": {
            "type": "timeout",
            "message": "Request timed out after 30 seconds",
            "code": 408,
            "should_retry": True,
            "retry_after": 5
        }
    }

def get_mock_llm_rate_limit() -> Dict[str, Any]:
    """Get a mock LLM rate limit response"""
    return {
        "error": {
            "type": "rate_limit",
            "message": "Rate limit exceeded",
            "code": 429,
            "should_retry": True,
            "retry_after": 60
        }
    } 