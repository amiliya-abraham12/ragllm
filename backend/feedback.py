"""
Feedback Logging Module for LlamaRAG Assist
- Logs user feedback (thumbs up/down) for each Q&A interaction
- Persists feedback to JSON file for analysis
- Provides summary statistics for admin dashboard
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional


# Feedback log file path
FEEDBACK_LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "feedback_log.json"
)


def log_feedback(
    question: str,
    answer: str,
    rating: str,
    comment: Optional[str] = None
) -> Dict:
    """
    Log a feedback entry to the feedback log file.
    
    Args:
        question: The user's original question
        answer: The system's response
        rating: "positive" (👍) or "negative" (👎)
        comment: Optional user comment
    
    Returns:
        The logged feedback entry
    """
    entry = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "rating": rating,
        "comment": comment or ""
    }
    
    # Load existing feedback
    feedback_list = load_feedback()
    feedback_list.append(entry)
    
    # Save updated feedback
    try:
        with open(FEEDBACK_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(feedback_list, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Feedback] Error saving feedback: {e}")
    
    return entry


def load_feedback() -> List[Dict]:
    """
    Load all feedback entries from the log file.
    
    Returns:
        List of feedback entries
    """
    try:
        if os.path.exists(FEEDBACK_LOG_FILE):
            with open(FEEDBACK_LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception as e:
        print(f"[Feedback] Error loading feedback: {e}")
    return []


def get_feedback_summary() -> Dict:
    """
    Get aggregate feedback statistics.
    
    Returns:
        Dictionary with:
        - total: Total feedback entries
        - positive: Number of positive ratings
        - negative: Number of negative ratings
        - accuracy_rate: Percentage of positive ratings (0-100)
        - recent: Last 20 feedback entries (newest first)
    """
    feedback = load_feedback()
    total = len(feedback)
    positive = sum(1 for f in feedback if f.get("rating") == "positive")
    negative = sum(1 for f in feedback if f.get("rating") == "negative")
    accuracy_rate = round((positive / total) * 100, 1) if total > 0 else 0.0
    
    # Most recent entries first
    recent = sorted(
        feedback,
        key=lambda x: x.get("timestamp", ""),
        reverse=True
    )[:20]
    
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "accuracy_rate": accuracy_rate,
        "recent": recent
    }


__all__ = ["log_feedback", "load_feedback", "get_feedback_summary"]
