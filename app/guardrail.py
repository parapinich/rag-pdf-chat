"""
Prompt Guardrail — validates user queries before they reach the RAG engine.

Checks performed:
  1. Empty or whitespace-only queries
  2. Queries exceeding maximum length
  3. Queries containing blocked patterns (prompt injection, SQL injection, etc.)
"""

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_QUERY_LENGTH = 500

BLOCKED_PATTERNS = [
    # Prompt injection attempts
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+the\s+above",
    r"disregard\s+(all\s+)?previous",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+if",
    r"pretend\s+you\s+are",
    r"system\s*prompt",
    r"reveal\s+(your|the)\s+(instructions|prompt|system)",

    # SQL injection patterns
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|UNION)\b\s)",
    r"(--|;)\s*(SELECT|DROP|INSERT)",

    # Command injection
    r"(&&|\|\|)\s*(rm|del|format|shutdown)",
    r"<script\b",
]

# Pre-compile for performance
_compiled_patterns = [
    re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS
]


# ---------------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """Result of a guardrail validation check."""
    is_safe: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Validation Function
# ---------------------------------------------------------------------------

def validate_query(query: str) -> GuardrailResult:
    """
    Validate a user query against safety guardrails.

    Args:
        query: The raw user input string.

    Returns:
        GuardrailResult indicating whether the query is safe to process.
    """
    # Check 1: Empty query
    if not query or not query.strip():
        return GuardrailResult(
            is_safe=False,
            reason="Query is empty. Please enter a question.",
        )

    cleaned = query.strip()

    # Check 2: Length limit
    if len(cleaned) > MAX_QUERY_LENGTH:
        return GuardrailResult(
            is_safe=False,
            reason=(
                f"Query is too long ({len(cleaned)} characters). "
                f"Maximum allowed is {MAX_QUERY_LENGTH} characters."
            ),
        )

    # Check 3: Blocked patterns
    for pattern in _compiled_patterns:
        if pattern.search(cleaned):
            return GuardrailResult(
                is_safe=False,
                reason=(
                    "Query contains a blocked pattern and was rejected "
                    "for safety reasons. Please rephrase your question."
                ),
            )

    # All checks passed
    return GuardrailResult(is_safe=True)
