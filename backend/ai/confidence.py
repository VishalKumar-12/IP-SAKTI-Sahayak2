import math


def _safe_float(value, default=0.0):
    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def _normalize_crossencoder_scores(scores):
    """
    CrossEncoder scores ko 0-1 range mein convert karta hai.

    Important:
    Ye probability nahi hai.
    Ye sirf relative relevance strength hai.
    """

    values = [_safe_float(score) for score in scores]

    if not values:
        return []

    # CrossEncoder score ko sigmoid se probability-like value mein convert
    normalized = []

    for value in values:
        score = 1 / (1 + math.exp(-value))

        normalized.append(
            max(0.0, min(1.0, score))
        )

    return normalized


def _normalize_rrf_scores(scores):
    """
    RRF scores ko relative 0-1 strength mein convert karta hai.
    """

    values = [_safe_float(score) for score in scores]

    if not values:
        return []

    maximum = max(values)

    if maximum <= 0:
        return [0.0 for _ in values]

    return [
        max(0.0, min(1.0, value / maximum))
        for value in values
    ]


def _get_confidence_level(confidence):
    if confidence >= 0.80:
        return "High"

    if confidence >= 0.50:
        return "Medium"

    return "Low"


def calculate_confidence(
    results,
    citation_valid=True,
    reranker_enabled=True,
    abstained=False
):
    """
    Calculates evidence-based confidence.

    Confidence means:
    How strongly the retrieved evidence supports
    the generated answer.

    It is NOT a probability that the answer is correct.
    """

    # ---------------------------------------------------------
    # 1. MODEL ABSTAINED
    # ---------------------------------------------------------

    if abstained:
        return 0.0

    # ---------------------------------------------------------
    # 2. NO RESULTS
    # ---------------------------------------------------------

    if not results:
        return 0.0

    # ---------------------------------------------------------
    # 3. GET SCORES
    # ---------------------------------------------------------

    raw_scores = []

    for item in results:

        try:
            document, score = item

            raw_scores.append(
                _safe_float(score)
            )

        except (TypeError, ValueError):
            continue

    if not raw_scores:
        return 0.0

    # ---------------------------------------------------------
    # 4. NORMALIZE
    # ---------------------------------------------------------

    if reranker_enabled:

        scores = _normalize_crossencoder_scores(
            raw_scores
        )

    else:

        scores = _normalize_rrf_scores(
            raw_scores
        )

    if not scores:
        return 0.0

    # ---------------------------------------------------------
    # 5. SORT BEST EVIDENCE FIRST
    # ---------------------------------------------------------

    scores = sorted(
        scores,
        reverse=True
    )

    # ---------------------------------------------------------
    # 6. TOP EVIDENCE
    # ---------------------------------------------------------

    top_score = scores[0]

    # ---------------------------------------------------------
    # 7. EVIDENCE COVERAGE
    # ---------------------------------------------------------

    strong = sum(
        1
        for score in scores
        if score >= 0.65
    )

    moderate = sum(
        1
        for score in scores
        if score >= 0.50
    )

    # ---------------------------------------------------------
    # 8. TOP 3 QUALITY
    # ---------------------------------------------------------

    top3 = scores[:3]

    average_top3 = (
        sum(top3) / len(top3)
        if top3
        else 0.0
    )

    # ---------------------------------------------------------
    # 9. EVIDENCE CONSISTENCY
    # ---------------------------------------------------------

    if len(top3) >= 2:

        consistency = (
            min(top3) / max(top3)
            if max(top3) > 0
            else 0.0
        )

    else:

        consistency = 0.0

    # ---------------------------------------------------------
    # 10. COVERAGE
    # ---------------------------------------------------------

    if strong >= 3:
        coverage = 1.0

    elif strong == 2:
        coverage = 0.75

    elif strong == 1:
        coverage = 0.50

    elif moderate >= 2:
        coverage = 0.35

    elif moderate == 1:
        coverage = 0.20

    else:
        coverage = 0.0

    # ---------------------------------------------------------
    # 11. CITATION QUALITY
    # ---------------------------------------------------------

    citation_score = (
        1.0
        if citation_valid
        else 0.25
    )

    # ---------------------------------------------------------
    # 12. FINAL CONFIDENCE
    # ---------------------------------------------------------

    confidence = (
        (top_score * 0.45)
        + (average_top3 * 0.20)
        + (coverage * 0.15)
        + (consistency * 0.10)
        + (citation_score * 0.10)
    )

    # ---------------------------------------------------------
    # 13. IMPORTANT SAFETY LIMIT
    # ---------------------------------------------------------

    # One weak document should never produce High confidence.
    if top_score < 0.55:
        confidence = min(
            confidence,
            0.49
        )

    elif top_score < 0.65:
        confidence = min(
            confidence,
            0.69
        )

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    return round(
        confidence,
        2
    )


def get_confidence_level(confidence):

    confidence = _safe_float(
        confidence
    )

    return _get_confidence_level(
        confidence
    )