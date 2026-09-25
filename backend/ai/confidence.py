import math


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_crossencoder_scores(scores):
    """
    Normalize CrossEncoder raw scores to 0-1.

    IMPORTANT:
    CrossEncoder scores are relevance scores, not probabilities.
    Min-max normalization is used only within the retrieved result set.
    """

    values = [
        _safe_float(score)
        for score in scores
    ]

    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    # All scores are identical.
    if maximum == minimum:

        # Identical positive/zero scores still represent
        # some retrieval evidence, but not strong differentiation.
        if maximum > 0:
            return [0.70 for _ in values]

        return [0.0 for _ in values]

    normalized = []

    for value in values:

        score = (
            (value - minimum)
            / (maximum - minimum)
        )

        score = max(
            0.0,
            min(1.0, score)
        )

        normalized.append(score)

    return normalized


def _normalize_rrf_scores(scores):
    """
    Normalize RRF scores relative to the strongest result.

    RRF scores are ranking scores and must NOT be interpreted
    directly as probabilities.
    """

    values = [
        _safe_float(score)
        for score in scores
    ]

    if not values:
        return []

    maximum = max(values)

    if maximum <= 0:
        return [0.0 for _ in values]

    normalized = []

    for value in values:

        score = value / maximum

        score = max(
            0.0,
            min(1.0, score)
        )

        normalized.append(score)

    return normalized


def _get_confidence_level(confidence):
    """
    Convert numerical confidence to a readable level.
    """

    if confidence >= 0.75:
        return "High"

    if confidence >= 0.50:
        return "Medium"

    return "Low"


def calculate_confidence(
    results,
    citation_valid=True,
    reranker_enabled=True
):
    """
    Calculate evidence-based confidence.

    This value represents the strength of retrieved evidence
    and citation support. It is NOT a statistical probability
    that the generated answer is correct.
    """

    # =========================================================
    # NO RESULTS
    # =========================================================

    if not results:
        return 0.0

    # =========================================================
    # EXTRACT SCORES
    # =========================================================

    raw_scores = []

    for item in results:

        try:
            document, score = item

            raw_scores.append(
                _safe_float(score)
            )

        except (TypeError, ValueError):
            raw_scores.append(0.0)

    if not raw_scores:
        return 0.0

    # =========================================================
    # NORMALIZE SCORE
    # =========================================================

    if reranker_enabled:

        normalized_scores = (
            _normalize_crossencoder_scores(
                raw_scores
            )
        )

    else:

        normalized_scores = (
            _normalize_rrf_scores(
                raw_scores
            )
        )

    if not normalized_scores:
        return 0.0

    # =========================================================
    # TOP RESULT
    # =========================================================

    top_score = normalized_scores[0]

    # =========================================================
    # TOP 3 EVIDENCE
    # =========================================================

    top_scores = normalized_scores[:3]

    average_top_score = (
        sum(top_scores)
        / len(top_scores)
    )

    # =========================================================
    # STRONG EVIDENCE
    # =========================================================

    strong_evidence_count = sum(
        1
        for score in normalized_scores
        if score >= 0.55
    )

    evidence_coverage = min(
        strong_evidence_count / 3.0,
        1.0
    )

    # =========================================================
    # RANK QUALITY
    # =========================================================

    rank_weights = [
        1.00,
        0.75,
        0.50
    ]

    weighted_sum = 0.0
    weight_sum = 0.0

    for score, weight in zip(
        normalized_scores[:3],
        rank_weights
    ):

        weighted_sum += (
            score * weight
        )

        weight_sum += weight

    if weight_sum > 0:

        rank_quality = (
            weighted_sum
            / weight_sum
        )

    else:
        rank_quality = 0.0

    # =========================================================
    # CITATION QUALITY
    # =========================================================

    if citation_valid:
        citation_score = 1.0
    else:
        citation_score = 0.30

    # =========================================================
    # FINAL CONFIDENCE
    # =========================================================

    confidence = (
        (top_score * 0.40)
        +
        (average_top_score * 0.25)
        +
        (evidence_coverage * 0.15)
        +
        (rank_quality * 0.10)
        +
        (citation_score * 0.10)
    )

    # =========================================================
    # SAFETY BOUND
    # =========================================================

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    return round(
        confidence,
        2
    )


def get_confidence_level(confidence):
    """
    Return confidence label.
    """

    confidence = _safe_float(
        confidence
    )

    return _get_confidence_level(
        confidence
    )