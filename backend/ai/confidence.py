def calculate_confidence(
    results,
    citation_valid=True
):
    """
    Calculate evidence confidence for the generated answer.

    The reranker currently returns relevance scores in the
    0-1 range, so we use those scores directly instead of
    applying arbitrary score buckets.

    This is an evidence-strength indicator, not a calibrated
    probability.
    """

    if not results:
        return 0.0

    # ---------------------------------
    # 1. Reranker relevance
    # ---------------------------------

    scores = []

    for _, score in results:
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        # Reranker is expected to return 0-1.
        score = max(0.0, min(1.0, score))
        scores.append(score)

    if not scores:
        return 0.0

    # Strongest retrieved evidence
    top_score = scores[0]

    # Average quality of retrieved evidence
    average_score = sum(scores) / len(scores)

    # ---------------------------------
    # 2. Evidence coverage
    # ---------------------------------

    # More supporting documents increase confidence,
    # but with diminishing returns.
    evidence_score = min(
        len(scores) / 5,
        1.0
    )

    # ---------------------------------
    # 3. Rank quality
    # ---------------------------------

    # Give more importance to higher-ranked documents.
    rank_weights = [
        1 / (index + 1)
        for index in range(len(scores))
    ]

    rank_score = (
        sum(rank_weights) /
        len(rank_weights)
    )

    # ---------------------------------
    # 4. Citation validation
    # ---------------------------------

    citation_score = (
        1.0 if citation_valid else 0.0
    )

    # ---------------------------------
    # 5. Final confidence
    # ---------------------------------

    confidence = (
        (top_score * 0.40)
        + (average_score * 0.20)
        + (evidence_score * 0.15)
        + (rank_score * 0.10)
        + (citation_score * 0.15)
    )

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    return round(
        confidence,
        2
    )