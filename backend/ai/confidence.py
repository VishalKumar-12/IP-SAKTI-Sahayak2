def calculate_confidence(
    results,
    citation_valid=True
):
    """
    Calculate an evidence-strength confidence score.

    This is NOT a probability.
    It indicates how strongly the retrieved evidence
    supports the generated answer.

    The function is designed to work both when the
    CrossEncoder reranker is enabled and when it is
    disabled.
    """

    if not results:
        return 0.0

    # ---------------------------------------------------------
    # 1. Evidence count
    # ---------------------------------------------------------

    evidence_count = min(
        len(results),
        5
    )

    evidence_score = evidence_count / 5.0

    # ---------------------------------------------------------
    # 2. Retrieval score
    # ---------------------------------------------------------
    #
    # Hybrid RRF scores are normally very small
    # (around 0.01), so they must NOT be treated
    # as 0-1 relevance scores.
    #
    # We therefore use rank-based evidence quality.
    # ---------------------------------------------------------

    rank_scores = []

    for index, (_, score) in enumerate(
        results[:5],
        start=1
    ):
        # Higher ranked evidence gets higher score.
        rank_score = 1.0 / index
        rank_scores.append(rank_score)

    if rank_scores:
        rank_quality = sum(rank_scores) / len(rank_scores)
    else:
        rank_quality = 0.0

    # Normalize rank quality approximately to 0-1.
    max_rank_quality = sum(
        1.0 / index
        for index in range(1, 6)
    )

    rank_quality = (
        rank_quality / max_rank_quality
        if max_rank_quality
        else 0.0
    )

    # ---------------------------------------------------------
    # 3. Citation validation
    # ---------------------------------------------------------

    citation_score = (
        1.0
        if citation_valid
        else 0.0
    )

    # ---------------------------------------------------------
    # 4. Final confidence
    # ---------------------------------------------------------

    confidence = (
        (evidence_score * 0.40)
        + (rank_quality * 0.35)
        + (citation_score * 0.25)
    )

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    return round(
        confidence,
        2
    )