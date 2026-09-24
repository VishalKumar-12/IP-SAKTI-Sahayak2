import os
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

_reranker = None


def get_reranker():
    global _reranker

    if os.getenv("DISABLE_RERANKER", "false").lower() == "true":
        return None

    if _reranker is None:
        print("Loading CrossEncoder reranker...")
        _reranker = CrossEncoder(
            MODEL_NAME,
            max_length=512
        )

    return _reranker


def rerank_documents(query, results, top_k=5):

    if not results:
        return []

    reranker = get_reranker()

    # Render / lightweight mode
    if reranker is None:
        return results[:top_k]

    documents = [
        document
        for document, _ in results
    ]

    pairs = [
        (
            query,
            document.page_content
        )
        for document in documents
    ]

    try:
        scores = reranker.predict(
            pairs,
            show_progress_bar=False
        )
    except Exception as e:
        print(f"RERANKER ERROR: {e}")
        return results[:top_k]

    reranked = []

    for document, score in zip(documents, scores):
        reranked.append(
            (
                document,
                float(score)
            )
        )

    reranked.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return reranked[:top_k]