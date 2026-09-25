from flask import Blueprint, request, jsonify
import time
import json
from urllib.parse import quote
import os

from flask_jwt_extended import jwt_required, get_jwt_identity

from backend.ai.query_rewriter import rewrite_query
from backend.extensions import db
from backend.models import Conversation, Message
from backend.rag.hybrid_search import hybrid_search
from backend.rag.reranker import rerank_documents
from backend.ai.answer_generator import generate_answer
from backend.safety.citation_validator import validate_citations
from backend.ai.confidence import calculate_confidence
from backend.classification.classifier import classify_query


chat_bp = Blueprint("chat", __name__)


def _build_considerations(classification):
    """
    Convert classification routing checks into
    user-friendly "what should be checked" items
    for the frontend.
    """

    if not classification:
        return []

    considerations = []

    jurisdiction = classification.get("jurisdiction")
    ip_type = classification.get("ip_type")
    checks = classification.get("checks") or {}

    # =========================================================
    # JURISDICTION
    # =========================================================

    if jurisdiction and jurisdiction != "unknown":
        considerations.append(
            f"Jurisdiction: Check the applicable IP and regulatory "
            f"requirements for {jurisdiction}."
        )

    # =========================================================
    # PATENTABILITY
    # =========================================================

    if checks.get("patent"):
        considerations.append(
            "Patentability: Check novelty, inventive step "
            "(non-obviousness), and industrial applicability."
        )

    # =========================================================
    # PRIOR ART
    # =========================================================

    if checks.get("prior_art"):
        considerations.append(
            "Prior art: Check whether the invention, formulation, "
            "or process has already been disclosed or is already known."
        )

    # =========================================================
    # TRADITIONAL KNOWLEDGE / TKDL
    # =========================================================

    if checks.get("tkdl"):
        considerations.append(
            "Traditional knowledge: Check whether the subject matter "
            "is already documented in traditional knowledge sources "
            "such as TKDL."
        )

    # =========================================================
    # PRODUCT CLASSIFICATION
    # =========================================================

    if checks.get("product_classification"):
        considerations.append(
            "Product classification: Determine the applicable "
            "classification of the Ayurvedic product or formulation."
        )

    # =========================================================
    # REGULATORY
    # =========================================================

    if checks.get("regulatory"):
        considerations.append(
            "Regulatory requirements: Check the applicable AYUSH "
            "and other regulatory requirements for the product."
        )

    # =========================================================
    # ABS
    # =========================================================

    if checks.get("abs"):
        considerations.append(
            "Access and Benefit-Sharing: Check whether biodiversity "
            "and ABS requirements apply to the biological resource."
        )

    # =========================================================
    # TRADEMARK
    # =========================================================

    if checks.get("trademark"):
        considerations.append(
            "Trademark: Check whether the proposed name, brand, or "
            "logo is available for trademark protection."
        )

    # =========================================================
    # DESIGN
    # =========================================================

    if checks.get("design"):
        considerations.append(
            "Design protection: Check whether the visual appearance "
            "or packaging qualifies for design protection."
        )

    # =========================================================
    # TRADE SECRET
    # =========================================================

    if checks.get("trade_secret"):
        considerations.append(
            "Trade secret: Check whether confidential information, "
            "formulas, or processes should be protected as trade secrets."
        )

    # =========================================================
    # INTERNATIONAL
    # =========================================================

    if checks.get("international"):
        considerations.append(
            "International protection: Check the applicable "
            "international filing or protection framework."
        )

    # =========================================================
    # GENERIC PATENT FALLBACK
    # =========================================================

    if not considerations and ip_type == "patent":
        considerations.extend([
            "Patentability: Check novelty, inventive step "
            "(non-obviousness), and industrial applicability.",
            "Prior art: Check whether the invention has already "
            "been disclosed or is already known."
        ])

    return considerations


def _get_or_create_conversation(
    user_id,
    conversation_id,
    first_message
):
    conversation = None

    if conversation_id:
        conversation = Conversation.query.filter_by(
            id=conversation_id,
            user_id=user_id
        ).first()

    if not conversation:
        title = first_message[:60] + (
            "..." if len(first_message) > 60 else ""
        )

        conversation = Conversation(
            user_id=user_id,
            title=title
        )

        db.session.add(conversation)
        db.session.commit()

    return conversation


def _save_assistant_message(
    conversation_id,
    answer,
    citations=None,
    confidence=None,
    classification=None,
    language=None
):
    conversation = Conversation.query.filter_by(
        id=conversation_id
    ).first()

    if not conversation:
        raise ValueError("Conversation not found")

    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        citations=citations or [],
        confidence=confidence,
        classification=json.dumps(
            classification or {}
        ),
        language=language
    )

    db.session.add(assistant_message)

    conversation.updated_at = db.func.now()

    db.session.commit()

    print(
        "ASSISTANT MESSAGE SAVED:",
        assistant_message.id,
        assistant_message.content[:100]
    )


@chat_bp.route("/chat", methods=["POST"])
@jwt_required()
def chat():

    user_id = get_jwt_identity()

    data = request.get_json(silent=True) or {}

    message = data.get("message", "").strip()
    language = data.get("language", "en")
    conversation_id = data.get("conversation_id")

    if not message:
        return jsonify({
            "success": False,
            "error": "Message is required"
        }), 400

    conversation = _get_or_create_conversation(
        user_id,
        conversation_id,
        message
    )

    # =========================================================
    # SAVE USER MESSAGE
    # =========================================================

    db.session.add(
        Message(
            conversation_id=conversation.id,
            role="user",
            content=message,
            language=language
        )
    )

    conversation.updated_at = db.func.now()

    db.session.commit()

    try:

        start = time.time()

        # =====================================================
        # 1. QUERY REWRITING
        # =====================================================

        rewritten_query = rewrite_query(
            message,
            language
        )

        print(
            "Original Query:",
            message
        )

        print(
            "Rewritten Query:",
            rewritten_query
        )

        # =====================================================
        # 2. CLASSIFICATION
        # =====================================================

        classification, classification_confidence = classify_query(
            rewritten_query
        )

        print(
            "Classification:",
            classification
        )

        print(
            "Classification Confidence:",
            classification_confidence
        )

        # =====================================================
        # 3. HYBRID RETRIEVAL
        # =====================================================

        results = hybrid_search(
            rewritten_query,
            k=8,
            min_score=0.65,
            classification=classification
        )

        print(
            "Hybrid Search:",
            round(time.time() - start, 2),
            "sec"
        )

        print(
            "Retrieved Documents:",
            len(results)
        )

        # =====================================================
        # NO RETRIEVAL RESULTS
        # =====================================================

        if not results:

            no_info_answer = (
                "I could not find sufficient information "
                "in the available sources."
            )

            _save_assistant_message(
                conversation.id,
                no_info_answer,
                confidence=0.0,
                classification=classification,
                language=language
            )

            return jsonify({
                "success": True,
                "conversation_id": conversation.id,
                "query": message,
                "rewritten_query": rewritten_query,
                "answer": no_info_answer,
                "citations": [],
                "confidence": 0.0,
                "confidence_level": "Low",
                "classification": classification,
                "classification_confidence": (
                    classification_confidence
                ),
                "considerations": _build_considerations(
                    classification
                ),
                "language": language
            })

        # =====================================================
        # 4. RERANKING
        # =====================================================

        reranked_results = rerank_documents(
            rewritten_query,
            results,
            top_k=5
        )

        print(
            "Reranker:",
            round(time.time() - start, 2),
            "sec"
        )

        print(
            "Reranked Documents:",
            len(reranked_results)
        )

        # =====================================================
        # CHECK RERANKER STATUS
        # =====================================================

        reranker_enabled = (
            os.getenv(
                "DISABLE_RERANKER",
                "false"
            ).strip().lower()
            != "true"
        )

        print(
            "Reranker Enabled:",
            reranker_enabled
        )

        # =====================================================
        # NO RERANKED RESULTS
        # =====================================================

        if not reranked_results:

            no_info_answer = (
                "I could not find sufficient information "
                "in the available sources."
            )

            _save_assistant_message(
                conversation.id,
                no_info_answer,
                confidence=0.0,
                classification=classification,
                language=language
            )

            return jsonify({
                "success": True,
                "conversation_id": conversation.id,
                "query": message,
                "rewritten_query": rewritten_query,
                "answer": no_info_answer,
                "citations": [],
                "confidence": 0.0,
                "confidence_level": "Low",
                "classification": classification,
                "classification_confidence": (
                    classification_confidence
                ),
                "considerations": _build_considerations(
                    classification
                ),
                "language": language
            })

        # =====================================================
        # 5. GENERATE ANSWER
        # =====================================================

        # Use ORIGINAL user question for answer generation.
        # Rewritten query was used for:
        # classification -> retrieval -> reranking.

        answer, source_citations = generate_answer(
            message,
            reranked_results,
            classification,
            language
        )

        print(
            "LLM:",
            round(time.time() - start, 2),
            "sec"
        )

        # =====================================================
        # 6. VALIDATE CITATIONS
        # =====================================================

        documents = [
            document
            for document, score in reranked_results
        ]

        validation = validate_citations(
            answer,
            documents
        )

        print(
            "Citation Validation:",
            validation
        )

        # =====================================================
        # 7. BUILD CITATIONS
        # =====================================================

        citations = []

        for number in validation["valid_citations"]:

            # Safety check
            if number < 1 or number > len(documents):
                continue

            document = documents[number - 1]

            source = document.metadata.get(
                "source",
                "Unknown source"
            )

            page = document.metadata.get(
                "page",
                "Unknown page"
            )

            citations.append({
                "source": source,
                "page": page,
                "url": (
                    "/api/source?"
                    "file="
                    + quote(str(source))
                    + "&page="
                    + quote(str(page))
                )
            })

        # =====================================================
        # 8. CONFIDENCE
        # =====================================================
        abstained = answer.strip().lower().startswith(
            "i could not find sufficient information"
        )
        confidence = calculate_confidence(
            reranked_results,
            citation_valid=validation["valid"],
            reranker_enabled=reranker_enabled,
            abstained=abstained
        )

        # =====================================================
        # CONFIDENCE LEVEL
        # =====================================================

        if confidence >= 0.75:
            confidence_level = "High"

        elif confidence >= 0.50:
            confidence_level = "Medium"

        else:
            confidence_level = "Low"

        print(
            "Confidence:",
            confidence
        )

        print(
            "Confidence Level:",
            confidence_level
        )

        # =====================================================
        # 9. TOTAL TIME
        # =====================================================

        print(
            "TOTAL:",
            round(time.time() - start, 2),
            "sec"
        )

        # =====================================================
        # 10. SAVE ASSISTANT MESSAGE
        # =====================================================

        _save_assistant_message(
            conversation.id,
            answer,
            citations=citations,
            confidence=confidence,
            classification=classification,
            language=language
        )

        # =====================================================
        # 11. RESPONSE
        # =====================================================

        return jsonify({
            "query": message,
            "rewritten_query": rewritten_query,
            "success": True,
            "conversation_id": conversation.id,
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "classification": classification,
            "classification_confidence": (
                classification_confidence
            ),
            "considerations": _build_considerations(
                classification
            ),
            "language": language
        })

    except Exception as error:

        print(
            "Chat Error:",
            error
        )

        db.session.rollback()

        return jsonify({
            "success": False,
            "conversation_id": conversation.id,
            "error": "Unable to process the question."
        }), 500