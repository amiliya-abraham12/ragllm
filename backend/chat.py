"""
RAG Chat Module with Anti-Hallucination Safeguards
- Strict grounding to retrieved context
- Source citation in responses
- Relevance threshold checking
- Post-generation validation
"""

from typing import List, Tuple, Optional, Any, Union

try:
    from config.settings import MAX_CONTEXT_CHARS  # type: ignore[import]
except ImportError:
    MAX_CONTEXT_CHARS = 1500

try:
    from backend.retrieval import HybridRetriever, RetrievalResult  # type: ignore[import]
except ImportError:
    # Fallback for running as a script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.retrieval import HybridRetriever, RetrievalResult  # type: ignore[import]


# ============================
# ANTI-HALLUCINATION PROMPT
# ============================

# COMPREHENSIVE EXTRACTIVE PROMPT with strict grounding
SYSTEM_PROMPT = """You are a University Regulation Assistant.

Your job is to answer questions ONLY using the exact information present
in the provided official university document excerpts.

-----------------------------------
STRICT ACCURACY RULES
-----------------------------------

1. You MUST answer ONLY if the answer is explicitly stated in the document text.

2. If the document does NOT clearly specify the answer:
   You MUST say:
   "The document does not explicitly specify this information."

3. You are NOT allowed to:
   - Guess
   - Infer
   - Assume
   - Generalize
   - Use external knowledge
   - Fill in missing details

4. If a question contains multiple parts:
   Answer ONLY the parts that are explicitly supported.
   Clearly state if any part is not covered.

-----------------------------------
HOW TO FORM ANSWERS
-----------------------------------

• Extract ALL relevant information from the document excerpts.
• Present your answer in a STRUCTURED, POINT-BY-POINT format using numbered lists or bullet points.
• Cover EVERY relevant detail mentioned in the documents — do NOT skip information.
• If there are multiple rules, conditions, or steps, list each one separately.
• Group related points under clear sub-headings if the answer has multiple aspects.
• Rephrase slightly ONLY for clarity — do NOT add meaning.
• Do NOT introduce new facts.
• Be THOROUGH and COMPLETE — include all relevant details from the documents.

-----------------------------------
USER-FRIENDLY BEHAVIOR
-----------------------------------

• Be polite and respectful.
• Use clear, simple language.
• Do NOT mention "context", "chunks", or "retrieval".
• Do NOT blame the user for missing information.

-----------------------------------
DOCUMENT EXCERPT (SOURCE OF TRUTH)
-----------------------------------
{context}

-----------------------------------
USER QUESTION
-----------------------------------
{query}

-----------------------------------
DETAILED ANSWER
-----------------------------------
"""


# ============================
# FORBIDDEN PHRASES CHECK
# ============================

FORBIDDEN_PHRASES = [
    "typically", "usually", "generally", "normally",
    "i think", "i believe", "probably", "likely",
    "might be", "could be", "may be",
    "in general", "as a rule",
    "based on my knowledge", "from what i know"
]


def check_hallucination_risk(answer: str) -> Tuple[bool, List[str]]:
    """
    Check if answer contains hallucination indicators.
    Returns (is_risky, list_of_found_phrases)
    """
    answer_lower = answer.lower()
    found = [phrase for phrase in FORBIDDEN_PHRASES if phrase in answer_lower]
    return len(found) > 0, found


def validate_grounding(answer: str, context: str) -> float:
    """
    Simple check: What fraction of answer words appear in context?
    Higher score = better grounding.
    """
    # Extract significant words (4+ chars)
    answer_words = set(
        word.lower() for word in answer.split()
        if len(word) >= 4 and word.isalpha()
    )
    
    context_lower = context.lower()
    
    if not answer_words:
        return 1.0  # Empty answer is "grounded"
    
    grounded_words = sum(1 for word in answer_words if word in context_lower)
    return grounded_words / len(answer_words)


def validate_query_coverage(answer: str, query: str) -> float:
    """
    Check if answer addresses the query.
    Returns fraction of query terms found in answer.
    """
    query_terms = set(
        word.lower() for word in query.split()
        if len(word) >= 3 and word.isalpha()
    )
    
    if not query_terms:
        return 1.0
    
    answer_lower = answer.lower()
    covered = sum(1 for t in query_terms if t in answer_lower)
    return covered / len(query_terms)


# ============================
# CONTEXT BUILDING
# ============================

def build_context(results: List[RetrievalResult], max_chars: int = 1500) -> Tuple[str, str]:
    """
    Build context string from retrieval results.
    Returns (context_text, citations)
    """
    if not results:
        return "", ""
    
    context_parts: List[str] = []
    citations: List[str] = []
    acc_chars: int = 0
    
    for i, result in enumerate(results):
        # Add chunk without document labels (cleaner for LLM)
        chunk_text = result.text
        
        if int(acc_chars) + int(len(chunk_text)) > int(max_chars):
            # Truncate this chunk to fit
            remaining = int(max_chars) - int(acc_chars) - 50  # Buffer
            if remaining > 100:
                chunk_text = chunk_text[:remaining] + "..."
                context_parts.append(chunk_text)
                citations.append(result.to_citation())
            break
        
        context_parts.append(chunk_text)
        citations.append(result.to_citation())
        _new_acc: int = int(len(chunk_text)) + 10
        acc_chars = int(acc_chars) + _new_acc  # type: ignore[assignment]  # Account for newlines
    
    context = "\n\n".join(context_parts)
    citation_str = ", ".join(set(citations))  # Deduplicate
    
    return context, citation_str


# ============================
# MAIN RAG FUNCTION
# ============================

def ask(
    query: str,
    llm,
    embedder=None,
    collection=None,
    chat_history: Optional[List[Tuple[str, str]]] = None,
    retriever=None,
    max_history: int = 2,
    max_tokens: int = 200,
    temperature: float = 0.1,
    top_k: int = 4,
    use_reranker: bool = False
) -> str:
    """
    Core RAG logic with anti-hallucination safeguards.
    
    Args:
        query: User question
        llm: GeminiLLM model instance
        embedder: SentenceTransformer instance
        collection: ChromaDB collection
        chat_history: List of (question, answer) tuples
        max_history: Maximum history turns to include
        max_tokens: Max output tokens
        temperature: LLM temperature (lower = more deterministic)
        top_k: Number of chunks to retrieve
        use_reranker: Whether to use cross-encoder re-ranking
    
    Returns:
        Generated answer string
    """
    
    if chat_history is None:
        chat_history = []
        
    # Step 1: Retrieve more documents
    if retriever is None:
        retriever = HybridRetriever(
            embedder=embedder,
            collection=collection,
            min_relevance=0.35,
            keyword_boost=0.30,
            use_reranker=use_reranker,
            use_bm25_rerank=True
        )
    
    # Fetch more candidates - correct answer might be ranked lower
    results = retriever.retrieve(query, top_k=max(top_k + 3, 6))
    
    # Step 2: Check if we found relevant documents
    if not results:
        no_info_response = "This information is not available in the official university documents."
        chat_history.append((query, no_info_response))
        return no_info_response
    
    # Step 3: Build context with citations
    context, citations = build_context(results, max_chars=MAX_CONTEXT_CHARS)
    
    if not context.strip():
        no_info_response = "This information is not available in the official university documents."
        chat_history.append((query, no_info_response))
        return no_info_response
    
    # Step 4: Build prompt
    prompt = SYSTEM_PROMPT.format(
        context=context,
        query=query
    )
    
    # Step 5: Generate response
    try:
        response = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["[QUESTION]", "---"],
        )
        
        # Handle None response
        if response is None or not isinstance(response, dict) or "choices" not in response:
            raise ValueError("Model returned empty response")
        
        if not response["choices"] or len(response["choices"]) == 0:
            raise ValueError("No choices in response")
        
        first_choice = response["choices"][0]
        if first_choice is None or not isinstance(first_choice, dict) or "text" not in first_choice:
            raise ValueError("Invalid choice format in response")
        
        if first_choice["text"] is None:
            raise ValueError("Model returned empty text")
        
        answer = first_choice["text"].strip()
    except Exception as e:
        err_msg: str = str(e)
        # Truncate to 50 chars (avoiding slice syntax for Pyre2 compatibility)
        if len(err_msg) > 50:
            err_msg = err_msg.__str__()[:50]  # type: ignore[index]
        error_response = f"Unable to process your question. Please try rephrasing. (Error: {err_msg})"
        chat_history.append((query, error_response))
        return error_response
    
    # Step 6: Multi-layer post-generation validation
    is_risky, risky_phrases = check_hallucination_risk(answer)
    grounding_score = validate_grounding(answer, context)
    query_coverage = validate_query_coverage(answer, query)
    
    # Layer 1: Forbidden phrases = immediate rejection
    if is_risky:
        answer = "This information is not available in the official university documents."
    # Layer 2: Very poor grounding = rejection (lowered to 0.25)
    elif grounding_score < 0.25:
        answer = "This information is not available in the official university documents."
    # Layer 3: Low confidence = add warning
    elif grounding_score < 0.4:
        answer += "\n\n⚠️ Note: Please verify this information with the original document."
    
    # Step 7: Add source citation if not already present
    if citations and "[Source:" not in answer and "not available" not in answer.lower():
        answer += f"\n\n[Source: {citations}]"
    
    # Step 8: Update chat history
    chat_history.append((query, answer))
    if len(chat_history) > max_history:
        chat_history.pop(0)
    
    return answer


def ask_simple(
    query: str,
    llm,
    embedder=None,
    collection=None,
    retriever=None
) -> str:
    """
    Simplified ask function with sensible defaults.
    """
    return ask(
        query=query,
        llm=llm,
        embedder=embedder,
        collection=collection,
        chat_history=[],
        retriever=retriever,
        max_history=0,
        max_tokens=150,
        temperature=0.1,
        top_k=3,
        use_reranker=False
    )


__all__ = ["ask", "ask_simple", "HybridRetriever"]
