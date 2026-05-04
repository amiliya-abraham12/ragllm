"""
Advanced Retrieval System
- Hybrid search (semantic + keyword boost)
- Cross-encoder re-ranking for accuracy
- Relevance threshold filtering
- Source citation support
"""

import math
try:
    import numpy as np  # type: ignore[import]
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import re

# BM25 for fast reranking
try:
    from rank_bm25 import BM25Okapi  # type: ignore[import]
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


@dataclass
class RetrievalResult:
    """Represents a retrieved document chunk with metadata"""
    text: str
    score: float
    source: str
    page: int
    section: Optional[str]
    chunk_index: int
    
    def to_citation(self) -> str:
        """Format as citation string"""
        if self.section:
            return f"[{self.source}, Page {self.page}, {self.section}]"
        return f"[{self.source}, Page {self.page}]"


class HybridRetriever:
    """
    Multi-stage retrieval system with:
    1. Semantic search via ChromaDB
    2. Keyword boost for exact matches
    3. Cross-encoder re-ranking (optional)
    4. Relevance threshold filtering
    """
    
    def __init__(
        self,
        embedder: Any,
        collection: Any,
        min_relevance: float = 0.45,
        keyword_boost: float = 0.15,
        use_reranker: bool = False,
        use_bm25_rerank: bool = True  # NEW: Fast BM25 reranking
    ):
        self.embedder = embedder
        self.collection = collection
        self.min_relevance = min_relevance
        self.keyword_boost = keyword_boost
        self.use_reranker = use_reranker
        self.use_bm25_rerank = use_bm25_rerank and HAS_BM25
        self.reranker = None
        
        # Load cross-encoder if enabled
        if use_reranker:
            self._load_reranker()
    
    def _load_reranker(self):
        """Load cross-encoder model for re-ranking (forced to CPU to save VRAM)"""
        if not HAS_NUMPY:
            print("⚠️ Numpy not found, disabling re-ranker")
            self.use_reranker = False
            return

        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            import torch  # type: ignore[import]
            # Light cross-encoder suitable for limited VRAM
            # Force CPU — MX450 2GB VRAM is reserved for Mistral GPU layers
            self.reranker = CrossEncoder(
                'cross-encoder/ms-marco-MiniLM-L-6-v2',
                max_length=512,
                device=torch.device("cpu")
            )
            print("✅ Cross-encoder loaded for re-ranking (CPU)")
        except ImportError:
            print("⚠️ sentence_transformers not found, disabling re-ranker")
            self.use_reranker = False
        except Exception as e:
            print(f"⚠️ Cross-encoder not available: {e}")
            self.use_reranker = False
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        expand_query: bool = False  # Disabled for speed
    ) -> List[RetrievalResult]:
        """
        Main retrieval pipeline:
        1. Optionally expand query
        2. Semantic search
        3. Keyword boost
        4. Re-rank if enabled
        5. Filter by relevance threshold
        """
        # Step 1: Query expansion
        search_query = query
        if expand_query:
            search_query = self._expand_query(query)
        
        # Step 2: Semantic search (fetch top_k * 2 for filtering - reduced for speed)
        candidates = self._semantic_search(search_query, top_k=top_k * 2)
        
        if not candidates:
            return []
        
        # Step 3: Keyword boost
        candidates = self._apply_keyword_boost(query, candidates)
        
        # Step 4: Re-ranking (BM25 is fast, cross-encoder is accurate but slow)
        if self.use_bm25_rerank and len(candidates) > 1:
            candidates = self._bm25_rerank(query, candidates)
        elif self.use_reranker and self.reranker and len(candidates) > 1:
            candidates = self._rerank_with_crossencoder(query, candidates)
        
        # Step 5: Sort by score and filter
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Apply relevance threshold
        filtered: List[RetrievalResult] = [c for c in candidates if c.score >= self.min_relevance]
        
        # Return top_k results
        return list(filtered)[:top_k]  # type: ignore[index]
    
    def _semantic_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Search ChromaDB for semantically similar chunks"""
        query_embedding = self.embedder.encode(query)
        
        # Handle query_embedding which might be a numpy array or list
        if HAS_NUMPY and hasattr(query_embedding, 'tolist'):
             query_list = query_embedding.tolist()
        else:
             query_list = query_embedding

        # Ensure query_list is a list of floats
        if isinstance(query_list, float): 
             query_list = [query_list]
        elif not isinstance(query_list, list):
             try:
                query_list = list(query_list)
             except:
                pass

        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=min(top_k, 15),  # Reduced cap for speed
            include=["documents", "metadatas", "distances"]  # Removed embeddings
        )
        
        # Comprehensive null checks for ChromaDB results
        if results is None:
            return []
        
        documents = results.get("documents")
        distances = results.get("distances")
        metadatas = results.get("metadatas")
        
        # Check if documents exist and have content
        if not documents or not documents[0]:
            return []
        
        # Ensure distances and metadatas exist
        if not distances or not distances[0]:
            return []
        if not metadatas or not metadatas[0]:
            return []
        
        candidates = []
        documents = documents[0]
        distances = distances[0]
        metadatas = metadatas[0]
        
        for i, (doc, dist, meta) in enumerate(zip(documents, distances, metadatas)):
            # ChromaDB returns L2 distance, convert to similarity
            # For cosine distance: similarity = 1 - distance
            # For L2 distance: similarity = 1 / (1 + distance)
            similarity = 1 / (1 + dist) if dist >= 0 else 0
            
            candidates.append(RetrievalResult(
                text=doc,
                score=similarity,
                source=meta.get("source", "unknown"),
                page=meta.get("page", 0),
                section=meta.get("section"),
                chunk_index=meta.get("chunk_index", i)
            ))
        
        return candidates
    
    def _apply_keyword_boost(
        self, 
        query: str, 
        candidates: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Boost scores for chunks containing exact query keywords"""
        # Extract important keywords (remove stopwords)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
            'because', 'until', 'while', 'what', 'which', 'who', 'whom',
            'this', 'that', 'these', 'those', 'am', 'you', 'your', 'i', 'me'
        }
        
        query_words = set(
            word.lower() 
            for word in re.findall(r'\b\w+\b', query) 
            if word.lower() not in stopwords and len(word) > 2
        )
        
        if not query_words:
            return candidates
        
        for candidate in candidates:
            doc_lower = candidate.text.lower()
            
            # Count keyword matches
            matches = sum(1 for word in query_words if word in doc_lower)
            match_ratio = matches / len(query_words)
            
            # Apply boost proportional to matches
            candidate.score += match_ratio * self.keyword_boost
        
        return candidates
    
    def _rerank_with_crossencoder(
        self, 
        query: str, 
        candidates: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Re-rank candidates using cross-encoder for higher accuracy"""
        if not self.reranker:
            return candidates
        
        # Prepare pairs for cross-encoder
        pairs = [(query, c.text) for c in candidates]
        
        try:
            reranker = self.reranker
            assert reranker is not None, "Reranker is not initialized"
            # Get cross-encoder scores
            scores = reranker.predict(pairs)  # type: ignore[union-attr]
            
            # Update candidate scores (blend with original)
            for i, candidate in enumerate(candidates):
                # Normalize cross-encoder score to 0-1 range
                val = -scores[i]  # type: ignore[operator]
                if HAS_NUMPY:
                     ce_score = 1 / (1 + np.exp(val))  # Sigmoid
                else:
                     ce_score = 1 / (1 + math.exp(val))
                     
                # Blend: 70% cross-encoder, 30% original
                candidate.score = 0.7 * ce_score + 0.3 * candidate.score
        except Exception as e:
            print(f"⚠️ Re-ranking failed: {e}")
        
        return candidates
    
    def _bm25_rerank(
        self,
        query: str,
        candidates: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """BM25 re-ranking: 10x faster than cross-encoder, good accuracy"""
        if not candidates:
            return candidates
            
        if not HAS_BM25:
            return candidates
        
        # Tokenize documents
        tokenized_docs = [c.text.lower().split() for c in candidates]
        bm25 = BM25Okapi(tokenized_docs)
        
        # Get BM25 scores
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Normalize scores to 0-1 range
        max_score = max(scores) if max(scores) > 0 else 1
        
        # Blend: 60% semantic + 40% BM25
        for i, candidate in enumerate(candidates):
            bm25_normalized = scores[i] / max_score
            candidate.score = 0.6 * candidate.score + 0.4 * bm25_normalized
        
        return candidates
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with related terms.
        Simple rule-based expansion for university domain.
        """
        expansions = {
            "procedure": "procedure step steps process how to",
            "transfer": "transfer college procedure step steps process",
            "eligibility": "eligibility criteria requirements qualifications",
            "admission": "admission requirements enrollment registration",
            "fee": "fee payment tuition charges remit",
            "scholarship": "scholarship financial aid merit",
            "exam": "examination test assessment evaluation",
            "grade": "grade marks score GPA CGPA",
            "attendance": "attendance presence absence leave",
            "hostel": "hostel accommodation residence dormitory",
            "library": "library books resources borrowing",
            "leave": "leave absence permission holiday",
            "certificate": "certificate document degree diploma",
            "step": "step procedure process",
            "how": "how procedure step steps process",
        }
        
        query_lower = query.lower()
        expanded = query
        
        for keyword, expansion in expansions.items():
            if keyword in query_lower:
                expanded = f"{query} {expansion}"
                break
        
        return expanded


def cosine_similarity(a: Any, b: Any) -> float:
    """Compute cosine similarity between two vectors"""
    if HAS_NUMPY:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    # Pure python implementation
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


# Legacy compatibility function
def search_docs(query, embedder, collection, top_k):
    """
    Legacy wrapper for backward compatibility.
    Use HybridRetriever for new code.
    """
    retriever = HybridRetriever(
        embedder=embedder,
        collection=collection,
        min_relevance=0.40,
        use_reranker=False
    )
    
    results = retriever.retrieve(query, top_k=top_k)
    
    return [r.text for r in results]
