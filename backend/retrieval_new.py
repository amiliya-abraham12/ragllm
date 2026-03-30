from dataclasses import dataclass
from typing import List, Optional
import math

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# If config module doesn't exist, you might need to import from config.settings

DB_FAISS_PATH = 'vectorstore'
EMBEDDING_E5_LARGE = 'intfloat/e5-large-v2'


@dataclass
class RetrievalResult:
    """Represents a retrieved document chunk with metadata, matching the original retrieval.py format"""
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


class LangChainFAISSRetriever:
    """
    Retrieval system using LangChain's FAISS implementation.
    Wraps the LangChain similarity search to output the exact same RetrievalResult 
    format as your original HybridRetriever.
    """
    
    def __init__(
        self,
        db_path: str = DB_FAISS_PATH,
        model_name: str = EMBEDDING_E5_LARGE,
        device: str = "cpu"
    ):
        print(f"Loading HuggingFace embeddings: {model_name} on {device}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,   
            model_kwargs={
                "trust_remote_code": True,
                "device": device 
            },
            encode_kwargs={"normalize_embeddings": True},
        )
        
        print(f"Loading FAISS vectorstore from {db_path}...")
        self.vectorstore = FAISS.load_local(
            db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Executes a FAISS similarity search and maps the resulting 
        langchain Documents to your custom RetrievalResult objects.
        """
        # FAISS returns L2 distance; lower is better
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        results = []
        for i, (doc, distance) in enumerate(docs_and_scores):
            # Convert L2 distance to a 0-1 similarity score 
            # (Similar to how the original retrieval.py calculates it)
            similarity = 1 / (1 + distance) if distance >= 0 else 0.0
            
            # Extract metadata safely
            metadata = doc.metadata or {}
            
            # Use 'source_file' or 'source' relying on your chunking strategy
            source = metadata.get("source", metadata.get("source_file", "unknown"))
            page = metadata.get("page", 0)
            section = metadata.get("section", None)
            
            results.append(RetrievalResult(
                text=doc.page_content,
                score=similarity,
                source=source,
                page=page,
                section=section,
                chunk_index=i
            ))
            
        return results

# =======================================================
# Example Usage mimicking your Jupyter Notebook Template
# =======================================================
if __name__ == "__main__":
    retriever = LangChainFAISSRetriever()
    
    user_input = (
        "how to do MOOC courses?"
    )
    
    print("\n--- Searching Vectorstore ---\n")
    results = retriever.retrieve(user_input, top_k=5)
    
    # Format identically to your notebook snippet but utilizing our parsed objects
    context_blocks = []
    for d in results:
        # Utilizing the to_citation() method to print source information
        context_blocks.append(f"[Doc {d.chunk_index + 1}] Citation: {d.to_citation()} | Score: {d.score:.4f}\n{d.text}")
        
    context = "\n\n".join(context_blocks)
    print(context)
