import sys
sys.path.append('d:/ragllm')
from sentence_transformers import SentenceTransformer
import chromadb
from backend.retrieval import HybridRetriever

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_collection('manuals')

retriever = HybridRetriever(embedder=embedder, collection=collection, use_bm25_rerank=True)

query = "What is the minimum attendance required?"
results = retriever.retrieve(query, top_k=10)

print(f"Results for query: '{query}'\n")
with open('attendance_debug.txt', 'w', encoding='utf-8') as f:
    f.write(f"Results for query: '{query}'\n\n")
    for i, res in enumerate(results):
        f.write(f"--- Chunk {i+1} [Score: {res.score:.4f}] ---\n")
        f.write(f"Source: {res.source} | Page: {res.page}\n")
        f.write(f"Section: {res.section}\n")
        f.write(f"Content: {res.text}\n\n")
print("Done writing to attendance_debug.txt")
