"""
Check chunk contents and test retrieval
"""
import chromadb
from sentence_transformers import SentenceTransformer

# Check stored chunks
client = chromadb.PersistentClient("chroma_db")
collection = client.get_collection("manuals")
results = collection.get(include=["documents", "metadatas"])

print("=" * 60)
print("STORED CHUNKS")
print("=" * 60)

for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    print(f"\n--- Chunk {i+1} (Page {meta.get('page', '?')}) ---")
    print(doc[:600])
    print("...")
    print(f"[Length: {len(doc)} chars]")

# Test retrieval
print("\n" + "=" * 60)
print("RETRIEVAL TEST")
print("=" * 60)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
query = "What is the procedure for college transfer"
query_emb = embedder.encode(query)

search_results = collection.query(
    query_embeddings=[query_emb.tolist()],
    n_results=4,
    include=["documents", "distances"]
)

print(f"\nQuery: {query}")
print("-" * 40)

for i, (doc, dist) in enumerate(zip(search_results["documents"][0], search_results["distances"][0])):
    print(f"\nResult {i+1} (distance: {dist:.4f}):")
    print(doc[:400])
    print("...")
