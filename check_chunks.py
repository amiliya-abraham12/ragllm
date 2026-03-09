"""Check chunks without loading embedding model"""
import chromadb

client = chromadb.PersistentClient("chroma_db")
collection = client.get_collection("manuals")
results = collection.get(include=["documents", "metadatas"])

print(f"Total chunks: {len(results['documents'])}")

for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
    print(f"\n{'='*60}")
    print(f"CHUNK {i+1} | Page {meta.get('page', '?')} | {len(doc)} chars")
    print("="*60)
    print(doc)
