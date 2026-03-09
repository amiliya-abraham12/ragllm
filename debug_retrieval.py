"""Debug script to check what chunks are being retrieved"""
import sys
sys.path.insert(0, 'd:/ragllm')
from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path='chroma_db/')
collection = client.get_collection('manuals')

query = 'duration of internal written examination 2024'
embedding = embedder.encode(query)

results = collection.query(
    query_embeddings=[embedding.tolist()],
    n_results=5,
    include=['documents', 'metadatas', 'distances']
)

with open('debug_output.txt', 'w', encoding='utf-8') as f:
    f.write('=== TOP 5 CHUNKS FOR QUERY ===\n')
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        dist = results['distances'][0][i]
        meta = results['metadatas'][0][i]
        f.write(f"\n--- Chunk {i+1} (distance: {dist:.3f}) ---\n")
        f.write(f"Source: {meta.get('source')}, Page: {meta.get('page')}\n")
        f.write(f"Text: {doc}\n")
print("Output saved to debug_output.txt")
