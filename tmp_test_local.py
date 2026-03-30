import sys
sys.path.append('d:/ragllm')
from sentence_transformers import SentenceTransformer
import chromadb
from backend.chat import ask_simple
from llama_cpp import Llama

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_collection('manuals')

llm = Llama(model_path="d:/ragllm/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_ctx=2048, n_gpu_layers=35, n_threads=4, n_batch=128, verbose=False)

q = "What is the minimum attendance required?"
ans = ask_simple(q, llm, embedder, collection)
print("\n=== FINAL ANSWER ===")
print(ans)
print("==================\n")
