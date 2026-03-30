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

questions = [
    "What is the minimum attendance required?",
    "How do I apply for re-evaluation of answer scripts?",
    "What are the eligibility criteria for inter-college transfer?",
    "What is the grading system used?",
    "How are internal marks calculated?",
    "What are the conditions for a leave of absence?",
]

with open('result_ask.txt', 'w', encoding='utf-8') as f:
    for q in questions:
        try:
            ans = ask_simple(q, llm, embedder, collection)
            if "not available" in ans.lower():
                f.write(f"NO_ANS: {q}\n")
            else:
                f.write(f"YES_ANS: {q}\n")
        except Exception as e:
            f.write(f"ERROR: {q} - {e}\n")
