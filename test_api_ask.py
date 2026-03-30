# import requests
# import json

# url = "http://localhost:8000/api/ask"
# payload = {
#     "question": "What is the minimum attendance required?",
#     "top_k": 3,
#     "max_tokens": 150,
#     "temperature": 0.1
# }

# try:
#     response = requests.post(url, json=payload)
#     if response.status_code == 200:
#         data = response.json()
#         with open('api_ask_result.txt', 'w', encoding='utf-8') as f:
#             f.write(data["answer"])
#         print("Success! Saved to api_ask_result.txt")
#     else:
#         print("Error:", response.status_code, response.text)
# except Exception as e:
#     print("Request failed:", e)


import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

# CONFIG
# Just change this to the directory containing your PDF files
PDF_DIRECTORY = r"data" 
DB_FAISS_PATH = r"vectorstore"
EMBED_MODEL = "intfloat/e5-large-v2"

def load_pdf_documents(pdf_dir: str) -> List[Document]:
    """Loads all PDF files from the given directory and splits them into chunks."""
    print(f"Loading PDFs from: {pdf_dir}")
    
    # 1. Load all PDFs from the directory
    loader = DirectoryLoader(pdf_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    raw_documents = loader.load()
    
    print(f"Loaded {len(raw_documents)} pages from PDF files.")
    
    # 2. Split text into manageable chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Max characters per chunk
        chunk_overlap=200,  # Overlap to maintain context between chunks
        length_function=len
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Split documents into {len(chunks)} chunks.")
    
    return chunks

def create_vector_db():
    print("Initializing document pipeline...")
    documents = load_pdf_documents(PDF_DIRECTORY)
    
    if not documents:
        print("No documents found. Please check your PDF_DIRECTORY path.")
        return

    print(f"Loading Embedding Model: {EMBED_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={
            "trust_remote_code": True,
            "device": "cpu" 
        },
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Creating FAISS index. This might take a moment...")
    db = FAISS.from_documents(documents, embeddings)

    db.save_local(DB_FAISS_PATH)
    print(f"✅ FAISS vectorstore saved successfully at: {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()
