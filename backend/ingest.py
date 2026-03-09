"""
Enhanced Document Ingestion Pipeline
- Semantic-aware chunking with overlap
- Preserves regulatory structure (sections, articles, rules)
- Rich metadata for better retrieval
"""

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import chromadb  # type: ignore[import]
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    import pypdf  # type: ignore[import]
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

import os
import re
import uuid

# ============================
# CONFIG
# ============================
DATA_PATH = "data/"
DB_PATH = "chroma_db/"
MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking parameters - SMALLER for better precision
CHUNK_SIZE = 500           # Reduced from 800 for more precise retrieval
CHUNK_OVERLAP = 125        # 25% overlap
MIN_CHUNK_SIZE = 80        # Minimum viable chunk size

# Legacy aliases for compatibility
CLAUSE_CHUNK_SIZE = 300
CONTEXT_CHUNK_SIZE = 500
CLAUSE_OVERLAP = 75
CONTEXT_OVERLAP = 125

# Policy keywords for key sentence detection
POLICY_KEYWORDS = [
    'shall', 'must', 'required', 'eligible', 'prohibited',
    'minimum', 'maximum', 'deadline', 'fee', 'penalty',
    'mandatory', 'compulsory', 'entitled', 'permitted',
    'duration', 'hours', 'marks', 'internal', 'examination'
]

# Section markers to preserve regulatory structure
SECTION_PATTERNS = [
    r'^(?:Article|Section|Rule|Part|Chapter)\s+\d+',  # Article 1, Section 2, etc.
    r'^\d+\.\s+[A-Z]',      # "1. Eligibility" style
    r'^[A-Z]\.\s+[A-Z]',    # "A. General" style
    r'^\([a-z]\)\s+',       # "(a) subsection" style
    r'^[IVX]+\.\s+',        # Roman numerals
    r'^Step\s+\d+',         # "Step 1", "Step 2" etc.
    r'^Procedure',          # Procedure headers
]

# ============================
# LOAD MODELS
# ============================
embedder = None
if HAS_SENTENCE_TRANSFORMERS:
    embedder = SentenceTransformer(MODEL_NAME)
else:
    print("⚠️ SentenceTransformer not found. Embeddings will fail.")

chroma_client = None
collection = None


def get_collection():
    """Get or create ChromaDB collection"""
    if not HAS_CHROMADB:
        print("⚠️ ChromaDB not found.")
        return None
        
    global chroma_client, collection
    if collection is None:
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection = chroma_client.get_or_create_collection(
            name="manuals",
            metadata={"hnsw:space": "cosine"}  # Ensure cosine similarity
        )
    return collection


def clear_collection():
    """Clear existing documents for rebuild"""
    if not HAS_CHROMADB:
        return None
        
    global chroma_client, collection
    try:
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        chroma_client.delete_collection(name="manuals")
    except Exception:
        pass
        
    collection = chroma_client.get_or_create_collection(
        name="manuals",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# ============================
# PDF TEXT EXTRACTION
# ============================
def clean_text(text):
    """Clean extracted text by removing problematic characters"""
    if not text:
        return ""
    # Remove zero-width spaces and other invisible characters
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    text = text.replace('\ufeff', '').replace('\xa0', ' ')
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def load_pdf_text(file_path):
    """Extract text from PDF with page tracking and proper cleaning"""
    if not HAS_PYPDF:
        print(f"⚠️ pypdf not found. Cannot read {file_path}")
        return []
        
    reader = pypdf.PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            cleaned = clean_text(page_text)
            if cleaned:
                pages.append({
                    "page_num": i + 1,
                    "text": cleaned
                })
    return pages


def is_section_header(line):
    """Check if a line is a section header"""
    line = line.strip()
    for pattern in SECTION_PATTERNS:
        if re.match(pattern, line):
            return True
    return False


def extract_section_title(text):
    """Extract section title from beginning of chunk"""
    lines = text.strip().split('\n')
    for line in lines[:3]:  # Check first 3 lines
        line = line.strip()
        if is_section_header(line):
            return line[:100]  # Truncate long headers
    return None


# ============================
# SEMANTIC-AWARE CHUNKING
# ============================
def chunk_text_with_overlap(text, page_num=None):
    """
    Chunk text with overlap, preserving semantic boundaries.
    
    Strategy:
    1. Split by paragraphs first (double newlines)
    2. Respect section headers as natural boundaries
    3. Apply overlap for context continuity
    4. Never break mid-sentence
    """
    # Normalize whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)       # Normalize spaces
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    current_section = None
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Check if this is a new section
        if is_section_header(para.split('\n')[0]):
            # Save current chunk before starting new section
            if len(current_chunk) >= MIN_CHUNK_SIZE:
                chunks.append({
                    "text": current_chunk.strip(),
                    "section": current_section,
                    "page": page_num
                })
            current_chunk = ""
            current_section = extract_section_title(para)
        
        # Add paragraph to current chunk
        if len(current_chunk) + len(para) <= CHUNK_SIZE:
            current_chunk += para + "\n\n"
        else:
            # Chunk is full, save it with overlap
            if len(current_chunk) >= MIN_CHUNK_SIZE:
                chunks.append({
                    "text": current_chunk.strip(),
                    "section": current_section,
                    "page": page_num
                })
                
                # Create overlap: keep last portion of previous chunk
                overlap_text = get_overlap_text(current_chunk)
                current_chunk = overlap_text + para + "\n\n"
            else:
                current_chunk += para + "\n\n"
    
    # Don't forget the last chunk
    if len(current_chunk) >= MIN_CHUNK_SIZE:
        chunks.append({
            "text": current_chunk.strip(),
            "section": current_section,
            "page": page_num
        })
    
    return chunks


def get_overlap_text(text):
    """Extract overlap portion from end of text"""
    if len(text) <= CHUNK_OVERLAP:
        return text
    
    # Try to break at sentence boundary
    overlap_region = text[-CHUNK_OVERLAP * 2:]  # Look at larger region
    sentences = overlap_region.split('. ')
    
    if len(sentences) > 1:
        # Take last 1-2 complete sentences
        overlap = '. '.join(sentences[-2:])
        if not overlap.endswith('.'):
            overlap += '.'
        return overlap + " "
    
    return text[-CHUNK_OVERLAP:]


# ============================
# INGESTION PIPELINE
# ============================
def ingest() -> int:
    """Main ingestion function"""
    collection = clear_collection()  # Clear old data for fresh rebuild
    
    total_chunks: int = 0
    
    for file in os.listdir(DATA_PATH):
        if not file.lower().endswith(".pdf"):
            continue
        
        print(f"\n📄 Processing: {file}")
        file_path = os.path.join(DATA_PATH, file)
        
        # Extract pages
        pages = load_pdf_text(file_path)
        print(f"   📑 Pages extracted: {len(pages)}")
        
        # Chunk each page with semantic awareness
        all_chunks = []
        for page_data in pages:
            page_chunks = chunk_text_with_overlap(
                page_data["text"],
                page_num=page_data["page_num"]
            )
            all_chunks.extend(page_chunks)
        
        print(f"   🧩 Chunks created: {len(all_chunks)}")
        
        if not all_chunks:
            print(f"   ⚠️ No chunks created from {file}")
            continue
        
        if not embedder:
             print("⚠️ Embedder not initialized.")
             continue
             
        # Extract texts and create embeddings
        texts = [c["text"] for c in all_chunks]
        embeddings = embedder.encode(texts, show_progress_bar=True)
        
        # Create IDs and metadata
        ids = [str(uuid.uuid4()) for _ in all_chunks]
        metadatas = []
        
        for i, chunk in enumerate(all_chunks):
            metadatas.append({
                "source": file,
                "domain": "university",
                "type": "policy_chunk",
                "page": chunk.get("page") or 0,
                "section": chunk.get("section") or "",
                "chunk_index": i,
                "char_count": len(chunk["text"])
            })
        
        # Add to ChromaDB
        if collection is not None:
            collection.add(  # type: ignore[misc]
                documents=texts,
                embeddings=embeddings.tolist(),
                ids=ids,
                metadatas=metadatas
            )
        
        total_chunks = int(total_chunks) + len(all_chunks)  # type: ignore[assignment]
        
        # Show sample chunk
        if texts:
            print(f"\n   📝 Sample chunk (first 200 chars):")
            sample: str = str(texts[0])
            if len(sample) > 200:
                sample = sample.__str__()[:200]  # type: ignore[index]
            print(f"   {sample}...")
    
    print(f"\n✅ Ingestion completed!")
    print(f"   Total documents: {len(os.listdir(DATA_PATH))}")
    print(f"   Total chunks: {total_chunks}")
    
    return int(total_chunks)


def run_ingestion():
    """Entry point for ingestion"""
    return ingest()


if __name__ == "__main__":
    run_ingestion()
