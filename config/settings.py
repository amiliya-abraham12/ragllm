"""
RAGllm Configuration Settings - SPEED OPTIMIZED
Centralized configuration for all components
"""

import os

# Project root directory (one level up from config/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ============================
# MODEL PATHS
# ============================

# Local LLM (Mistral 7B GGUF — fully offline)
LOCAL_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

# GPU offload layers — MX450 has 2 GB VRAM; keep low to avoid VRAM overflow
N_GPU_LAYERS = 10

# Context window (tokens)
N_CTX = 2048

# Sentence embedding model (unchanged)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Legacy aliases (for backward compatibility)
MODEL_NAME = EMBEDDING_MODEL
GEMINI_MODEL_NAME = ""  # Kept to avoid ImportError in any stale imports

# ============================
# VECTOR DATABASE
# ============================
DB_PATH = os.path.join(_PROJECT_ROOT, "chroma_db")
DATA_PATH = os.path.join(_PROJECT_ROOT, "data")

# ============================
# DOCUMENT PROCESSING
# ============================
CHUNK_SIZE = 500          # Smaller chunks to fit within 2048-token context window
CHUNK_OVERLAP = 200       # 25% overlap for context continuity
MIN_CHUNK_SIZE = 50       # Lower minimum to capture short but important admin sections

# ============================
# RETRIEVAL SETTINGS (ACCURACY + SPEED BALANCED)
# ============================
TOP_K = 3                       # Fewer chunks to avoid exceeding context window
MIN_RELEVANCE_SCORE = 0.50      # Higher threshold to filter noisy/irrelevant chunks
USE_RERANKER = True             # Cross-encoder enabled for higher accuracy
USE_BM25_RERANK = False         # Disabled — cross-encoder supersedes BM25
KEYWORD_BOOST = 0.20            # Keep keyword boost for exact matches
CANDIDATE_MULTIPLIER = 1        # Fetch top_k * 1 candidates (limits cross-encoder workload)

# ============================
# LLM SETTINGS (BALANCED for MX 450)
# ============================
MAX_TOKENS = 512          # Allow detailed point-by-point answers
TEMPERATURE = 0.0         # Greedy = fastest + most deterministic
TOP_P = 1.0               # Disable nucleus sampling for speed
REPEAT_PENALTY = 1.0      # Disable for speed

# ============================
# CHAT SETTINGS (SPEED OPTIMIZED)
# ============================
MAX_HISTORY = 2           # Rolling chat history turns
MAX_CONTEXT_CHARS = 1500  # Reduced to prevent 2048-token context window overflow

# ============================
# ANTI-HALLUCINATION
# ============================
FALLBACK_RESPONSE = "This information is not available in the official university documents."
GROUNDING_THRESHOLD = 0.4  # Minimum grounding score to trust answer