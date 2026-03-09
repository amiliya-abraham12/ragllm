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
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Legacy alias (for backward compatibility)
MODEL_NAME = EMBEDDING_MODEL

# ============================
# VECTOR DATABASE
# ============================
DB_PATH = os.path.join(_PROJECT_ROOT, "chroma_db")
DATA_PATH = os.path.join(_PROJECT_ROOT, "data")

# ============================
# DOCUMENT PROCESSING
# ============================
CHUNK_SIZE = 800          # Larger chunks to keep procedures together
CHUNK_OVERLAP = 200       # 25% overlap for context continuity
MIN_CHUNK_SIZE = 50       # Lower minimum to capture short but important admin sections

# ============================
# RETRIEVAL SETTINGS (ACCURACY + SPEED BALANCED)
# ============================
TOP_K = 6                       # More chunks = better recall for specific queries
MIN_RELEVANCE_SCORE = 0.35      # Lower threshold for broader recall
USE_RERANKER = False            # Cross-encoder disabled (too slow)
USE_BM25_RERANK = True          # NEW: Fast BM25 reranking for accuracy
KEYWORD_BOOST = 0.20            # Keep keyword boost for exact matches
CANDIDATE_MULTIPLIER = 2        # Fetch top_k * 2 candidates

# ============================
# LLM SETTINGS (BALANCED for MX 450)
# ============================
N_CTX = 2048              # Larger context for new prompt template
N_GPU_LAYERS = 35         # Offload ALL layers to GPU (llama.cpp auto-limits if VRAM full)
N_THREADS = 8             # Match physical CPU cores
N_BATCH = 512             # Larger batch = better GPU utilization during prompt eval
MAX_TOKENS = 512          # Allow detailed point-by-point answers
TEMPERATURE = 0.0         # Greedy = fastest + most deterministic
TOP_P = 1.0               # Disable nucleus sampling for speed
REPEAT_PENALTY = 1.0      # Disable for speed

# For CPU-only mode (if GPU issues occur)
N_GPU_LAYERS_CPU = 0

# ============================
# CHAT SETTINGS (SPEED OPTIMIZED)
# ============================
MAX_HISTORY = 2           # Rolling chat history turns
MAX_CONTEXT_CHARS = 2500  # More context for detailed answers

# ============================
# ANTI-HALLUCINATION
# ============================
FALLBACK_RESPONSE = "This information is not available in the official university documents."
GROUNDING_THRESHOLD = 0.4  # Minimum grounding score to trust answer