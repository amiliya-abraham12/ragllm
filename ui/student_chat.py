"""
Student Chat UI - University Regulation Assistant
Optimized for accuracy with anti-hallucination safeguards
"""

import os
import sys
import json
from datetime import datetime

# -------------------------------
# Fix import path
# -------------------------------
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# -------------------------------
# Disable telemetry and force offline mode
# -------------------------------
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# (No CUDA DLL patch needed — llama-cpp-python handles its own CUDA linking)

# -------------------------------
# Imports
# -------------------------------
import streamlit as st  # type: ignore[import]
from backend.llm_wrapper import LocalLlamaLLM  # type: ignore[import]
from sentence_transformers import SentenceTransformer  # type: ignore[import]
import chromadb  # type: ignore[import]

from config.settings import (  # type: ignore[import]
    LOCAL_MODEL_PATH,
    N_GPU_LAYERS,
    N_CTX,
    EMBEDDING_MODEL,
    DB_PATH,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    REPEAT_PENALTY,
    TOP_K,
    MAX_HISTORY,
    USE_RERANKER
)

from backend.chat import ask  # type: ignore[import]
from backend.feedback import log_feedback  # type: ignore[import]

# -------------------------------
# Chat History Persistence
# -------------------------------
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chat_history.json")

def load_saved_history():
    """Load chat history from JSON file."""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_history(conversations):
    """Save chat history to JSON file."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def add_message_to_history(question, answer):
    """Add a new Q&A pair to the current conversation and save."""
    convos = load_saved_history()
    # Find today's conversation or create a new one
    today = datetime.now().strftime("%Y-%m-%d")
    current_convo = None
    for convo in convos:
        if convo.get("date") == today and convo.get("id") == st.session_state.get("current_convo_id"):
            current_convo = convo
            break
    if current_convo is None:
        convo_id = datetime.now().strftime("%Y%m%d%H%M%S")
        current_convo = {
            "id": convo_id,
            "date": today,
            "title": question[:50] + ("..." if len(question) > 50 else ""),
            "messages": []
        }
        convos.insert(0, convo_id)  # placeholder
        convos[0] = current_convo
        st.session_state["current_convo_id"] = convo_id
    current_convo["messages"].append({"question": question, "answer": answer})
    save_history(convos)
    return convos

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="University AI Assistant",
    layout="centered"
)

# -------------------------------
# Custom CSS — Dark Enterprise Chat UI
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700&display=swap');

:root {
    --bg-deep: #0b0f1a;
    --bg-main: #0f1522;
    --bg-surface: #161d2e;
    --bg-elevated: #1c2538;
    --bg-hover: #232e44;
    --accent: #3b82f6;
    --accent-hover: #2563eb;
    --accent-glow: rgba(59, 130, 246, 0.15);
    --border: #1e293b;
    --border-light: #2a3650;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --user-bubble: #3b82f6;
    --bot-bubble: #1c2538;
}

/* ── Global ── */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, header, footer { visibility: hidden; }
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 7rem !important;
    max-width: 820px !important;
    background-color: var(--bg-main) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--bg-deep) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div {
    background-color: var(--bg-deep) !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 1rem;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stMarkdown small {
    color: var(--text-secondary) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
}
/* Sidebar info box */
section[data-testid="stSidebar"] [data-testid="stAlert"] {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text-secondary) !important;
}
/* Sidebar button */
section[data-testid="stSidebar"] .stButton button {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background-color: var(--bg-hover) !important;
    border-color: var(--accent) !important;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 1.25rem 0 1.75rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.75rem;
}
.app-header h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 1.625rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.375rem;
    letter-spacing: -0.02em;
}
.app-header p {
    color: var(--text-muted);
    font-size: 0.875rem;
    margin: 0;
}

/* ── Chat Messages ── */
.stChatMessage {
    padding: 0.5rem 0 !important;
    background: transparent !important;
}
[data-testid="stChatMessage"] {
    background: transparent !important;
}

/* Common bubble styles */
[data-testid="stChatMessage"] > div {
    padding: 0.875rem 1.125rem !important;
    font-size: 0.9375rem !important;
    line-height: 1.7 !important;
    max-width: 82% !important;
}

/* Bot bubble (left) */
[data-testid="stChatMessageAssistant"] {
    flex-direction: row !important;
}
[data-testid="stChatMessageAssistant"] > div {
    background-color: var(--bot-bubble) !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text-primary) !important;
    border-radius: 16px 16px 16px 4px !important;
}

/* Force ALL text inside bot messages to be visible */
[data-testid="stChatMessageAssistant"] p,
[data-testid="stChatMessageAssistant"] span,
[data-testid="stChatMessageAssistant"] li,
[data-testid="stChatMessageAssistant"] div,
[data-testid="stChatMessageAssistant"] td,
[data-testid="stChatMessageAssistant"] th,
[data-testid="stChatMessageAssistant"] h1,
[data-testid="stChatMessageAssistant"] h2,
[data-testid="stChatMessageAssistant"] h3,
[data-testid="stChatMessageAssistant"] h4,
[data-testid="stChatMessageAssistant"] strong,
[data-testid="stChatMessageAssistant"] em,
[data-testid="stChatMessageAssistant"] code {
    color: var(--text-primary) !important;
}

/* User bubble (right) */
[data-testid="stChatMessageUser"] {
    flex-direction: row-reverse !important;
}
[data-testid="stChatMessageUser"] > div {
    background-color: var(--user-bubble) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 16px 16px 4px 16px !important;
}
[data-testid="stChatMessageUser"] p,
[data-testid="stChatMessageUser"] span,
[data-testid="stChatMessageUser"] div {
    color: #ffffff !important;
}

/* Hide avatars */
[data-testid="stChatMessageAvatar"] {
    display: none !important;
}

/* ── Warning / Info / Error boxes inside chat ── */
[data-testid="stAlert"] {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"] p {
    color: var(--text-primary) !important;
}

/* ── Spinner ── */
.stSpinner > div {
    color: var(--text-secondary) !important;
}

/* ── Fixed Bottom Input ── */
.stChatInputContainer {
    position: fixed !important;
    bottom: 1rem !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 92% !important;
    max-width: 780px !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 14px !important;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.3),
                0 4px 16px rgba(0,0,0,0.2) !important;
    z-index: 1000;
}
.stChatInputContainer:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.3),
                0 0 20px var(--accent-glow) !important;
}
/* Input text color */
.stChatInputContainer input,
.stChatInputContainer textarea {
    color: var(--text-primary) !important;
    background: transparent !important;
}
.stChatInputContainer input::placeholder,
.stChatInputContainer textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ── Typing Indicator ── */
.typing-pulse {
    display: flex;
    gap: 5px;
    padding: 0.5rem 0;
}
.typing-pulse span {
    width: 5px;
    height: 5px;
    background: var(--accent);
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out both;
}
.typing-pulse span:nth-child(1) { animation-delay: -0.32s; }
.typing-pulse span:nth-child(2) { animation-delay: -0.16s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1.0); }
}

/* ── Welcome Screen ── */
.welcome-screen {
    text-align: center;
    padding: 3rem 1.5rem 2rem;
    max-width: 600px;
    margin: 0 auto;
}
.welcome-screen .welcome-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
}
.welcome-screen h2 {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 0.5rem;
}
.welcome-screen p {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin: 0 0 2rem;
    line-height: 1.6;
}
.welcome-screen .suggestions-label {
    color: var(--text-muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 0.75rem;
}

/* ── Suggestion Chips ── */
.suggestion-chip {
    display: inline-block;
    background: var(--bg-surface);
    border: 1px solid var(--border-light);
    border-radius: 20px;
    padding: 0.5rem 1rem;
    color: var(--text-secondary);
    font-size: 0.8125rem;
    margin: 0.25rem;
    cursor: pointer;
    transition: all 0.2s ease;
}
.suggestion-chip:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--accent-glow);
    transform: translateY(-1px);
}

/* ── Clear Chat Button (top area) ── */
.clear-chat-bar {
    display: flex;
    justify-content: flex-end;
    padding: 0 0 0.5rem;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 1.5rem 0 0;
    color: var(--text-muted);
    font-size: 0.6875rem;
    line-height: 1.5;
}

/* ── Responsive ── */
@media (max-width: 640px) {
    .block-container { max-width: 100% !important; }
    .app-header h1 { font-size: 1.375rem; }
    [data-testid="stChatMessage"] > div { max-width: 90% !important; }
    .stChatInputContainer { width: 95% !important; }
    .welcome-screen { padding: 2rem 1rem 1.5rem; }
    .welcome-screen h2 { font-size: 1.25rem; }
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Sidebar & Header
# -------------------------------
with st.sidebar:
    st.markdown("### 🎓 University AI Assistant")
    st.markdown("---")

    # ── Primary Actions ──
    if st.button("➕  New Chat", use_container_width=True, key="new_chat_btn"):
        new_id = datetime.now().strftime("%Y%m%d%H%M%S")
        st.session_state["current_convo_id"] = new_id
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.rated_messages = set()
        st.rerun()

    if st.button("🗑️  Clear Current Chat", use_container_width=True, key="clear_current_btn"):
        # Clear only the current conversation
        current_id = st.session_state.get("current_convo_id", "")
        if current_id:
            convos = load_saved_history()
            convos = [c for c in convos if c.get("id") != current_id]
            save_history(convos)
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.rated_messages = set()
        new_id = datetime.now().strftime("%Y%m%d%H%M%S")
        st.session_state["current_convo_id"] = new_id
        st.rerun()

    st.markdown("---")

    # ── Quick Help ──
    with st.expander("💡 How to use", expanded=False):
        st.markdown(
            "<small style='color:var(--text-secondary)'>"
            "• Type your question in the chat box below<br>"
            "• The AI answers using official university documents<br>"
            "• Rate answers with 👍/👎 to help improve accuracy<br>"
            "• Click <b>New Chat</b> to start a fresh conversation<br>"
            "• Click <b>Clear Current Chat</b> to erase this chat"
            "</small>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Chat History List ──
    st.markdown("#### 💬 Chat History")
    saved_convos = load_saved_history()
    if saved_convos:
        for i, convo in enumerate(saved_convos[:15]):  # Show last 15
            title = convo.get("title", "Untitled")
            date = convo.get("date", "")
            msg_count = len(convo.get("messages", []))
            convo_id = convo.get("id", "")
            is_active = convo_id == st.session_state.get("current_convo_id", "")

            hist_col1, hist_col2 = st.columns([5, 1])
            with hist_col1:
                label = f"{'▶ ' if is_active else ''}{title}"
                if st.button(label, key=f"convo_{i}", use_container_width=True):
                    st.session_state["current_convo_id"] = convo_id
                    st.session_state.messages = convo.get("messages", [])
                    st.session_state.chat_history = [
                        (m["question"], m["answer"]) for m in convo.get("messages", [])
                    ]
                    st.session_state.rated_messages = set()
                    st.rerun()
            with hist_col2:
                if st.button("✕", key=f"del_convo_{i}", help=f"Delete: {title}"):
                    convos = load_saved_history()
                    convos = [c for c in convos if c.get("id") != convo_id]
                    save_history(convos)
                    if convo_id == st.session_state.get("current_convo_id", ""):
                        st.session_state.chat_history = []
                        st.session_state.messages = []
                        st.session_state.rated_messages = set()
                        st.session_state.pop("current_convo_id", None)
                    st.rerun()

            st.caption(f"{date} · {msg_count} message{'s' if msg_count != 1 else ''}")

        # Clear All History button
        st.markdown("---")
        if st.button("🧹 Clear All History", use_container_width=True, key="clear_all_btn"):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.session_state.rated_messages = set()
            st.session_state.pop("current_convo_id", None)
            save_history([])
            st.rerun()
    else:
        st.markdown("<small style='color:var(--text-muted)'>No chat history yet. Start a conversation!</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:var(--text-muted)'>LlamaRAG Assist v1.0<br>"
        "Powered by Mistral 7B + ChromaDB</small>",
        unsafe_allow_html=True,
    )

# ── Header ──
st.markdown("""
    <div class="app-header">
        <h1>🎓 University AI Assistant</h1>
        <p>Ask anything about university regulations, policies, and procedures</p>
    </div>
""", unsafe_allow_html=True)

# ── Clear Chat button in main area (only if there are messages) ──
if st.session_state.get("messages"):
    clr_col1, clr_col2 = st.columns([7, 2])
    with clr_col2:
        if st.button("🗑️ Clear Chat", key="clear_main_btn", use_container_width=True):
            current_id = st.session_state.get("current_convo_id", "")
            if current_id:
                convos = load_saved_history()
                convos = [c for c in convos if c.get("id") != current_id]
                save_history(convos)
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.session_state.rated_messages = set()
            new_id = datetime.now().strftime("%Y%m%d%H%M%S")
            st.session_state["current_convo_id"] = new_id
            st.rerun()

# -------------------------------
# Cached Model Loaders
# -------------------------------
@st.cache_resource(show_spinner="🧠 Loading local AI model (may take ~30s)...")
def load_llm():
    """Load local Mistral 7B model via llama-cpp-python (fully offline)"""
    try:
        return LocalLlamaLLM(
            model_path=LOCAL_MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
        )
    except Exception as e:
        st.error(f"Failed to load local model: {e}")
        return None


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    """Load sentence transformer for embeddings (offline mode, CPU to save VRAM for LLM)"""
    device = "cpu"  # Force CPU — MX450 2GB VRAM is reserved for Mistral GPU layers
    try:
        model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True, device=device)
    except Exception:
        model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print(f"Embedding model loaded on: {device.upper()}")
    return model


@st.cache_resource(show_spinner="Connecting to document database...")
def load_collection():
    """Connect to ChromaDB collection"""
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name="manuals",
        metadata={"hnsw:space": "cosine"}
    )


# -------------------------------
# Load Resources
# -------------------------------
with st.spinner("Initializing AI system..."):
    llm = load_llm()
    embedder = load_embedder()
    collection = load_collection()

# Guard: if model failed to load, show clear error with retry option
if llm is None:
    st.error(
        "⚠️ **Local AI model failed to load.** "
        "This may be due to a missing model file, insufficient memory, "
        "or a cached failure from a previous session."
    )
    if st.button("🔄 Retry Model Load", key="retry_model_btn"):
        load_llm.clear()  # Clear the @st.cache_resource cache
        st.rerun()
    st.stop()  # Prevent the rest of the UI from rendering with a broken model

doc_count = collection.count()
if doc_count == 0:
    st.warning("No documents in knowledge base. Please use the Admin Portal to upload documents.")

# -------------------------------
# Session State
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rated_messages" not in st.session_state:
    st.session_state.rated_messages = set()

# Auto-load the most recent conversation on first visit
if "current_convo_id" not in st.session_state:
    saved = load_saved_history()
    if saved:
        latest = saved[0]
        st.session_state["current_convo_id"] = latest.get("id", "")
        st.session_state.messages = latest.get("messages", [])
        st.session_state.chat_history = [
            (m["question"], m["answer"]) for m in latest.get("messages", [])
        ]

# -------------------------------
# Display Chat History or Welcome Screen
# -------------------------------
if not st.session_state.messages:
    # ── Welcome Screen with Example Questions ──
    st.markdown("""
        <div class="welcome-screen">
            <div class="welcome-icon">🎓</div>
            <h2>Welcome! How can I help you?</h2>
            <p>I'm your university AI assistant. I can answer questions about
            academic regulations, policies, examinations, and more — all based
            on official university documents.</p>
            <div class="suggestions-label">Try asking</div>
        </div>
    """, unsafe_allow_html=True)

    # Clickable suggestion buttons
    SUGGESTIONS = [
        "What is the attendance policy?",
        "How do I apply for re-evaluation?",
        "What are the eligibility criteria for inter-college transfer?",
        "What is the grading system?",
        "How to apply for a leave of absence?",
    ]
    # Display in 2 columns
    sg_col1, sg_col2 = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        target_col = sg_col1 if i % 2 == 0 else sg_col2
        with target_col:
            if st.button(f"💬  {suggestion}", key=f"suggest_{i}", use_container_width=True):
                st.session_state["prefill_question"] = suggestion
                st.rerun()

else:
    # ── Display Existing Messages ──
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message("user"):
            st.markdown(msg["question"])
        with st.chat_message("assistant"):
            if "not available" in msg["answer"].lower():
                st.warning(msg["answer"])
            else:
                st.markdown(msg["answer"])
            
            if "top_context" in msg:
                with st.expander("Show Ranked Retrieval Output"):
                    st.markdown(msg["top_context"])
            # Feedback buttons for past messages
            msg_key = f"hist_{idx}"
            if msg_key in st.session_state.rated_messages:
                st.caption("✅ Thanks for your feedback!")
            else:
                fb_col1, fb_col2, fb_spacer = st.columns([1, 1, 8])
                with fb_col1:
                    if st.button("👍", key=f"up_{msg_key}", help="Helpful"):
                        log_feedback(msg["question"], msg["answer"], "positive")
                        st.session_state.rated_messages.add(msg_key)
                        st.rerun()
                with fb_col2:
                    if st.button("👎", key=f"dn_{msg_key}", help="Not helpful"):
                        log_feedback(msg["question"], msg["answer"], "negative")
                        st.session_state.rated_messages.add(msg_key)
                        st.rerun()

# -------------------------------
# Chat Input & Processing
# -------------------------------
# Handle prefilled question from suggestion button
prefill = st.session_state.pop("prefill_question", None)
prompt = st.chat_input("Ask a question about university regulations...")

# Use prefill if user clicked a suggestion
if prefill and not prompt:
    prompt = prefill

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    if doc_count == 0:
        st.error("No documents available. Please upload documents using Admin UI first.")
    else:
        with st.chat_message("assistant"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown('<div class="typing-pulse"><span></span><span></span><span></span></div>', unsafe_allow_html=True)

            try:
                answer = ask(
                    query=prompt,
                    llm=llm,
                    embedder=embedder,
                    collection=collection,
                    chat_history=st.session_state.chat_history,
                    max_history=MAX_HISTORY,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    use_reranker=USE_RERANKER
                )

                typing_placeholder.empty()

                if "not available" in answer.lower():
                    st.warning(answer)
                else:
                    st.markdown(answer)

                st.session_state.messages.append({
                    "question": prompt,
                    "answer": answer
                })
                st.session_state.chat_history.append((prompt, answer))
                add_message_to_history(prompt, answer)

                # Feedback buttons for the new response
                new_idx = len(st.session_state.messages) - 1
                new_key = f"hist_{new_idx}"
                fb_c1, fb_c2, fb_sp = st.columns([1, 1, 8])
                with fb_c1:
                    if st.button("👍", key=f"up_{new_key}", help="Helpful"):
                        log_feedback(prompt, answer, "positive")
                        st.session_state.rated_messages.add(new_key)
                        st.rerun()
                with fb_c2:
                    if st.button("👎", key=f"dn_{new_key}", help="Not helpful"):
                        log_feedback(prompt, answer, "negative")
                        st.session_state.rated_messages.add(new_key)
                        st.rerun()

            except Exception as e:
                typing_placeholder.empty()
                st.error(f"Error processing question: {str(e)}")

# Footer
st.markdown("""
    <div class="app-footer">
        This system provides information based on official university documents.<br>
        For critical decisions, please consult the original documentation.<br>
        Rate responses with 👍/👎 to help us improve.
    </div>
""", unsafe_allow_html=True)