"""
Admin UI - Document Management & Knowledge Base Rebuild
Enhanced with secure login, admin dashboard, and progress feedback
"""

import streamlit as st  # type: ignore[import]
import os
import sys
import shutil
import time

# -------------------------------
# Fix import path
# -------------------------------
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

import chromadb  # type: ignore[import]
from config.settings import DB_PATH, DATA_PATH  # type: ignore[import]
from backend.feedback import load_feedback, get_feedback_summary  # type: ignore[import]

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Admin Portal | University AI Assistant",
    layout="centered"
)

# ═══════════════════════════════════════════
# AUTHENTICATION GATE
# ═══════════════════════════════════════════
from ui.auth import is_authenticated, logout, get_user_info  # type: ignore[import]
from ui.login_page import render_login_page  # type: ignore[import]

if not is_authenticated():
    render_login_page()
    st.stop()

# ═══════════════════════════════════════════
# AUTHENTICATED — Admin Dashboard
# ═══════════════════════════════════════════

# -------------------------------
# Custom CSS — Dark Enterprise Admin UI
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
}

/* ── Global ── */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
#MainMenu, header, footer { visibility: hidden; }

/* ── All text elements ── */
.stMarkdown, .stMarkdown p, .stMarkdown span,
.stMarkdown li, .stMarkdown div {
    color: var(--text-primary) !important;
}
.stMarkdown small {
    color: var(--text-secondary) !important;
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

/* ── Welcome Banner ── */
.welcome-banner {
    background: var(--bg-surface);
    border: 1px solid var(--border-light);
    border-top: 3px solid var(--accent);
    border-radius: 14px;
    padding: 1.75rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.welcome-banner h2 {
    color: var(--text-primary);
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.375rem;
    margin: 0 0 0.25rem;
    letter-spacing: -0.01em;
}
.welcome-banner p {
    color: var(--text-secondary);
    font-size: 0.875rem;
    margin: 0;
}

/* ── Stat Cards ── */
.stat-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-light);
    border-radius: 12px;
    padding: 1.375rem 1rem;
    text-align: center;
    transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}
.stat-card:hover {
    border-color: var(--accent);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.12);
}
.stat-card .number {
    font-size: 1.875rem;
    font-weight: 700;
    font-family: 'Outfit', sans-serif;
    color: var(--accent);
    margin-bottom: 0.25rem;
    line-height: 1.2;
}
.stat-card .label {
    color: var(--text-muted);
    font-size: 0.6875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Outfit', sans-serif;
    font-size: 1.0625rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 2rem 0 1.125rem;
    display: flex;
    align-items: center;
    gap: 0.875rem;
}
.section-header::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border-light);
}

/* ── File List Items ── */
.file-item {
    background: var(--bg-surface);
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 0.625rem 0.875rem;
    margin-bottom: 0.375rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: border-color 0.15s ease;
}
.file-item:hover {
    border-color: var(--border-light);
}

/* ── Buttons ── */
.stButton button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}
.stButton button[kind="primary"] {
    background-color: var(--accent) !important;
    border: none !important;
    color: white !important;
}
.stButton button[kind="primary"]:hover {
    background-color: var(--accent-hover) !important;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3) !important;
}
/* Secondary buttons */
.stButton button:not([kind="primary"]) {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text-primary) !important;
}
.stButton button:not([kind="primary"]):hover {
    background-color: var(--bg-hover) !important;
    border-color: var(--accent) !important;
}

/* ── User Info in Sidebar ── */
.sidebar-user-info {
    font-size: 0.8125rem;
    line-height: 1.6;
}
.sidebar-user-info .user-name {
    font-weight: 600;
    color: var(--text-primary);
}
.sidebar-user-info .user-meta {
    color: var(--text-muted);
    font-size: 0.75rem;
}
.role-badge {
    display: inline-block;
    background: var(--accent-glow);
    color: var(--accent);
    font-size: 0.6875rem;
    font-weight: 600;
    padding: 0.125rem 0.5rem;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border: 1px solid rgba(59, 130, 246, 0.2);
}

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"] p {
    color: var(--text-primary) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background-color: var(--bg-surface) !important;
    border: 1px dashed var(--border-light) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-secondary) !important;
}
[data-testid="stFileUploader"] small {
    color: var(--text-muted) !important;
}

/* ── Progress Bar ── */
.stProgress > div > div {
    background-color: var(--accent) !important;
}
.stProgress {
    background-color: var(--bg-elevated) !important;
}

/* ── Spinner ── */
.stSpinner > div {
    color: var(--text-secondary) !important;
}

/* ── Horizontal Rules ── */
hr {
    border-color: var(--border) !important;
}

/* ── Footer ── */
.admin-footer {
    text-align: center;
    padding: 1rem 0 0;
    color: var(--text-muted);
    font-size: 0.6875rem;
}

/* ── Top Bar ── */
.top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
}
.top-bar-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.125rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* ── Responsive ── */
@media (max-width: 640px) {
    .welcome-banner { padding: 1.25rem 1.25rem; }
    .welcome-banner h2 { font-size: 1.15rem; }
    .stat-card .number { font-size: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)



# ── Sidebar: user info + logout ──
user_info = get_user_info()
with st.sidebar:
    st.markdown("### Admin Portal")
    st.markdown("---")
    st.markdown(
        f'<div class="sidebar-user-info">'
        f'<div class="user-name">{user_info["display_name"]}</div>'
        f'<span class="role-badge">{user_info["role"]}</span>'
        f'<div class="user-meta">Session: {user_info["login_time"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    if st.button("Logout", use_container_width=True):
        logout()
        st.rerun()
    st.markdown("---")
    st.markdown(
        "<small style='color:var(--text-muted)'>LlamaRAG Assist v1.0<br>"
        "University AI Lab</small>",
        unsafe_allow_html=True,
    )


# ── Top Bar with Logout ──
top_col1, top_col2 = st.columns([5, 1])
with top_col1:
    st.markdown('<div class="top-bar-title">Admin Portal</div>', unsafe_allow_html=True)
with top_col2:
    if st.button("Logout", key="main_logout", use_container_width=True):
        logout()
        st.rerun()

# ── Welcome Banner ──
st.markdown(
    f'<div class="welcome-banner">'
    f'<h2>Welcome, {user_info["display_name"]}</h2>'
    f'<p>Administrative portal for knowledge base management and configuration.</p>'
    f'</div>',
    unsafe_allow_html=True,
)

# -------------------------------
# Ensure data folder exists
# -------------------------------
UPLOAD_FOLDER = DATA_PATH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# Knowledge Base Stats
# -------------------------------
st.markdown('<div class="section-header">Knowledge Base Overview</div>', unsafe_allow_html=True)

try:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="manuals")
    chunk_count = collection.count()
    pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith('.pdf')]
    doc_count = len(pdf_files)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stat-card"><div class="number">{doc_count}</div>'
            f'<div class="label">Documents</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card"><div class="number">{chunk_count}</div>'
            f'<div class="label">Total Chunks</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        status_txt = "Active" if chunk_count > 0 else "Empty"
        st.markdown(
            f'<div class="stat-card"><div class="number" style="font-size:1.375rem">{status_txt}</div>'
            f'<div class="label">System Status</div></div>',
            unsafe_allow_html=True,
        )

    if chunk_count == 0:
        st.warning("Knowledge base is empty. Please upload documents and rebuild.")
except Exception as e:
    st.error(f"Could not connect to database: {e}")
    pdf_files = []

st.markdown("---")

# -------------------------------
# Feedback Analytics
# -------------------------------
st.markdown('<div class="section-header"> Feedback Analytics</div>', unsafe_allow_html=True)

try:
    fb_summary = get_feedback_summary()
    fb_total = fb_summary["total"]
    fb_pos = fb_summary["positive"]
    fb_neg = fb_summary["negative"]
    fb_rate = fb_summary["accuracy_rate"]

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.markdown(
            f'<div class="stat-card"><div class="number">{fb_total}</div>'
            f'<div class="label">Total Feedback</div></div>',
            unsafe_allow_html=True,
        )
    with fc2:
        st.markdown(
            f'<div class="stat-card"><div class="number" style="color:#22c55e">{fb_pos}</div>'
            f'<div class="label">👍 Positive</div></div>',
            unsafe_allow_html=True,
        )
    with fc3:
        st.markdown(
            f'<div class="stat-card"><div class="number" style="color:#ef4444">{fb_neg}</div>'
            f'<div class="label">👎 Negative</div></div>',
            unsafe_allow_html=True,
        )
    with fc4:
        st.markdown(
            f'<div class="stat-card"><div class="number">{fb_rate}%</div>'
            f'<div class="label">Accuracy Rate</div></div>',
            unsafe_allow_html=True,
        )

    # Recent feedback details
    if fb_summary["recent"]:
        with st.expander(f"View Recent Feedback ({min(len(fb_summary['recent']), 20)} entries)", expanded=False):
            for entry in fb_summary["recent"]:
                rating_icon = "👍" if entry.get("rating") == "positive" else "👎"
                ts = entry.get("timestamp", "")[:16].replace("T", " ")
                q_preview = entry.get("question", "")[:80]
                st.markdown(
                    f"**{rating_icon}** {q_preview}{'...' if len(entry.get('question', '')) > 80 else ''}  \n"
                    f"<small style='color:var(--text-muted)'>{ts}</small>",
                    unsafe_allow_html=True,
                )
                st.markdown("---")
    elif fb_total == 0:
        st.info("No feedback received yet. Users can rate responses using 👍/👎 buttons in the chat interface.")

except Exception as e:
    st.warning(f"Could not load feedback data: {e}")

st.markdown("---")

# -------------------------------
# Document Upload
# -------------------------------
st.markdown('<div class="section-header">Upload Documents</div>', unsafe_allow_html=True)
st.write("Upload PDF files containing university regulations and policies.")

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Only PDF files are supported"
)

if uploaded_files:
    success_count = 0
    for file in uploaded_files:
        try:
            save_path = os.path.join(UPLOAD_FOLDER, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            success_count += 1  # type: ignore[operator]
        except Exception as e:
            st.error(f"Failed to save {file.name}: {e}")

    if success_count > 0:
        st.success(f"Successfully uploaded {success_count} file(s).")
        st.info("Remember to rebuild the knowledge base to process new documents.")

st.markdown("---")

# -------------------------------
# Current Documents
# -------------------------------
st.markdown('<div class="section-header">Managed Documents</div>', unsafe_allow_html=True)

pdf_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith('.pdf')]

if pdf_files:
    for pdf in pdf_files:
        file_path = os.path.join(UPLOAD_FOLDER, pdf)
        file_size = os.path.getsize(file_path) / 1024

        with st.container():
            st.markdown(f'<div class="file-item">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.markdown(f"<span style='color:var(--text-primary);font-weight:500'>{pdf}</span>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<span style='color:var(--text-muted);font-size:0.8rem'>{file_size:.1f} KB</span>", unsafe_allow_html=True)
            with col3:
                if st.button("Delete", key=f"del_{pdf}", help=f"Delete {pdf}", use_container_width=True):
                    try:
                        os.remove(file_path)
                        st.success(f"Deleted {pdf}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("No documents uploaded yet.")

st.markdown("---")

# -------------------------------
# Rebuild Knowledge Base
# -------------------------------
st.markdown('<div class="section-header">System Operations</div>', unsafe_allow_html=True)
st.write("Process all documents or reset the knowledge base.")

col1, col2 = st.columns(2)

with col1:
    if st.button("Rebuild Knowledge Base", type="primary", use_container_width=True):
        if not pdf_files:
            st.error("No documents found to process.")
        else:
            with st.spinner("Processing..."):
                try:
                    from backend.ingest import run_ingestion  # type: ignore[import]

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("Starting ingestion...")
                    progress_bar.progress(20)

                    total_chunks = run_ingestion()

                    progress_bar.progress(100)
                    status_text.text("Complete!")

                    st.success(f"""
                    Knowledge base rebuilt successfully.
                    - Documents processed: {len(pdf_files)}
                    - Total chunks created: {total_chunks}
                    """)

                except Exception as e:
                    st.error(f"Ingestion failed: {str(e)}")
                    st.exception(e)

with col2:
    if st.button("Clear Data", use_container_width=True):
        try:
            client = chromadb.PersistentClient(path=DB_PATH)
            client.delete_collection(name="manuals")
            st.success("Knowledge base cleared!")
            st.rerun()
        except Exception as e:
            st.warning(f"Could not clear: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    '<div class="admin-footer">After uploading new documents, rebuild the knowledge base '
    'to ensure the assistant has the latest information.</div>',
    unsafe_allow_html=True
)
