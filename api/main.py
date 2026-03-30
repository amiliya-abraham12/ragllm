"""
FastAPI Application Entry Point for GeminiRAG Assist.

Loads the LLM, embedder, and ChromaDB collection once at startup
via the lifespan handler, then serves the REST API.

Usage:
    python -m api.main
    # or
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import HTMLResponse, RedirectResponse  # type: ignore

# Ensure project root is on the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import (  # type: ignore
    GEMINI_MODEL_NAME,
    EMBEDDING_MODEL,
    DB_PATH,
)


# ============================
# LIFESPAN — Load Models Once
# ============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load heavy resources (LLM, embedder, ChromaDB) once at startup
    and release them on shutdown.
    """
    print("=" * 50)
    print("🚀 GeminiRAG Assist API — Starting up...")
    print("=" * 50)

    # --- Load Embedder ---
    print("\n📦 Loading sentence embedder...")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        app.state.embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(f"   ✅ Embedder loaded: {EMBEDDING_MODEL}")
    except Exception as e:
        print(f"   ❌ Embedder failed: {e}")
        app.state.embedder = None

    # --- Load ChromaDB Collection ---
    print("\n📦 Connecting to ChromaDB...")
    try:
        import chromadb  # type: ignore
        client = chromadb.PersistentClient(path=DB_PATH)
        app.state.collection = client.get_or_create_collection(
            name="manuals",
            metadata={"hnsw:space": "cosine"},
        )
        count = app.state.collection.count()
        print(f"   ✅ ChromaDB ready — {count} chunks loaded")
    except Exception as e:
        print(f"   ❌ ChromaDB failed: {e}")
        app.state.collection = None

    # --- Load LLM ---
    print("\n📦 Loading LLM (this may take a minute)...")
    try:
        from backend.gemini_wrapper import GeminiLLM  # type: ignore
        app.state.llm = GeminiLLM(model_name="gemini-2.0-flash")
        print("   ✅ LLM loaded: Gemini API (gemini-2.0-flash)")
    except Exception as e:
        print(f"   ❌ LLM failed: {e}")
        app.state.llm = None

    print("\n" + "=" * 50)
    print("✅ API server ready!  Docs → http://localhost:8000/docs")
    print("=" * 50 + "\n")

    yield  # ← App runs here

    # --- Shutdown ---
    print("\n🛑 Shutting down GeminiRAG Assist API...")
    app.state.llm = None
    app.state.embedder = None
    app.state.collection = None


# ============================
# SWAGGER UI CUSTOM THEME
# ============================

SWAGGER_DARK_CSS = """
<style>
  /* ── Dark Theme for Swagger UI ── */
  body { background: #0f1117 !important; }
  .swagger-ui { background: #0f1117; }
  .swagger-ui .topbar { background: linear-gradient(135deg, #1a1d29 0%, #232840 100%); border-bottom: 1px solid rgba(99,102,241,0.3); }
  .swagger-ui .topbar .download-url-wrapper input { background: #1e2235; color: #e2e8f0; border: 1px solid #3d4268; }
  .swagger-ui .info { margin: 20px 0; }
  .swagger-ui .info .title { color: #e2e8f0 !important; font-family: 'Inter', sans-serif; }
  .swagger-ui .info .description { color: #94a3b8; }
  .swagger-ui .info a { color: #818cf8; }
  .swagger-ui .scheme-container { background: #1a1d29; border: 1px solid #2d3154; box-shadow: none; }
  .swagger-ui .opblock-tag { color: #e2e8f0 !important; border-bottom: 1px solid #2d3154; }
  .swagger-ui .opblock-tag:hover { background: rgba(99,102,241,0.05); }
  .swagger-ui .opblock { background: #1a1d29; border: 1px solid #2d3154; border-radius: 8px; margin-bottom: 8px; }
  .swagger-ui .opblock .opblock-summary { border: none; }
  .swagger-ui .opblock .opblock-summary-method { border-radius: 6px; font-weight: 700; min-width: 70px; text-align: center; }
  .swagger-ui .opblock.opblock-get { border-color: rgba(34,197,94,0.3); background: rgba(34,197,94,0.04); }
  .swagger-ui .opblock.opblock-get .opblock-summary-method { background: #16a34a; }
  .swagger-ui .opblock.opblock-post { border-color: rgba(59,130,246,0.3); background: rgba(59,130,246,0.04); }
  .swagger-ui .opblock.opblock-post .opblock-summary-method { background: #2563eb; }
  .swagger-ui .opblock.opblock-delete { border-color: rgba(239,68,68,0.3); background: rgba(239,68,68,0.04); }
  .swagger-ui .opblock.opblock-delete .opblock-summary-method { background: #dc2626; }
  .swagger-ui .opblock .opblock-summary-path { color: #e2e8f0 !important; }
  .swagger-ui .opblock .opblock-summary-description { color: #94a3b8; }
  .swagger-ui .opblock-body { background: #12141d; }
  .swagger-ui .opblock-description-wrapper, .swagger-ui .opblock-section-header { background: #12141d !important; }
  .swagger-ui .opblock-section-header { border-bottom: 1px solid #2d3154 !important; box-shadow: none !important; }
  .swagger-ui .opblock-section-header h4 { color: #e2e8f0 !important; }
  .swagger-ui table thead tr td, .swagger-ui table thead tr th { color: #94a3b8; border-bottom: 1px solid #2d3154; }
  .swagger-ui .parameter__name { color: #e2e8f0; }
  .swagger-ui .parameter__type { color: #818cf8; }
  .swagger-ui .parameter__in { color: #64748b; }
  .swagger-ui .model-title { color: #e2e8f0 !important; }
  .swagger-ui .model { color: #94a3b8; }
  .swagger-ui .prop-type { color: #818cf8; }
  .swagger-ui .model-box { background: #1a1d29 !important; }
  .swagger-ui section.models { border: 1px solid #2d3154; border-radius: 8px; }
  .swagger-ui section.models.is-open h4 { border-bottom: 1px solid #2d3154; }
  .swagger-ui section.models h4 { color: #e2e8f0 !important; }
  .swagger-ui .btn { color: #e2e8f0; border-color: #3d4268; background: #1e2235; }
  .swagger-ui .btn:hover { background: #252a42; }
  .swagger-ui .btn.execute { background: #6366f1; border-color: #6366f1; }
  .swagger-ui .btn.execute:hover { background: #4f46e5; }
  .swagger-ui .btn.cancel { background: #dc2626; border-color: #dc2626; }
  .swagger-ui select { background: #1e2235; color: #e2e8f0; border: 1px solid #3d4268; }
  .swagger-ui input[type=text], .swagger-ui textarea { background: #1e2235 !important; color: #e2e8f0 !important; border: 1px solid #3d4268 !important; }
  .swagger-ui .response-col_status { color: #e2e8f0; }
  .swagger-ui .response-col_description { color: #94a3b8; }
  .swagger-ui .responses-inner { background: #12141d; }
  .swagger-ui .highlight-code { background: #0f1117 !important; }
  .swagger-ui .microlight { background: #0f1117 !important; color: #e2e8f0 !important; }
  .swagger-ui .copy-to-clipboard { background: #1e2235; }
  .swagger-ui .loading-container .loading { animation: none; }
  .swagger-ui .loading-container .loading::after { color: #818cf8; }
  .swagger-ui .markdown p, .swagger-ui .markdown li { color: #94a3b8; }
  .swagger-ui .renderedMarkdown p { color: #94a3b8; }
  .swagger-ui .model-toggle::after { background: url("data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'%3E%3Cpath fill='%2394a3b8' d='M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z'/%3E%3C/svg%3E"); }
  .swagger-ui .expand-operation svg { fill: #94a3b8 !important; }
  .swagger-ui .arrow { fill: #94a3b8 !important; }
  /* Response codes */
  .swagger-ui .opblock-body pre.microlight { background: #0f1117 !important; border: 1px solid #2d3154; border-radius: 6px; padding: 12px; }
  .swagger-ui .responses-table .response-col_description__inner p { color: #94a3b8; }
  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0f1117; }
  ::-webkit-scrollbar-thumb { background: #3d4268; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #6366f1; }
</style>
"""


# ============================
# CUSTOM LANDING PAGE
# ============================

LANDING_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GeminiRAG Assist API</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: #0f1117;
            color: #e2e8f0;
            min-height: 100vh;
            overflow-x: hidden;
        }
        /* Animated gradient bg */
        .bg-gradient {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: radial-gradient(ellipse at 20% 50%, rgba(99,102,241,0.08) 0%, transparent 50%),
                        radial-gradient(ellipse at 80% 20%, rgba(139,92,246,0.06) 0%, transparent 50%),
                        radial-gradient(ellipse at 50% 80%, rgba(59,130,246,0.05) 0%, transparent 50%);
            z-index: 0;
        }
        .container { position: relative; z-index: 1; max-width: 960px; margin: 0 auto; padding: 2rem 1.5rem; }

        /* Hero */
        .hero { text-align: center; padding: 4rem 0 3rem; }
        .hero-badge {
            display: inline-flex; align-items: center; gap: 6px;
            background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.25);
            border-radius: 100px; padding: 6px 16px; font-size: 0.75rem; color: #818cf8;
            font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 1.5rem;
        }
        .hero-badge .dot { width: 7px; height: 7px; border-radius: 50%; background: #22c55e; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        .hero h1 {
            font-size: 3rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; margin-bottom: 1rem;
            background: linear-gradient(135deg, #e2e8f0 0%, #818cf8 50%, #6366f1 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .hero p { font-size: 1.125rem; color: #94a3b8; max-width: 600px; margin: 0 auto; line-height: 1.7; }

        /* Action buttons */
        .actions { display: flex; justify-content: center; gap: 12px; margin-top: 2rem; flex-wrap: wrap; }
        .btn {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 12px 28px; border-radius: 10px; font-size: 0.9rem; font-weight: 600;
            text-decoration: none; transition: all 0.2s ease;
        }
        .btn-primary { background: #6366f1; color: #fff; border: 1px solid #6366f1; }
        .btn-primary:hover { background: #4f46e5; transform: translateY(-1px); box-shadow: 0 8px 25px rgba(99,102,241,0.3); }
        .btn-secondary { background: rgba(255,255,255,0.04); color: #e2e8f0; border: 1px solid #2d3154; }
        .btn-secondary:hover { background: rgba(255,255,255,0.08); border-color: #3d4268; }

        /* Endpoint cards */
        .section-title { font-size: 1.25rem; font-weight: 700; margin-bottom: 1.5rem; color: #e2e8f0; }
        .endpoints { display: grid; grid-template-columns: 1fr; gap: 10px; margin-bottom: 3rem; }
        .endpoint {
            display: flex; align-items: center; gap: 14px;
            background: rgba(255,255,255,0.02); border: 1px solid #1e2235;
            border-radius: 10px; padding: 16px 20px; transition: all 0.2s ease;
        }
        .endpoint:hover { background: rgba(99,102,241,0.04); border-color: rgba(99,102,241,0.2); }
        .method {
            font-size: 0.7rem; font-weight: 700; padding: 4px 10px; border-radius: 5px;
            text-transform: uppercase; letter-spacing: 0.05em; min-width: 54px; text-align: center;
        }
        .method-get { background: rgba(34,197,94,0.15); color: #22c55e; }
        .method-post { background: rgba(59,130,246,0.15); color: #3b82f6; }
        .method-delete { background: rgba(239,68,68,0.15); color: #ef4444; }
        .path { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.875rem; color: #e2e8f0; font-weight: 500; }
        .desc { color: #64748b; font-size: 0.8rem; margin-left: auto; }

        /* Quick start */
        .quickstart {
            background: rgba(255,255,255,0.02); border: 1px solid #1e2235;
            border-radius: 12px; padding: 24px; margin-bottom: 3rem;
        }
        .code-block {
            background: #0a0c14; border: 1px solid #1e2235; border-radius: 8px;
            padding: 16px 20px; margin-top: 12px; overflow-x: auto;
            font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.8rem;
            line-height: 1.8; color: #94a3b8;
        }
        .code-block .kw { color: #c084fc; }
        .code-block .str { color: #34d399; }
        .code-block .fn { color: #60a5fa; }
        .code-block .cm { color: #4a5568; }

        /* Stats */
        .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 3rem; }
        .stat {
            text-align: center; padding: 20px;
            background: rgba(255,255,255,0.02); border: 1px solid #1e2235; border-radius: 10px;
        }
        .stat-value { font-size: 1.5rem; font-weight: 800; color: #818cf8; }
        .stat-label { font-size: 0.75rem; color: #64748b; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

        /* Footer */
        .footer { text-align: center; padding: 2rem 0; border-top: 1px solid #1e2235; color: #4a5568; font-size: 0.75rem; }

        @media (max-width: 640px) {
            .hero h1 { font-size: 2rem; }
            .stats { grid-template-columns: 1fr; }
            .desc { display: none; }
        }
    </style>
</head>
<body>
    <div class="bg-gradient"></div>
    <div class="container">
        <!-- Hero -->
        <div class="hero">
            <div class="hero-badge"><span class="dot"></span> API Online</div>
            <h1>GeminiRAG Assist API</h1>
            <p>Production-ready REST API for AI-powered university regulation Q&A. Ask questions, get grounded answers with source citations and confidence scoring.</p>
            <div class="actions">
                <a href="/docs" class="btn btn-primary">📖 Interactive Docs</a>
                <a href="/redoc" class="btn btn-secondary">📋 ReDoc</a>
                <a href="/api/health" class="btn btn-secondary">💚 Health Check</a>
            </div>
        </div>

        <!-- Live Stats -->
        <div class="stats" id="stats">
            <div class="stat">
                <div class="stat-value" id="stat-chunks">—</div>
                <div class="stat-label">Knowledge Chunks</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-docs">—</div>
                <div class="stat-label">PDF Documents</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-status">—</div>
                <div class="stat-label">System Status</div>
            </div>
        </div>

        <!-- Endpoints -->
        <div class="section-title">⚡ API Endpoints</div>
        <div class="endpoints">
            <div class="endpoint">
                <span class="method method-get">GET</span>
                <span class="path">/api/health</span>
                <span class="desc">Service health check</span>
            </div>
            <div class="endpoint">
                <span class="method method-post">POST</span>
                <span class="path">/api/ask</span>
                <span class="desc">Ask a question (RAG pipeline)</span>
            </div>
            <div class="endpoint">
                <span class="method method-get">GET</span>
                <span class="path">/api/stats</span>
                <span class="desc">Knowledge base statistics</span>
            </div>
            <div class="endpoint">
                <span class="method method-get">GET</span>
                <span class="path">/api/documents</span>
                <span class="desc">List all documents</span>
            </div>
            <div class="endpoint">
                <span class="method method-post">POST</span>
                <span class="path">/api/documents/upload</span>
                <span class="desc">Upload a PDF document</span>
            </div>
            <div class="endpoint">
                <span class="method method-delete">DELETE</span>
                <span class="path">/api/documents/{name}</span>
                <span class="desc">Delete a document</span>
            </div>
            <div class="endpoint">
                <span class="method method-post">POST</span>
                <span class="path">/api/ingest</span>
                <span class="desc">Rebuild knowledge base</span>
            </div>
        </div>

        <!-- Quick Start -->
        <div class="quickstart">
            <div class="section-title">🚀 Quick Start</div>
            <div class="code-block">
<span class="cm"># Ask a question using Python</span>
<span class="kw">import</span> requests

response = requests.<span class="fn">post</span>(<span class="str">"http://localhost:8000/api/ask"</span>, json={
    <span class="str">"question"</span>: <span class="str">"What is the attendance policy?"</span>,
    <span class="str">"top_k"</span>: 6,
    <span class="str">"max_tokens"</span>: 512
})

data = response.<span class="fn">json</span>()
<span class="fn">print</span>(data[<span class="str">"answer"</span>])
<span class="fn">print</span>(<span class="str">f"Confidence: {data['confidence']}"</span>)
            </div>
        </div>

        <div class="quickstart">
            <div class="section-title">💻 cURL Example</div>
            <div class="code-block">
curl -X POST http://localhost:8000/api/ask \\
  -H <span class="str">"Content-Type: application/json"</span> \\
  -d <span class="str">'{"question": "What is the grading policy?", "top_k": 6}'</span>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            GeminiRAG Assist API v1.0.0 &bull; Powered by Gemini + ChromaDB &bull; Built with FastAPI
        </div>
    </div>

    <script>
        // Fetch live stats
        fetch('/api/health')
            .then(r => r.json())
            .then(d => {
                document.getElementById('stat-status').textContent = d.status === 'healthy' ? '✅ Healthy' : '⚠️ Degraded';
                document.getElementById('stat-status').style.color = d.status === 'healthy' ? '#22c55e' : '#f59e0b';
            })
            .catch(() => {
                document.getElementById('stat-status').textContent = '❌ Offline';
                document.getElementById('stat-status').style.color = '#ef4444';
            });
        fetch('/api/stats')
            .then(r => r.json())
            .then(d => {
                document.getElementById('stat-chunks').textContent = d.total_chunks;
                document.getElementById('stat-docs').textContent = d.total_documents;
            })
            .catch(() => {});
    </script>
</body>
</html>"""


# ============================
# CREATE APP
# ============================

app = FastAPI(
    title="GeminiRAG Assist API",
    description=(
        "## 🎓 AI-Powered University Regulation Assistant\n\n"
        "Production REST API for the GeminiRAG Assist chatbot. "
        "Ask questions about university policies and get accurate, "
        "source-cited answers powered by RAG (Retrieval-Augmented Generation).\n\n"
        "### ✨ Features\n"
        "- 🔍 **RAG-powered Q&A** with source citations\n"
        "- 🛡️ **Anti-hallucination** with confidence scoring\n"
        "- 📄 **Document management** — upload, list, delete PDFs\n"
        "- ⚙️ **Knowledge base** ingestion and statistics\n\n"
        "### 🚀 Quick Start\n"
        "1. Check health → `GET /api/health`\n"
        "2. Ask a question → `POST /api/ask`\n"
        "3. View documents → `GET /api/documents`\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={
        "docExpansion": "list",
        "defaultModelsExpandDepth": 1,
        "tryItOutEnabled": True,
        "persistAuthorization": True,
        "filter": True,
    },
)

# Inject dark theme CSS into Swagger UI
from fastapi.openapi.docs import get_swagger_ui_html  # type: ignore

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url or "/openapi.json",
        title=f"{app.title} — Interactive Docs",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_ui_parameters=app.swagger_ui_parameters,
    ) 

# Override the default /docs to add dark theme
original_docs_url = app.docs_url
app.docs_url = None  # Disable default, we use custom above

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
from api.routes import router  # type: ignore
app.include_router(router)


# ============================
# LANDING PAGE + DARK SWAGGER
# ============================

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def landing_page():
    """Serve the custom API landing page."""
    return HTMLResponse(content=LANDING_PAGE_HTML)


@app.get("/swagger-dark.css", include_in_schema=False)
def swagger_dark_css():
    """Serve dark theme CSS for Swagger UI."""
    from fastapi.responses import Response  # type: ignore
    return Response(content=SWAGGER_DARK_CSS, media_type="text/css")


# ============================
# RUN WITH UVICORN
# ============================

if __name__ == "__main__":
    import uvicorn  # type: ignore
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
