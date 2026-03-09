"""
Login Page UI for LlamaRAG Assist Admin Portal
Professional dark-themed enterprise login form using Streamlit + custom CSS
"""

import streamlit as st  # type: ignore[import]
import time
from ui.auth import (  # type: ignore[import]
    verify_credentials,
    login,
    check_lockout,
    is_authenticated,
)

# ============================
# CSS — Dark Professional Login UI
# ============================
LOGIN_CSS = """
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
    --danger: #ef4444;
    --danger-bg: rgba(239, 68, 68, 0.1);
    --danger-border: rgba(239, 68, 68, 0.25);
    --success-bg: rgba(34, 197, 94, 0.1);
    --success-border: rgba(34, 197, 94, 0.25);
    --success-text: #4ade80;
}

/* Hide default Streamlit chrome */
#MainMenu, header, footer { visibility: hidden; }
.block-container {
    padding-top: 10vh !important;
    max-width: 360px !important;
    margin: 0 auto !important;
    background-color: var(--bg-main) !important;
}

/* Global */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background-color: var(--bg-main) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Login Card ── */
.login-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-light);
    border-top: 3px solid var(--accent);
    border-radius: 14px;
    padding: 2.75rem 2.5rem 2.25rem;
    max-width: 360px;
    margin: 0 auto;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3),
                0 0 60px var(--accent-glow);
}

/* ── Title & Subtitle ── */
.login-title {
    text-align: center;
    color: var(--text-primary);
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
    letter-spacing: -0.01em;
}
.login-subtitle {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.875rem;
    font-weight: 400;
    margin-bottom: 2rem;
}

/* ── Form Labels & Inputs ── */
div[data-testid="stTextInput"] {
    max-width: 360px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
div[data-testid="stTextInput"] label {
    color: var(--text-secondary) !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    margin-bottom: 0.375rem !important;
}
.stTextInput input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    padding: 0.5rem 0.75rem !important;
    font-size: 0.8125rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextInput input::placeholder {
    color: var(--text-muted) !important;
    font-weight: 400 !important;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ── Primary Button ── */
.stButton {
    max-width: 360px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
.stButton button {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    padding: 0.5rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.8125rem !important;
    font-family: 'Inter', sans-serif !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.2s ease, box-shadow 0.2s ease !important;
    margin-top: 0.75rem;
}
.stButton button:hover {
    background: var(--accent-hover) !important;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3) !important;
}

/* ── Feedback Messages ── */
.feedback-msg {
    border-radius: 8px;
    padding: 0.875rem 1rem;
    text-align: center;
    font-size: 0.8125rem;
    margin-top: 1rem;
    line-height: 1.5;
}
.error-msg {
    background-color: var(--danger-bg);
    border: 1px solid var(--danger-border);
    color: #fca5a5;
}
.success-msg {
    background-color: var(--success-bg);
    border: 1px solid var(--success-border);
    color: var(--success-text);
}

/* ── Lockout ── */
.lockout-box {
    background: var(--danger-bg);
    border: 1px solid var(--danger-border);
    border-radius: 12px;
    padding: 1.75rem;
    text-align: center;
    color: #fca5a5;
    font-size: 0.875rem;
}
.lockout-box .timer {
    font-size: 2rem;
    font-weight: 700;
    color: var(--danger);
    margin: 0.75rem 0;
    font-family: 'Outfit', sans-serif;
}
.lockout-box .lockout-label {
    font-weight: 600;
    font-size: 0.9375rem;
    color: var(--text-primary);
}

/* ── Forgot Password & Checkbox ── */
.forgot-link {
    text-align: right;
    margin-top: 0.375rem;
}
.forgot-link a {
    color: var(--accent);
    font-size: 0.8125rem;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.15s ease;
}
.forgot-link a:hover {
    color: #60a5fa;
    text-decoration: underline;
}
.stCheckbox label span {
    color: var(--text-muted) !important;
    font-size: 0.8125rem !important;
}
/* Checkbox styling */
.stCheckbox [data-testid="stCheckbox"] {
    color: var(--text-secondary) !important;
}

/* ── CAPTCHA ── */
.captcha-box {
    background: rgba(234, 179, 8, 0.1);
    border: 1px solid rgba(234, 179, 8, 0.25);
    border-radius: 8px;
    padding: 0.875rem;
    text-align: center;
    font-size: 0.8125rem;
    color: #fbbf24;
    margin-top: 0.75rem;
}

/* ── Footer ── */
.login-footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.6875rem;
    margin-top: 1.75rem;
    line-height: 1.5;
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

/* ── Spinner ── */
.stSpinner > div {
    color: var(--text-secondary) !important;
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

/* ── Responsive ── */
@media (max-width: 480px) {
    .login-card {
        padding: 2rem 1.5rem 1.75rem;
        margin: 0 1rem;
    }
    .login-title { font-size: 1.25rem; }
    .block-container { padding-top: 6vh !important; }
}
</style>
"""


# ============================
# HTML Helpers
# ============================
def _render_header():
    """Render title and subtitle."""
    st.markdown(
        '<div class="login-title">Admin Login</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="login-subtitle">Chatbot Access Portal</div>',
        unsafe_allow_html=True,
    )


# ============================
# Main Login Page
# ============================
def render_login_page():
    """
    Render the full login page.
    Returns True if the user is now authenticated, False otherwise.
    """
    # Inject CSS
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)

    # ── Clean sidebar ──
    with st.sidebar:
        st.markdown("### LlamaRAG Assist")
        st.markdown("---")
        st.markdown(
            "<small style='color:var(--text-muted)'>LlamaRAG Assist v1.0<br>"
            "University AI Lab</small>",
            unsafe_allow_html=True,
        )

    # ── Success splash (just logged in) ──
    if st.session_state.get("just_logged_in"):
        st.session_state["just_logged_in"] = False
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="success-msg feedback-msg">'
            f'Welcome, <strong>{st.session_state["display_name"]}</strong>.<br>'
            '<span style="font-size:12px;opacity:0.7">Redirecting to dashboard...</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        time.sleep(1.5)
        st.rerun()
        return True

    # Already authenticated
    if is_authenticated():
        return True

    # ── Lockout check ──
    locked, remaining = check_lockout()

    # ── Login Card ──
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    _render_header()

    if locked:
        mins = remaining // 60
        secs = remaining % 60
        st.markdown(
            f'<div class="lockout-box">'
            f'<div class="lockout-label">Account Locked</div>'
            f'<div class="timer">{mins:02d}:{secs:02d}</div>'
            f'Too many failed attempts. Please wait.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        time.sleep(5)
        st.rerun()
        return False

    # ── Form Fields ──
    username = st.text_input("Username", placeholder="Enter your username", key="login_user")
    password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")

    st.markdown(
        '<div class="forgot-link"><a href="#">Forgot password?</a></div>',
        unsafe_allow_html=True,
    )

    remember = st.checkbox("Remember me", key="login_remember")

    # CAPTCHA placeholder after 3 failed attempts
    failed = st.session_state.get("failed_attempts", 0)
    if failed >= 3:
        st.markdown(
            '<div class="captcha-box">CAPTCHA verification required<br>'
            '<span style="font-size:11px">Placeholder for production integration</span></div>',
            unsafe_allow_html=True,
        )

    # Login Button
    if st.button("Sign In", use_container_width=True):
        with st.spinner("Authenticating..."):
            time.sleep(0.6)
            success, message = verify_credentials(username, password)

        if success:
            login(username)
            st.rerun()
        else:
            st.markdown(
                f'<div class="error-msg feedback-msg">'
                f'{message}</div>',
                unsafe_allow_html=True,
            )

    # ── Footer ──
    st.markdown(
        '<div class="login-footer">'
        'Secured with session-based authentication<br>'
        'University AI Assistant'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    return False
