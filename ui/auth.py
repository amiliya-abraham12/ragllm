"""
Authentication Module for LlamaRAG Assist Admin Portal
Handles password hashing, session management, brute-force lockout, and RBAC
"""

import hashlib
import time
import streamlit as st  # type: ignore[import]

# ============================
# Configuration
# ============================
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_SECONDS = 300  # 5 minutes

# Demo admin account (password hash is SHA-256)
# admin / admin123
ADMIN_USERS = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "display_name": "Administrator",
    },
}


# ============================
# Password Utilities
# ============================
def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


# ============================
# Session Helpers
# ============================
def _init_session():
    """Ensure all auth-related session keys exist."""
    defaults = {
        "authenticated": False,
        "username": None,
        "role": None,
        "display_name": None,
        "login_time": None,
        "failed_attempts": 0,
        "lockout_until": 0.0,
        "just_logged_in": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def is_authenticated() -> bool:
    """Return True if the current session is authenticated."""
    _init_session()
    return st.session_state["authenticated"]


def get_user_info() -> dict:
    """Return current user info dict."""
    _init_session()
    return {
        "username": st.session_state["username"],
        "role": st.session_state["role"],
        "display_name": st.session_state["display_name"],
        "login_time": st.session_state["login_time"],
    }


# ============================
# Lockout Logic
# ============================
def check_lockout() -> tuple[bool, int]:
    """
    Check if the account is currently locked out.
    Returns (is_locked, seconds_remaining).
    """
    _init_session()
    if st.session_state["lockout_until"] > time.time():
        remaining = int(st.session_state["lockout_until"] - time.time())
        return True, remaining
    return False, 0


# ============================
# Core Auth Actions
# ============================
def verify_credentials(username: str, password: str) -> tuple[bool, str]:
    """
    Verify username/password.
    Returns (success, message).
    """
    _init_session()

    # Check lockout first
    locked, remaining = check_lockout()
    if locked:
        return False, f"🔒 Account locked. Try again in {remaining} seconds."

    if not username or not password:
        return False, "Please enter both username and password."

    user = ADMIN_USERS.get(username.lower())
    if user is None or user["password_hash"] != hash_password(password):
        st.session_state["failed_attempts"] += 1
        attempts_left = MAX_FAILED_ATTEMPTS - st.session_state["failed_attempts"]

        if st.session_state["failed_attempts"] >= MAX_FAILED_ATTEMPTS:
            st.session_state["lockout_until"] = time.time() + LOCKOUT_DURATION_SECONDS
            st.session_state["failed_attempts"] = 0
            return False, f"🔒 Too many failed attempts. Account locked for {LOCKOUT_DURATION_SECONDS // 60} minutes."

        return False, f"❌ Invalid username or password. {attempts_left} attempt(s) remaining."

    # Success
    return True, "✅ Login successful!"


def login(username: str):
    """Set session state to authenticated for the given user."""
    _init_session()
    user = ADMIN_USERS[username.lower()]
    st.session_state["authenticated"] = True
    st.session_state["username"] = username.lower()
    st.session_state["role"] = user["role"]
    st.session_state["display_name"] = user["display_name"]
    st.session_state["login_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["failed_attempts"] = 0
    st.session_state["just_logged_in"] = True


def logout():
    """Clear authentication session state."""
    keys = [
        "authenticated", "username", "role", "display_name",
        "login_time", "failed_attempts", "lockout_until", "just_logged_in",
    ]
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
