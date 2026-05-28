"""Browser-based session persistence utilities for Streamlit.

Integrates Streamlit authentication state with window.sessionStorage using
streamlit-javascript, allowing seamless session persistence across page refreshes.
"""

from __future__ import annotations
import streamlit as st
from streamlit_javascript import st_javascript
from frontend.auth import login_session, get_db_connection

AUTH_COOKIE_NAME = "vton_auth"

def persist_login(session_state, user_data: dict) -> None:
    """
    Saves the user data to Streamlit session state and browser sessionStorage.

    Args:
        session_state: Streamlit session state object.
        user_data: Dictionary of user details (id, username, email).
    """
    # 1. Update Streamlit session state in-memory
    login_session(session_state, user_data)
    
    # 2. Inject JS script to store the session token in the parent window's sessionStorage
    js_code = f"""
    <script>
        window.parent.sessionStorage.setItem("{AUTH_COOKIE_NAME}", "{user_data.get('username')}");
    </script>
    """
    st.components.v1.html(js_code, height=0, width=0)


def check_persistent_login(session_state) -> bool:
    """
    Checks the browser's sessionStorage for a valid login token and authenticates the user if found.

    This should be called on application startup.

    Args:
        session_state: Streamlit session state object.

    Returns:
        True if the user is authenticated (or was restored), False otherwise.
    """
    # If already logged in, return True immediately
    if session_state.get("authenticated"):
        return True
        
    # Query browser sessionStorage for the authentication username
    username = st_javascript(f"sessionStorage.getItem('{AUTH_COOKIE_NAME}')")
    
    # BUG FIX: Allow the script to continue running so the iframe mounts properly
    if username == 0:
        return False
        
    # If a valid username is returned, authenticate them against the database
    if username and isinstance(username, str) and username not in ("null", "undefined", ""):
        try:
            with get_db_connection() as conn:
                user = conn.execute(
                    "SELECT id, username, email FROM users WHERE lower(username) = ?",
                    (username.lower().strip(),)
                ).fetchone()
                
                if user:
                    user_dict = {
                        "id": user["id"],
                        "username": user["username"],
                        "email": user["email"]
                    }
                    # Restore the user's session
                    login_session(session_state, user_dict)
                    return True
        except Exception:
            pass
            
    return False


def clear_persisted_login() -> None:
    """
    Clears the persistent session data from the browser's sessionStorage on logout.
    """
    js_code = f"""
    <script>
        window.parent.sessionStorage.removeItem("{AUTH_COOKIE_NAME}");
    </script>
    """
    st.components.v1.html(js_code, height=0, width=0)
