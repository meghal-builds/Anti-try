"""Virtual Try-On System - Streamlit Frontend
AI-Based Virtual Try-On and Fit Recommendation System
"""

import io
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from frontend.auth import (
    authenticate_user,
    create_user,
    init_auth_db,
    initialize_auth_session,
    is_session_valid,
    login_session,
    logout_session,
    request_password_reset,
    reset_password_with_token,
)
from frontend.auth_browser import (
    persist_login,
    check_persistent_login,
    clear_persisted_login,
)
from ml_ai.core.garment_manager import list_available_garments, load_garment_image, load_garment_metadata
from ml_ai.core.image_utils import load_image
from ml_ai.core.measurement_inference import infer_measurements, validate_measurements
from ml_ai.core.model_layer import load_models
from ml_ai.core.pose_detection import detect_pose
from ml_ai.core.segmentation import segment_body
from ml_ai.core.size_recommendation import explain_recommendation, recommend_size
from ml_ai.core.validation import validate_image


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Virtual Try-On",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

# Animated background orbs — must use st.components.v1.html so script tags execute
st.components.v1.html(
    """
    <canvas id="bg-canvas" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -1; pointer-events: none;"></canvas>
    <script>
    (function() {
        const canvas = document.getElementById('bg-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        const colors = [
            {r: 139, g: 92, b: 246, a: 0.2},
            {r: 56, g: 189, b: 248, a: 0.2},
            {r: 236, g: 72, b: 153, a: 0.15},
        ];

        class Orb {
            constructor() {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.radius = Math.random() * 200 + 200;
                this.color = colors[Math.floor(Math.random() * colors.length)];
                this.vx = (Math.random() - 0.5) * 0.8;
                this.vy = (Math.random() - 0.5) * 0.8;
            }
            update() {
                this.x += this.vx;
                this.y += this.vy;
                if (this.x < -this.radius) this.x = canvas.width + this.radius;
                if (this.x > canvas.width + this.radius) this.x = -this.radius;
                if (this.y < -this.radius) this.y = canvas.height + this.radius;
                if (this.y > canvas.height + this.radius) this.y = -this.radius;
            }
            draw() {
                const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius);
                gradient.addColorStop(0, `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, ${this.color.a})`);
                gradient.addColorStop(0.5, `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, ${this.color.a * 0.5})`);
                gradient.addColorStop(1, `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, 0)`);
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        const orbs = Array.from({length: 6}, () => new Orb());
        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            orbs.forEach(orb => { orb.update(); orb.draw(); });
            requestAnimationFrame(animate);
        }
        animate();
    })();
    </script>
    """,
    height=0,
)

st.markdown(
    """
    <style>
    /* Global Styles */
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        color: #f1f5f9 !important;
    }

    /* Background panels */
    [data-testid="stAppViewContainer"] {
        background-color: #0b0f19 !important;
    }
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    .main .block-container {
        background-color: transparent !important;
    }

    /* Glassmorphism sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(24px) saturate(150%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(150%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Premium Title/Subtitle */
    .title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        font-family: 'Outfit', sans-serif;
    }

    .subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }

    /* Input field styling */
    .stTextInput input, .stNumberInput input, .stSelectbox div[role="button"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }
    .stTextInput label, .stNumberInput label, .stSelectbox label,
    .stSlider label, .stCheckbox label, .stRadio label, .stFileUploader label {
        color: #cbd5e1 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(15, 23, 42, 0.3) !important;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #f1f5f9 !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(15, 23, 42, 0.4) !important;
        color: #cbd5e1 !important;
        border-radius: 8px !important;
    }
    .streamlit-expanderContent {
        background-color: rgba(15, 23, 42, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(15, 23, 42, 0.3) !important;
        border: 1px dashed rgba(139, 92, 246, 0.3) !important;
        border-radius: 12px !important;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    /* Dataframe */
    .stDataFrame {
        background-color: rgba(15, 23, 42, 0.4) !important;
        border-radius: 8px !important;
    }

    /* Form containers */
    [data-testid="stForm"] {
        background-color: rgba(15, 23, 42, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
    }

    /* Slider track */
    .stSlider [data-baseweb="slider"] {
        background-color: rgba(139, 92, 246, 0.2) !important;
    }

    /* Measurement and success cards as premium glass blocks */
    .measurement-box {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }

    .success-box {
        background: rgba(16, 185, 129, 0.1) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #34d399 !important;
    }

    .error-box {
        background: rgba(239, 68, 68, 0.1) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(239, 68, 68, 0.2) !important;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #f87171 !important;
    }

    /* Micro-animations on buttons */
    .stButton > button {
        background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2.2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.2) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5) !important;
    }

    @keyframes click-pulse {
        0% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(139, 92, 246, 0); }
        100% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0); }
    }

    .stButton > button:active {
        transform: translateY(2px) scale(0.95) !important;
        box-shadow: 0 2px 10px rgba(139, 92, 246, 0.4) !important;
        animation: click-pulse 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        filter: brightness(1.2);
    }

    /* Protect Material Icons */
    .material-icons, .material-symbols-rounded, [data-testid="stIconMaterial"], [class*="stIcon"] {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# AUTH PAGE
# ============================================================================

def render_auth_page() -> None:
    st.markdown('<p class="title">Secure Access</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Login or create an account to use the Virtual Try-On system</p>',
        unsafe_allow_html=True,
    )
    login_tab, register_tab, forgot_tab = st.tabs(["Login", "Register", "Forgot Password"])

    with login_tab:
        with st.form("login_form", clear_on_submit=False):
            login_id = st.text_input("Email or Username")
            password = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Login")
        if login_submit:
            success, message, user = authenticate_user(login_id, password)
            if success and user is not None:
                persist_login(st.session_state, user)
                st.success("Login successful. Initializing...")
                time.sleep(1)  # Allow JS iframe to mount and save sessionStorage
                st.rerun()
            else:
                st.error(message)

    with register_tab:
        with st.form("register_form", clear_on_submit=True):
            new_username = st.text_input("Username")
            new_email    = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_pw   = st.text_input("Confirm Password", type="password")
            register_submit = st.form_submit_button("Create Account")
        if register_submit:
            if new_password != confirm_pw:
                st.error("Passwords do not match.")
            else:
                success, message = create_user(new_username, new_email, new_password)
                st.success(message) if success else st.error(message)

    with forgot_tab:
        st.caption("Development mode: reset token shown here instead of email.")
        with st.form("forgot_password_form", clear_on_submit=False):
            forgot_login_id = st.text_input("Email or Username", key="forgot_login_id")
            forgot_submit   = st.form_submit_button("Generate Reset Token")
        if forgot_submit:
            success, message, reset_token = request_password_reset(forgot_login_id)
            if success:
                st.info(message)
                if reset_token:
                    st.code(reset_token)
                    st.warning("This token expires in 15 minutes.")
            else:
                st.error(message)

        with st.form("reset_password_form", clear_on_submit=True):
            reset_token_input = st.text_input("Reset Token")
            new_password      = st.text_input("New Password", type="password")
            confirm_pw        = st.text_input("Confirm New Password", type="password")
            reset_submit      = st.form_submit_button("Reset Password")
        if reset_submit:
            if new_password != confirm_pw:
                st.error("Passwords do not match.")
            else:
                success, message = reset_password_with_token(reset_token_input, new_password)
                st.success(message) if success else st.error(message)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_ai_models():
    try:
        return load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


@st.cache_resource(show_spinner=False)
def get_tryon_engine():
    """Load TryOnEngine once per session."""
    from ml_ai.core.tryon_engine import TryOnEngine
    return TryOnEngine()


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert BGR/BGRA numpy array to PIL Image."""
    if len(image.shape) == 3 and image.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def image_to_bytes(image: np.ndarray) -> bytes:
    """Convert BGR numpy array to PNG bytes."""
    pil_img = bgr_to_pil(image)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def process_user_image(image_path, user_height_cm: float = 0.0):
    """Process user image: detect pose, segment body, infer measurements."""
    try:
        seg_model, pose_model = load_ai_models()
        if seg_model is None or pose_model is None:
            st.error("Models not loaded")
            return None

        image = load_image(image_path)
        image_height = image.shape[0]

        with st.spinner("Segmenting body..."):
            seg_result = segment_body(image, seg_model)

        with st.spinner("Detecting pose..."):
            try:
                pose_result = detect_pose(image, pose_model)
            except RuntimeError as e:
                st.error(f"Pose detection failed: {e}")
                return None

        with st.spinner("Inferring measurements..."):
            measurements = infer_measurements(
                pose_result, seg_result,
                image_height=image_height,
                user_height_cm=user_height_cm
            )

        is_valid, errors = validate_measurements(measurements)
        if not is_valid:
            st.error(f"Measurement validation failed: {errors}")
            return None

        return {
            "image":        image,
            "measurements": measurements,
            "pose":         pose_result,
            "segmentation": seg_result,
        }

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


# ============================================================================
# SIDEBAR
# ============================================================================

init_auth_db()
initialize_auth_session(st.session_state)

# Check for persistent browser login
check_persistent_login(st.session_state)

session_valid = is_session_valid(st.session_state)

st.sidebar.markdown("# Virtual Try-On System")
st.sidebar.markdown("---")

if session_valid:
    user_info = st.session_state.get("auth_user") or {}
    st.sidebar.success(f"Logged in as {user_info.get('username', 'user')}")
    if st.sidebar.button("Logout"):
        clear_persisted_login()
        logout_session(st.session_state)
        st.rerun()

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Select Page:",
        ["Upload & Measure", "Try-On", "Garments"],
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**How it works:**\n\n"
        "1. Upload a photo of yourself\n"
        "2. We detect your body and pose\n"
        "3. We infer your measurements\n"
        "4. We recommend clothing sizes\n"
        "5. Try on different garments!"
    )
else:
    page = None
    st.sidebar.info("Please login to use the application.")


# ============================================================================
# PAGE: Upload & Measure
# ============================================================================

if not session_valid:
    render_auth_page()

elif page == "Upload & Measure":
    st.markdown('<p class="title">Upload & Measure</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a photo to measure your body</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Photo")

        # ── Height input for accurate calibration ────────────────────
        st.markdown("#### 📏 Your Height")
        user_height_cm = st.number_input(
            "Enter your height (cm)",
            min_value=100.0,
            max_value=250.0,
            value=170.0,
            step=1.0,
            help="Your real height is used to calibrate accurate body measurements from the photo."
        )
        if user_height_cm > 0:
            st.success(f"✅ Height set: {user_height_cm:.0f} cm — measurements will be calibrated to your body")
        else:
            st.warning("⚠️ Without your height, measurements will be rough estimates only")

        st.markdown("---")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear front-facing full-body photo of yourself",
        )

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            validation = validate_image(tmp_path)

            if not validation.is_valid:
                st.error("Image validation failed")
                for error in validation.errors:
                    st.write(f"- {error}")
            else:
                if validation.warnings:
                    st.warning("Warnings:")
                    for warning in validation.warnings:
                        st.write(f"- {warning}")

                image = load_image(tmp_path)
                st.image(bgr_to_pil(image), caption="Uploaded image", use_column_width=True)

                if st.button("Analyze Photo", key="analyze_btn", type="primary"):
                    result = process_user_image(tmp_path, user_height_cm=user_height_cm)
                    if result:
                        st.session_state.result    = result
                        st.session_state.temp_path = tmp_path
                        st.success("✅ Image processed! Go to **Try-On** page to try garments.")

    with col2:
        st.subheader("Your Measurements")
        if "result" in st.session_state:
            result       = st.session_state.result
            measurements = result["measurements"]
            pose         = result["pose"]

            # Calibration status badge
            if measurements.calibration_method == 'height':
                st.success(f"✅ **Calibrated** — using your height ({measurements.user_height_cm:.0f} cm)")
            else:
                st.warning("⚠️ **Estimated** — enter your height for accurate measurements")

            st.markdown("#### Body Measurements")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Shoulder Width",  f"{measurements.shoulder_width_cm:.1f} cm")
                st.metric("Torso Length",    f"{measurements.torso_length_cm:.1f} cm")
            with col_m2:
                st.metric("Chest Circumference", f"{measurements.chest_circumference_cm:.1f} cm")
                st.metric("Confidence",          f"{measurements.confidence * 100:.1f}%")

            st.markdown("#### Pose Analysis")
            st.write(f"**Is Frontal:** {'Yes' if pose.is_frontal else 'No'}")
            st.write(f"**Shoulder Width (px):** {pose.shoulder_width_px:.1f}")
            st.write(f"**Keypoints Detected:** {len(pose.keypoints)}")

            if pose.warnings:
                st.warning("**Pose Warnings:**")
                for w in pose.warnings:
                    st.write(f"- {w}")
        else:
            st.info("Upload and analyze a photo to see measurements here.")


# ============================================================================
# PAGE: Try-On
# ============================================================================

elif page == "Try-On":
    st.markdown('<p class="title">Virtual Try-On</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">See how garments fit you</p>', unsafe_allow_html=True)

    # ── Guard ────────────────────────────────────────────────────────
    if "result" not in st.session_state or "temp_path" not in st.session_state:
        st.warning("⚠️ Please upload and analyze a photo first on the **Upload & Measure** page.")
        st.stop()

    result       = st.session_state.result
    measurements = result["measurements"]
    temp_path    = st.session_state.temp_path

    # ── TEMP DEBUG: expand this to see keypoint coordinates ──────────
    pose = result["pose"]
    with st.expander("🔍 Debug: Detected Keypoints (expand to see)", expanded=False):
        st.write(f"**Total keypoints detected:** {len(pose.keypoints)}")
        st.write(f"**Is frontal:** {pose.is_frontal}")
        st.write(f"**Shoulder width (px):** {pose.shoulder_width_px:.1f}")
        st.markdown("---")
        for kp in pose.keypoints:
            st.write(f"**{kp.name}** → x={kp.x:.0f}, y={kp.y:.0f}, conf={kp.confidence:.2f}")

    garments = list_available_garments()
    if not garments:
        st.error("No garments available in the database.")
        st.stop()

    # ── Layout ───────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.subheader("🛍️ Select Garment")

        selected_garment = st.selectbox("Choose garment:", garments, key="tryon_garment_select")

        try:
            metadata   = load_garment_metadata(selected_garment)
            size_chart = metadata.get("size_chart", {})
        except FileNotFoundError:
            st.error(f"Garment not found: {selected_garment}")
            st.stop()

        # Garment thumbnail
        try:
            garment_img = load_garment_image(selected_garment)
            st.image(bgr_to_pil(garment_img), width=220, caption=metadata.get("name", selected_garment))
        except Exception:
            st.caption("(Preview unavailable)")

        # Garment details
        with st.expander("📋 Garment Details", expanded=False):
            st.write(f"**Name:** {metadata.get('name', 'N/A')}")
            st.write(f"**Brand:** {metadata.get('brand', 'N/A')}")
            st.write(f"**Category:** {metadata.get('category', 'N/A')}")
            st.write(f"**Material:** {metadata.get('material', 'N/A')}")
            st.write(f"**Price:** ${metadata.get('price_usd', 0):.2f}")
            colors = metadata.get("available_colors", [])
            if colors:
                st.write(f"**Colors:** {', '.join(colors)}")

        # Size recommendation
        if size_chart:
            recommendation = recommend_size(measurements, size_chart)
            st.success(
                f"📏 Recommended Size: **{recommendation.size}** "
                f"({recommendation.confidence * 100:.0f}% confidence)"
            )

            with st.expander("📊 All Size Fit Scores", expanded=False):
                for size in sorted(recommendation.fit_scores.keys()):
                    score = recommendation.fit_scores[size] * 100
                    st.write(f"**{size}:** {score:.1f}%")
                    st.progress(min(recommendation.fit_scores[size], 1.0))

            with st.expander("📐 Size Chart", expanded=False):
                size_chart_data = []
                for size in sorted(size_chart.keys()):
                    m = size_chart[size]
                    size_chart_data.append({
                        "Size":          size,
                        "Shoulder (cm)": m.get("shoulder_width_cm", "N/A"),
                        "Chest (cm)":    m.get("chest_circumference_cm", "N/A"),
                        "Torso (cm)":    m.get("torso_length_cm", "N/A"),
                    })
                st.dataframe(size_chart_data, use_container_width=True)

        # Fit controls
        st.subheader("⚙️ Fit Settings")
        blend_alpha    = st.slider("Garment opacity",  0.5,  1.0,  1.0, 0.01, key="blend_alpha")
        shoulder_scale = st.slider("Fit width",        0.85, 1.20, 1.00, 0.01, key="shoulder_scale",
                                   help="1.00 = exact fit | >1.00 = looser | <1.00 = tighter")

        st.markdown("---")
        st.subheader("🌐 Cloud Processing Settings")
        use_cloud_api = st.checkbox(
            "Use Cloud API (Colab / IDM-VTON)",
            value=False,
            help="Bypasses local VRAM limits by sending requests to a Google Colab GPU worker or Hugging Face Space."
        )
        cloud_url = st.text_input(
            "Cloud API URL / Space Name:",
            value="yisol/IDM-VTON",
            help="Specify the Hugging Face Space name (e.g., yisol/IDM-VTON) or Gradio sharing live link."
        )

        st.markdown(" ")
        run_tryon = st.button("✨ Try It On", type="primary", use_container_width=True, key="tryon_btn")

    # ── Right column: result ─────────────────────────────────────────
    with col_right:
        st.subheader("🪞 Try-On Preview")

        result_key = f"tryon_result_{selected_garment}"

        if run_tryon:
            with st.spinner("Processing Try-On..."):
                try:
                    person_img  = load_image(temp_path)
                    garment_img = load_garment_image(selected_garment)
                    category    = metadata.get("category", "tshirt").lower()

                    # Initialize result state
                    tryon_success = False
                    composite_bytes = None
                    proc_time = 0.0
                    warnings = []
                    error_msg = ""

                    if use_cloud_api:
                        with st.spinner("Invoking remote IDM-VTON Engine on Cloud..."):
                            from ml_ai.core.cloud_engine import call_cloud_api
                            
                            # IDM-VTON requires local file paths, let's create a temporary file for the garment image
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_garment:
                                pil_garm = bgr_to_pil(garment_img)
                                pil_garm.save(tmp_garment.name, format="PNG")
                                tmp_garment_path = tmp_garment.name
                            
                            try:
                                cloud_result = call_cloud_api(
                                    person_img_path=temp_path,
                                    garment_img_path=tmp_garment_path,
                                    category=category,
                                    client_url=cloud_url
                                )
                                
                                # Unpack results from Gradio Client predict
                                if isinstance(cloud_result, (list, tuple)) and len(cloud_result) > 0:
                                    composite_path = cloud_result[0]
                                    comp_img = cv2.imread(composite_path)
                                    if comp_img is not None:
                                        composite_bytes = image_to_bytes(comp_img)
                                        tryon_success = True
                                    else:
                                        raise ValueError("Failed to load composite image returned from cloud API.")
                                else:
                                    raise ValueError(f"Unexpected prediction response: {cloud_result}")
                            except Exception as cloud_err:
                                st.warning(f"Cloud API Call failed: {cloud_err}. Falling back to local TPS Warping...")
                                use_cloud_api = False
                                
                    if not use_cloud_api:
                        with st.spinner("Warping garment to your body shape locally…"):
                            try:
                                from ml_ai.core.garment_manager import load_garment_mask
                                garment_mask_img = load_garment_mask(selected_garment)
                            except FileNotFoundError:
                                garment_mask_img = None

                            engine       = get_tryon_engine()
                            tryon_result = engine.run(
                                person_image=person_img,
                                garment_image=garment_img,
                                garment_category=category,
                                blend_alpha=blend_alpha,
                                shoulder_scale=shoulder_scale,
                                use_segmentation_mask=True,
                                garment_mask=garment_mask_img,
                            )

                            if tryon_result.success and tryon_result.composite_image is not None:
                                tryon_success = True
                                composite_bytes = image_to_bytes(tryon_result.composite_image)
                                proc_time = tryon_result.processing_time_s
                                warnings = tryon_result.warnings
                            else:
                                error_msg = tryon_result.error
                                warnings = tryon_result.warnings

                    if tryon_success:
                        st.session_state[result_key] = {
                            "success":           True,
                            "composite_bytes":   composite_bytes,
                            "processing_time_s": proc_time,
                            "warnings":          warnings,
                        }
                    else:
                        st.session_state[result_key] = {
                            "success":  False,
                            "error":    error_msg or "Unknown error",
                            "warnings": warnings,
                        }

                except Exception as e:
                    st.session_state[result_key] = {
                        "success":  False,
                        "error":    str(e),
                        "warnings": [],
                    }

        # Display result
        tryon_data = st.session_state.get(result_key)

        if tryon_data is None:
            try:
                person_img = load_image(temp_path)
                st.image(bgr_to_pil(person_img), caption="Your photo — press '✨ Try It On'", use_column_width=True)
            except Exception:
                st.info("Press **✨ Try It On** to see the result here.")

        elif tryon_data.get("success"):
            composite_bytes = tryon_data["composite_bytes"]
            st.image(composite_bytes, caption=f"Wearing: {metadata.get('name', selected_garment)}", use_column_width=True)

            proc_time = tryon_data.get("processing_time_s", 0)
            warnings  = tryon_data.get("warnings", [])
            st.caption(f"⏱️ Processed in {proc_time:.2f}s")

            if warnings:
                with st.expander("⚠️ Warnings", expanded=False):
                    for w in warnings:
                        st.warning(w)

            st.download_button(
                label="⬇️ Download Try-On Image",
                data=composite_bytes,
                file_name=f"tryon_{selected_garment}.png",
                mime="image/png",
                use_container_width=True,
                key=f"download_{selected_garment}"
            )

            with st.expander("🔍 Before / After Comparison", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    try:
                        person_img = load_image(temp_path)
                        st.image(bgr_to_pil(person_img), caption="Original", use_column_width=True)
                    except Exception:
                        st.write("Unavailable")
                with c2:
                    st.image(composite_bytes, caption="With Garment", use_column_width=True)

            # ── Debug visualization ──────────────────────────────────────
            with st.expander("🛠️ Debug: Pipeline Stage Outputs", expanded=False):
                debug_dir = Path("database/data/tryon_debug")
                if not debug_dir.exists():
                    debug_dir = Path("data/tryon_debug")
                if debug_dir.exists():
                    d1, d2, d3 = st.columns(3)
                    wg_path = debug_dir / "debug_warped_garment.png"
                    wm_path = debug_dir / "debug_warped_mask.png"
                    dc_path = debug_dir / "debug_composite.png"
                    if wg_path.exists():
                        with d1:
                            wg = cv2.imread(str(wg_path), cv2.IMREAD_UNCHANGED)
                            st.image(bgr_to_pil(wg), caption="Warped Garment", use_column_width=True)
                    if wm_path.exists():
                        with d2:
                            wm = cv2.imread(str(wm_path), cv2.IMREAD_GRAYSCALE)
                            st.image(wm, caption="Warped Mask", use_column_width=True)
                    if dc_path.exists():
                        with d3:
                            dc = cv2.imread(str(dc_path))
                            st.image(bgr_to_pil(dc), caption="Final Composite", use_column_width=True)
                else:
                    st.info("Debug images not available yet. Run a try-on first.")

        else:
            st.error(f"❌ Try-on failed: {tryon_data.get('error', 'Unknown error')}")
            if tryon_data.get("warnings"):
                for w in tryon_data["warnings"]:
                    st.warning(w)
            if st.button("🔄 Try Again", key="retry_btn"):
                st.session_state.pop(result_key, None)
                st.rerun()


# ============================================================================
# PAGE: Garments
# ============================================================================

elif page == "Garments":
    st.markdown('<p class="title">Browse Garments</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Explore available clothing items</p>', unsafe_allow_html=True)

    garments = list_available_garments()
    if not garments:
        st.error("No garments available")
        st.stop()

    st.subheader(f"Available Garments ({len(garments)})")
    cols = st.columns(3)

    for idx, garment_id in enumerate(garments):
        try:
            metadata = load_garment_metadata(garment_id)
            with cols[idx % 3]:
                st.markdown(f"### {metadata.get('name', garment_id)}")
                try:
                    garment_img = load_garment_image(garment_id)
                    st.image(bgr_to_pil(garment_img), use_column_width=True,
                             caption=metadata.get("name", garment_id))
                except Exception:
                    st.info("No image available")

                st.write(f"**Brand:** {metadata.get('brand', 'N/A')}")
                st.write(f"**Category:** {metadata.get('category', 'N/A')}")
                st.write(f"**Price:** ${metadata.get('price_usd', 0):.2f}")
                colors = metadata.get("available_colors", [])
                if colors:
                    st.write(f"**Colors:** {', '.join(colors)}")

                if st.button(f"Try {metadata.get('name', 'this')}", key=f"try_{garment_id}"):
                    st.session_state.selected_garment = garment_id
                    st.rerun()

        except Exception as e:
            st.error(f"Error loading {garment_id}: {e}")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #94a3b8; margin-top: 2rem;'>
        <p>AI-Based Virtual Try-On and Fit Recommendation System v0.1.0</p>
        <p>Built with Streamlit, OpenCV, MediaPipe and TPS Warping</p>
    </div>
    """,
    unsafe_allow_html=True,
)