"""
Virtual Try-On UI Module
AI-Based Virtual Try-On and Fit Recommendation System

A self-contained Streamlit section that can be called from frontend/app.py:

    from frontend.tryon_ui import render_tryon_section
    render_tryon_section(session_state=st.session_state)

Requires in session_state:
    temp_path   str   — uploaded person image path (set by upload step)
    result      dict  — measurements dict (set by process step), optional
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, MutableMapping

import cv2
import numpy as np
import streamlit as st

from ml_ai.core.garment_manager import (
    list_available_garments,
    load_garment_metadata,
    load_garment_image,
)
from ml_ai.core.tryon_engine import TryOnEngine
from ml_ai.core.image_utils import load_image


# ---------------------------------------------------------------------------
# Engine singleton — one instance per Streamlit session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_engine() -> TryOnEngine:
    """Load TryOnEngine once per app session (cached by Streamlit)."""
    return TryOnEngine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_bytes(image: np.ndarray, fmt: str = "PNG") -> bytes:
    """Convert BGR numpy array to PNG/JPEG bytes for Streamlit display."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(image_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def _load_garment_options() -> dict[str, dict]:
    """
    Return {garment_id: metadata_dict} for all valid garments.
    Silently skips garments that fail to load.
    """
    options = {}
    for gid in list_available_garments():
        try:
            meta = load_garment_metadata(gid)
            options[gid] = meta
        except Exception:
            pass
    return options


def _display_name(garment_id: str, meta: dict) -> str:
    """Format garment display name for selectbox."""
    name     = meta.get("name", garment_id)
    brand    = meta.get("brand", "")
    category = meta.get("category", "").capitalize()
    price    = meta.get("price_usd", 0)
    label    = f"{name}"
    if brand:
        label += f" — {brand}"
    if category:
        label += f" ({category})"
    if price:
        label += f"  ${price:.0f}"
    return label


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_tryon_section(
    session_state: MutableMapping[str, Any],
) -> None:
    """
    Render the full Virtual Try-On UI section.

    Call this from your main frontend/app.py after the user has
    uploaded and processed their photo.
    """

    st.markdown("---")
    st.header("👕 Virtual Try-On")

    # ── Guard: need uploaded photo ────────────────────────────────────
    person_path = session_state.get("temp_path")
    if not person_path or not Path(person_path).exists():
        st.info(
            "📸 Please upload and process your photo first "
            "(Steps 1 & 2 above) to use the try-on feature."
        )
        return

    # ── Load available garments ───────────────────────────────────────
    garment_options = _load_garment_options()
    if not garment_options:
        st.error(
            "No garments found in the database. "
            "Please add garments to `database/data/garments/`."
        )
        return

    # ── Layout: controls left, preview right ─────────────────────────
    col_controls, col_preview = st.columns([1, 1.6], gap="large")

    with col_controls:
        st.subheader("🛍️ Select Garment")

        # Garment selectbox
        garment_ids    = list(garment_options.keys())
        display_names  = [_display_name(gid, garment_options[gid]) for gid in garment_ids]
        selected_index = st.selectbox(
            "Choose a garment",
            options=range(len(garment_ids)),
            format_func=lambda i: display_names[i],
            key="tryon_garment_select"
        )
        selected_id   = garment_ids[selected_index]
        selected_meta = garment_options[selected_id]

        # Show garment thumbnail
        try:
            garment_img_raw = load_garment_image(selected_id)
            garment_thumb   = cv2.resize(garment_img_raw, (220, 220))
            st.image(
                _image_to_bytes(garment_thumb),
                caption=selected_meta.get("name", selected_id),
                use_container_width=False,
                width=220
            )
        except Exception:
            st.caption("(Garment preview unavailable)")

        # Garment details
        with st.expander("📋 Garment details", expanded=False):
            st.write(f"**Brand:** {selected_meta.get('brand', '—')}")
            st.write(f"**Category:** {selected_meta.get('category', '—').capitalize()}")
            st.write(f"**Material:** {selected_meta.get('material', '—')}")
            st.write(f"**Price:** ${selected_meta.get('price_usd', 0):.2f}")
            if selected_meta.get("description"):
                st.write(f"**Description:** {selected_meta['description']}")

        st.subheader("⚙️ Fit Settings")

        blend_alpha = st.slider(
            "Garment opacity",
            min_value=0.5,
            max_value=1.0,
            value=0.92,
            step=0.01,
            key="tryon_blend_alpha",
            help="Higher = more opaque garment"
        )

        shoulder_scale = st.slider(
            "Fit width",
            min_value=0.85,
            max_value=1.20,
            value=1.00,
            step=0.01,
            key="tryon_shoulder_scale",
            help="1.00 = exact fit  |  >1.00 = looser  |  <1.00 = tighter"
        )

        # Show recommended size if measurements available
        result_data = session_state.get("result")
        if result_data and "measurements" in result_data:
            size_chart = selected_meta.get("size_chart", {})
            if size_chart:
                from ml_ai.core.models import Measurements
                from ml_ai.core.size_recommendation import recommend_size
                meas = result_data["measurements"]
                try:
                    measurements = Measurements(
                        shoulder_width_cm=meas.get("shoulder_width_cm", 0),
                        chest_circumference_cm=meas.get("chest_circumference_cm", 0),
                        torso_length_cm=meas.get("torso_length_cm", 0),
                        source="inferred",
                        confidence=meas.get("confidence", 0.8)
                    )
                    rec = recommend_size(measurements, size_chart)
                    st.success(
                        f"📏 Recommended size: **{rec.size}** "
                        f"({rec.confidence * 100:.0f}% confidence)"
                    )
                except Exception:
                    pass

        # Try-On button
        st.markdown(" ")
        run_tryon = st.button(
            "✨ Try It On",
            type="primary",
            use_container_width=True,
            key="tryon_run_btn"
        )

    # ── Preview column ────────────────────────────────────────────────
    with col_preview:
        st.subheader("🪞 Try-On Preview")

        result_key = f"tryon_result_{selected_id}"

        if run_tryon:
            with st.spinner("Warping garment to your body shape…"):
                _run_and_store(
                    session_state=session_state,
                    person_path=person_path,
                    selected_id=selected_id,
                    garment_options=garment_options,
                    blend_alpha=blend_alpha,
                    shoulder_scale=shoulder_scale,
                    result_key=result_key,
                )

        # Display result or placeholder
        tryon_result = session_state.get(result_key)

        if tryon_result is None:
            # Show person photo as placeholder
            try:
                person_img = load_image(person_path)
                st.image(
                    _image_to_bytes(person_img),
                    caption="Your photo — press 'Try It On' to see the result",
                    use_container_width=True
                )
            except Exception:
                st.info("Press **Try It On** to see the result here.")

        elif tryon_result.get("success"):
            composite_bytes = tryon_result["composite_bytes"]
            st.image(
                composite_bytes,
                caption=f"Wearing: {selected_meta.get('name', selected_id)}",
                use_container_width=True
            )

            # Processing info
            proc_time = tryon_result.get("processing_time_s", 0)
            warnings  = tryon_result.get("warnings", [])

            st.caption(f"⏱️ Processed in {proc_time:.2f}s")

            if warnings:
                with st.expander("⚠️ Warnings", expanded=False):
                    for w in warnings:
                        st.warning(w)

            # Download button
            st.download_button(
                label="⬇️ Download Try-On Image",
                data=composite_bytes,
                file_name=f"tryon_{selected_id}.png",
                mime="image/png",
                use_container_width=True,
                key=f"tryon_download_{selected_id}"
            )

            # Side-by-side comparison
            with st.expander("🔍 Before / After comparison", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    try:
                        person_img = load_image(person_path)
                        st.image(
                            _image_to_bytes(person_img),
                            caption="Original",
                            use_container_width=True
                        )
                    except Exception:
                        st.write("Original unavailable")
                with c2:
                    st.image(
                        composite_bytes,
                        caption="With garment",
                        use_container_width=True
                    )

        else:
            # Show error
            st.error(f"❌ Try-on failed: {tryon_result.get('error', 'Unknown error')}")
            if tryon_result.get("warnings"):
                for w in tryon_result["warnings"]:
                    st.warning(w)
            if st.button("🔄 Try again", key="tryon_retry_btn"):
                session_state.pop(result_key, None)
                st.rerun()


# ---------------------------------------------------------------------------
# Internal: run engine and store result in session_state
# ---------------------------------------------------------------------------

def _run_and_store(
    session_state: MutableMapping[str, Any],
    person_path: str,
    selected_id: str,
    garment_options: dict,
    blend_alpha: float,
    shoulder_scale: float,
    result_key: str,
) -> None:
    """Run TryOnEngine and persist result (as bytes) in session_state."""
    try:
        person_img   = load_image(person_path)
        garment_img  = load_garment_image(selected_id)
        garment_meta = garment_options[selected_id]
        category     = garment_meta.get("category", "tshirt").lower()

        engine = _get_engine()
        result = engine.run(
            person_image=person_img,
            garment_image=garment_img,
            garment_category=category,
            blend_alpha=blend_alpha,
            shoulder_scale=shoulder_scale,
            use_segmentation_mask=True,
        )

        if result.success and result.composite_image is not None:
            composite_bytes = _image_to_bytes(result.composite_image)
            session_state[result_key] = {
                "success":           True,
                "composite_bytes":   composite_bytes,
                "processing_time_s": result.processing_time_s,
                "warnings":          result.warnings,
            }
        else:
            session_state[result_key] = {
                "success":  False,
                "error":    result.error,
                "warnings": result.warnings,
            }

    except Exception as e:
        session_state[result_key] = {
            "success":  False,
            "error":    str(e),
            "warnings": [],
        }