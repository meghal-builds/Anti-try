"""Flask API for Virtual Try-On System
AI-Based Virtual Try-On and Fit Recommendation System

Endpoints:
    GET  /api/health
    POST /api/upload
    POST /api/process
    POST /api/recommend
    POST /api/tryon          ← NEW Phase 3
    GET  /api/garments
    GET  /api/garments/<id>
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

import cv2
import numpy as np
import werkzeug
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from ml_ai.core.image_utils import load_image, save_image, get_image_dimensions
from ml_ai.core.validation import validate_image
from ml_ai.core.model_layer import load_models
from ml_ai.core.segmentation import segment_body
from ml_ai.core.pose_detection import detect_pose
from ml_ai.core.measurement_inference import infer_measurements, validate_measurements
from ml_ai.core.garment_manager import list_available_garments, load_garment_metadata, load_garment_image
from ml_ai.core.size_recommendation import recommend_size
from ml_ai.core.tryon_engine import TryOnEngine


# ── Configuration ────────────────────────────────────────────────────────────

UPLOAD_FOLDER = Path("database/data/uploads")
if not UPLOAD_FOLDER.parent.exists() and Path("data").exists():
    UPLOAD_FOLDER = Path("data/uploads")

OUTPUT_FOLDER = Path("database/data/tryon_results")
if not OUTPUT_FOLDER.parent.exists() and Path("data").exists():
    OUTPUT_FOLDER = Path("data/tryon_results")

# Flask 3.0 / Werkzeug compatibility shim
if not hasattr(werkzeug, "__version__"):
    werkzeug.__version__ = "3"

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ── App setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
CORS(app)

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ── Model loading (lazy on first request) ────────────────────────────────────

_seg_model = None
_pose_model = None
_tryon_engine: TryOnEngine | None = None


def get_models():
    """Lazy-load segmentation and pose models."""
    global _seg_model, _pose_model
    if _seg_model is None or _pose_model is None:
        _seg_model, _pose_model = load_models()
    return _seg_model, _pose_model


def get_tryon_engine() -> TryOnEngine:
    """Lazy-load and reuse TryOnEngine singleton."""
    global _tryon_engine
    if _tryon_engine is None:
        _tryon_engine = TryOnEngine()
    return _tryon_engine


# ── Utilities ────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + os.urandom(4).hex()


def _safe_image_path(raw_path: str, base_folder: Path) -> tuple[Path | None, str]:
    """
    Resolve and validate that raw_path is inside base_folder.
    Returns (resolved_path, error_message).  error_message is "" on success.
    """
    try:
        resolved = Path(raw_path).resolve()
        base     = base_folder.resolve()
        if not str(resolved).startswith(str(base)):
            return None, "Invalid image path"
        if not resolved.exists():
            return None, "Image file not found"
        return resolved, ""
    except Exception:
        return None, "Invalid image path"


# ── Existing endpoints (unchanged) ───────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health_check():
    seg, pose = get_models()
    return jsonify({
        'status': 'healthy',
        'models_loaded': seg is not None and pose is not None,
        'tryon_engine_ready': _tryon_engine is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'File type not allowed'}), 400

        session_id = generate_session_id()
        filename   = secure_filename(f"{session_id}_{file.filename}")
        filepath   = UPLOAD_FOLDER / filename
        file.save(str(filepath))

        validation_result = validate_image(str(filepath))
        if not validation_result.is_valid:
            filepath.unlink(missing_ok=True)
            return jsonify({
                'success': False,
                'error': 'Image validation failed',
                'errors': validation_result.errors,
                'warnings': validation_result.warnings
            }), 400

        height, width = get_image_dimensions(str(filepath))

        return jsonify({
            'success': True,
            'session_id': session_id,
            'image_path': str(filepath),
            'dimensions': {'height': height, 'width': width},
            'validation': {
                'is_valid': validation_result.is_valid,
                'warnings': validation_result.warnings
            }
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/process', methods=['POST'])
def process_image():
    try:
        seg_model, pose_model = get_models()
        data       = request.get_json()
        raw_path   = data.get('image_path', '')
        user_height_cm = float(data.get('user_height_cm', 0))
        safe_path, err = _safe_image_path(raw_path, UPLOAD_FOLDER)
        if err:
            return jsonify({'success': False, 'error': err}), 400

        image      = load_image(str(safe_path))
        image_height = image.shape[0]
        seg_result = segment_body(image, seg_model)

        try:
            pose_result = detect_pose(image, pose_model)
        except RuntimeError as e:
            return jsonify({'success': False, 'error': 'Pose detection failed',
                            'details': str(e)}), 400

        measurements = infer_measurements(
            pose_result, seg_result,
            image_height=image_height,
            user_height_cm=user_height_cm
        )
        is_valid, errors = validate_measurements(measurements)
        if not is_valid:
            return jsonify({'success': False,
                            'error': 'Measurement validation failed',
                            'details': errors}), 400

        return jsonify({
            'success': True,
            'measurements': {
                'shoulder_width_cm':      round(measurements.shoulder_width_cm, 2),
                'chest_circumference_cm': round(measurements.chest_circumference_cm, 2),
                'torso_length_cm':        round(measurements.torso_length_cm, 2),
                'confidence':             round(measurements.confidence, 2),
                'calibration_method':     measurements.calibration_method,
                'user_height_cm':         measurements.user_height_cm
            },
            'pose': {
                'shoulder_width_px': round(pose_result.shoulder_width_px, 2),
                'is_frontal':        pose_result.is_frontal,
                'keypoint_count':    len(pose_result.keypoints),
                'warnings':          pose_result.warnings
            },
            'segmentation': {
                'confidence':        round(seg_result.confidence, 2),
                'torso_percentage':  round(seg_result.torso_percentage, 2),
                'warnings':          seg_result.warnings
            }
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/recommend', methods=['POST'])
def recommend_size_endpoint():
    try:
        from ml_ai.core.models import Measurements
        data      = request.get_json()
        meas_data = data.get('measurements', {})

        # Validate required fields present
        required = ['shoulder_width_cm', 'chest_circumference_cm', 'torso_length_cm']
        missing  = [f for f in required if not meas_data.get(f)]
        if missing:
            return jsonify({'success': False,
                            'error': f"Missing measurement fields: {missing}"}), 400

        measurements = Measurements(
            shoulder_width_cm=float(meas_data['shoulder_width_cm']),
            chest_circumference_cm=float(meas_data['chest_circumference_cm']),
            torso_length_cm=float(meas_data['torso_length_cm']),
            source='inferred',
            confidence=float(meas_data.get('confidence', 0.8))
        )

        garment_id = data.get('garment_id')
        if not garment_id:
            return jsonify({'success': False, 'error': 'garment_id required'}), 400

        try:
            metadata = load_garment_metadata(garment_id)
        except FileNotFoundError:
            return jsonify({'success': False,
                            'error': f'Garment not found: {garment_id}'}), 404

        size_chart     = metadata.get('size_chart', {})
        recommendation = recommend_size(measurements, size_chart)

        return jsonify({
            'success': True,
            'recommendation': {
                'size':               recommendation.size,
                'confidence':         round(recommendation.confidence, 2),
                'recommended_sizes':  recommendation.recommended_sizes,
                'fit_scores': {
                    size: round(score, 2)
                    for size, score in recommendation.fit_scores.items()
                }
            }
        }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/garments', methods=['GET'])
def get_garments():
    try:
        garment_ids = list_available_garments()
        garments    = []
        errors      = []

        for garment_id in garment_ids:
            try:
                meta = load_garment_metadata(garment_id)
                garments.append({
                    'id':        meta.get('id'),
                    'name':      meta.get('name'),
                    'category':  meta.get('category'),
                    'brand':     meta.get('brand'),
                    'price_usd': meta.get('price_usd', 0)
                })
            except Exception as e:
                errors.append(f"{garment_id}: {str(e)}")

        response = {'success': True, 'count': len(garments), 'garments': garments}
        if errors:
            response['load_errors'] = errors
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/garments/<garment_id>', methods=['GET'])
def get_garment_details(garment_id: str):
    try:
        metadata = load_garment_metadata(garment_id)
        return jsonify({
            'success': True,
            'garment': {
                'id':               metadata.get('id'),
                'name':             metadata.get('name'),
                'category':         metadata.get('category'),
                'brand':            metadata.get('brand'),
                'description':      metadata.get('description', ''),
                'material':         metadata.get('material', ''),
                'price_usd':        metadata.get('price_usd', 0),
                'available_colors': metadata.get('available_colors', []),
                'size_chart':       metadata.get('size_chart', {})
            }
        }), 200
    except FileNotFoundError:
        return jsonify({'success': False,
                        'error': f'Garment not found: {garment_id}'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── NEW: Virtual Try-On endpoint ─────────────────────────────────────────────

@app.route('/api/tryon', methods=['POST'])
def virtual_tryon():
    """
    Run virtual try-on: warp garment onto person and return result image.

    Request (multipart/form-data OR JSON):
        person_image_path  str   — server path from /api/upload response
        garment_id         str   — garment ID from /api/garments
        blend_alpha        float — garment opacity 0..1  (default 0.92)
        shoulder_scale     float — fit width multiplier  (default 1.0)
        return_base64      bool  — if true, embed image in JSON response
                                   if false (default), return image file

    Returns (return_base64=false):
        image/png file of the composite result

    Returns (return_base64=true):
        {
            'success': true,
            'session_id': str,
            'result_path': str,
            'processing_time_s': float,
            'warnings': [...],
            'image_base64': str   (PNG bytes, base64-encoded)
        }
    """
    try:
        # ── Parse request (supports both JSON and form-data) ──────────
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        person_image_path = data.get('person_image_path', '')
        garment_id        = data.get('garment_id', '')
        blend_alpha       = float(data.get('blend_alpha', 0.92))
        shoulder_scale    = float(data.get('shoulder_scale', 1.0))
        return_base64     = str(data.get('return_base64', 'false')).lower() == 'true'

        # ── Validate person image path ────────────────────────────────
        safe_path, err = _safe_image_path(person_image_path, UPLOAD_FOLDER)
        if err:
            return jsonify({'success': False, 'error': err}), 400

        # ── Validate garment ──────────────────────────────────────────
        if not garment_id:
            return jsonify({'success': False, 'error': 'garment_id required'}), 400

        try:
            garment_meta  = load_garment_metadata(garment_id)
            garment_img   = load_garment_image(garment_id)
        except FileNotFoundError:
            return jsonify({'success': False,
                            'error': f'Garment not found: {garment_id}'}), 404

        garment_category = garment_meta.get('category', 'tshirt').lower()

        # ── Load pre-computed garment mask (from normalization) ────────
        garment_mask_img = None
        try:
            from ml_ai.core.garment_manager import load_garment_mask
            garment_mask_img = load_garment_mask(garment_id)
        except FileNotFoundError:
            pass  # Garment not normalized yet — engine will use fallback mask

        # ── Validate parameters ───────────────────────────────────────
        if not (0.1 <= blend_alpha <= 1.0):
            return jsonify({'success': False,
                            'error': 'blend_alpha must be between 0.1 and 1.0'}), 400
        if not (0.5 <= shoulder_scale <= 2.0):
            return jsonify({'success': False,
                            'error': 'shoulder_scale must be between 0.5 and 2.0'}), 400

        # ── Load person image ─────────────────────────────────────────
        person_img = load_image(str(safe_path))

        # ── Run try-on engine ─────────────────────────────────────────
        engine = get_tryon_engine()
        result = engine.run(
            person_image=person_img,
            garment_image=garment_img,
            garment_category=garment_category,
            blend_alpha=blend_alpha,
            shoulder_scale=shoulder_scale,
            use_segmentation_mask=True,
            garment_mask=garment_mask_img,
        )

        if not result.success:
            return jsonify({
                'success': False,
                'error': result.error,
                'warnings': result.warnings
            }), 422

        # ── Save result image ─────────────────────────────────────────
        session_id  = generate_session_id()
        result_filename = f"tryon_{session_id}_{garment_id}.png"
        result_path = OUTPUT_FOLDER / result_filename
        save_image(result.composite_image, str(result_path))

        # ── Return response ───────────────────────────────────────────
        if return_base64:
            import base64
            _, buf = cv2.imencode('.png', result.composite_image)
            b64     = base64.b64encode(buf.tobytes()).decode('utf-8')
            return jsonify({
                'success':           True,
                'session_id':        session_id,
                'result_path':       str(result_path),
                'processing_time_s': result.processing_time_s,
                'warnings':          result.warnings,
                'image_base64':      b64
            }), 200
        else:
            _, buf = cv2.imencode('.png', result.composite_image)
            from flask import Response
            return Response(
                buf.tobytes(),
                mimetype='image/png',
                headers={
                    'X-Processing-Time': str(result.processing_time_s),
                    'X-Session-Id':      session_id,
                    'X-Warnings':        json.dumps(result.warnings),
                    'Content-Disposition': f'inline; filename="{result_filename}"'
                }
            )

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/tryon/result/<filename>', methods=['GET'])
def get_tryon_result(filename: str):
    """
    Retrieve a previously saved try-on result by filename.

    Returns:
        image/png file
    """
    try:
        safe_filename = secure_filename(filename)
        result_path   = OUTPUT_FOLDER / safe_filename

        if not result_path.exists():
            return jsonify({'success': False, 'error': 'Result not found'}), 404

        # Verify it's inside OUTPUT_FOLDER
        if not str(result_path.resolve()).startswith(str(OUTPUT_FOLDER.resolve())):
            return jsonify({'success': False, 'error': 'Invalid path'}), 403

        return send_file(str(result_path), mimetype='image/png')

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'success': False, 'error': 'File too large (max 10MB)'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)