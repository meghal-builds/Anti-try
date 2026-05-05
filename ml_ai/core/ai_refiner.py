"""
Phase 4 — AI Realism Refinement Module
Virtual Try-On Pipeline

Uses Stable Diffusion 1.5 (img2img) with ControlNet Canny conditioning
to upgrade Phase 3 composites into photorealistic outputs while preserving
garment structure, body pose, and facial identity.

Safety:
    - If torch / diffusers are not installed → returns input unchanged.
    - If no CUDA GPU is available → returns input unchanged.
    - If any runtime error occurs → returns input unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed prompts (intentionally simple — do NOT overcomplicate)
# ---------------------------------------------------------------------------
_POSITIVE_PROMPT = (
    "a realistic photo of a person wearing a t-shirt, "
    "natural lighting, detailed fabric, realistic folds, "
    "no artifacts, high quality"
)

_NEGATIVE_PROMPT = (
    "blurry, distorted, extra limbs, unrealistic cloth, "
    "artifacts, deformed body"
)

# ---------------------------------------------------------------------------
# Diffusion settings
# ---------------------------------------------------------------------------
_STRENGTH = 0.30          # low — preserves ~70 % of original pixels
_GUIDANCE_SCALE = 7.0
_NUM_STEPS = 25
_CONTROLNET_SCALE = 0.8   # strong structural guidance
_CANNY_LOW = 100
_CANNY_HIGH = 200

# SD 1.5 operates at multiples of 8; we round to nearest 64 for safety
_ROUND_TO = 64


class AIRefiner:
    """
    Diffusion-based photorealistic refinement for virtual try-on composites.

    Lifecycle
    ---------
    >>> refiner = AIRefiner()
    >>> refined = refiner.refine(phase3_composite)   # lazy-loads on first call

    The pipeline is loaded **once** and reused across calls.
    """

    def __init__(self) -> None:
        self._pipe = None               # lazy-loaded diffusion pipeline
        self._available: Optional[bool] = None  # None = not yet checked

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine(self, image: np.ndarray) -> np.ndarray:
        """
        Refine a Phase 3 composite into a photorealistic image.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image (H, W, 3) — the Phase 3 composite.

        Returns
        -------
        np.ndarray
            Refined BGR uint8 image at the **same** resolution as input.
            If refinement is unavailable, returns the input unchanged.
        """
        if not self.is_available():
            logger.info("AI refiner not available — returning Phase 3 output.")
            return image

        try:
            return self._run_refinement(image)
        except Exception as e:
            logger.warning(f"AI refinement failed — falling back to Phase 3: {e}")
            return image

    def is_available(self) -> bool:
        """Return True if the diffusion pipeline can be loaded."""
        if self._available is not None:
            return self._available

        try:
            import torch
            if not torch.cuda.is_available():
                logger.info("No CUDA GPU detected — AI refiner disabled.")
                self._available = False
                return False

            # Quick sanity check for diffusers
            import diffusers  # noqa: F401
            self._available = True
            return True

        except ImportError as e:
            logger.info(f"Missing dependency for AI refiner: {e}")
            self._available = False
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """Lazy-load Stable Diffusion 1.5 + ControlNet Canny pipeline."""
        if self._pipe is not None:
            return

        import torch
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetImg2ImgPipeline,
            UniPCMultistepScheduler,
        )

        logger.info("Loading ControlNet Canny model …")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16,
        )

        logger.info("Loading Stable Diffusion 1.5 + ControlNet pipeline …")
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,           # skip NSFW filter for try-on
            requires_safety_checker=False,
        )

        # Faster scheduler — same quality, fewer steps
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config
        )

        # Memory optimisations - CRUCIAL for 4GB VRAM (RTX 3050)
        # Instead of pipe.to("cuda") which loads everything at once,
        # we offload parts of the model to CPU when not actively computing.
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory-efficient attention enabled.")
        except Exception:
            pass  # xformers optional

        self._pipe = pipe
        logger.info("AI refiner pipeline loaded and ready.")

    def _run_refinement(self, image: np.ndarray) -> np.ndarray:
        """Core refinement logic — assumes availability already checked."""
        import torch
        from PIL import Image as PILImage

        self._load_pipeline()

        orig_h, orig_w = image.shape[:2]

        # ── 1. Convert BGR → RGB PIL ──────────────────────────────────
        input_pil = self._to_pil(image)

        # ── 2. Resize to SD-compatible dimensions (multiple of 64) ────
        sd_w = self._round_dim(orig_w)
        sd_h = self._round_dim(orig_h)
        input_resized = input_pil.resize((sd_w, sd_h), PILImage.LANCZOS)

        # ── 3. Extract Canny edge map for ControlNet conditioning ─────
        canny_pil = self._extract_canny(image, sd_w, sd_h)

        # ── 4. Run diffusion ──────────────────────────────────────────
        logger.info(
            f"Running AI refinement: {sd_w}×{sd_h}, "
            f"strength={_STRENGTH}, steps={_NUM_STEPS}"
        )

        with torch.inference_mode():
            result = self._pipe(
                prompt=_POSITIVE_PROMPT,
                negative_prompt=_NEGATIVE_PROMPT,
                image=input_resized,
                control_image=canny_pil,
                strength=_STRENGTH,
                guidance_scale=_GUIDANCE_SCALE,
                controlnet_conditioning_scale=_CONTROLNET_SCALE,
                num_inference_steps=_NUM_STEPS,
            )

        refined_pil = result.images[0]

        # ── 5. Resize back to original resolution ─────────────────────
        if (refined_pil.width, refined_pil.height) != (orig_w, orig_h):
            refined_pil = refined_pil.resize(
                (orig_w, orig_h), PILImage.LANCZOS
            )

        # ── 6. Convert RGB PIL → BGR numpy ────────────────────────────
        refined_bgr = self._to_bgr(refined_pil)

        logger.info("AI refinement complete.")
        return refined_bgr

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pil(bgr: np.ndarray):
        """BGR uint8 numpy → RGB PIL Image."""
        from PIL import Image as PILImage

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb)

    @staticmethod
    def _to_bgr(pil_img) -> np.ndarray:
        """RGB PIL Image → BGR uint8 numpy."""
        rgb = np.array(pil_img, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _extract_canny(
        image: np.ndarray,
        target_w: int,
        target_h: int,
    ):
        """
        Generate a Canny edge map from the input image,
        resized to the target SD dimensions.

        Returns an RGB PIL Image (3-channel Canny map).
        """
        from PIL import Image as PILImage

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, _CANNY_LOW, _CANNY_HIGH)

        # Resize to match SD input dimensions
        edges = cv2.resize(edges, (target_w, target_h))

        # ControlNet expects 3-channel RGB image
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return PILImage.fromarray(edges_rgb)

    @staticmethod
    def _round_dim(dim: int) -> int:
        """Round a dimension to the nearest multiple of _ROUND_TO."""
        return max(_ROUND_TO, (dim + _ROUND_TO // 2) // _ROUND_TO * _ROUND_TO)
