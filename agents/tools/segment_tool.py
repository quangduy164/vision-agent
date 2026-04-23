# agents/tools/segment_tool.py
"""
LangChain Tool: segment_lesion
Phân vùng tổn thương từ Grad-CAM → trả về vị trí, kích thước, bên trái/phải.
"""
import os
import json
import time
import logging
import numpy as np
import torch
import cv2
from PIL import Image
from langchain.tools import tool

import torchxrayvision as xrv
from models.classifier import CLASSES, _transform_224
from models.gradcam import generate_heatmap
from models.segmenter import segment_from_cam, get_location_text
from agents.tools.model_registry import get_models

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_DEFAULT = {"location": "chest", "size": "moderate", "side": "unspecified"}


@tool
def segment_lesion(image_path: str, disease_name: str) -> str:
    """
    Segment the lesion region in a chest X-ray and return spatial information.
    Input:
      - image_path: path to the X-ray image
      - disease_name: disease to localize (one of the 17 CLASSES)
    Output: JSON string with keys:
      - location: anatomical location (e.g. 'right lung lower zone/base')
      - size: lesion size ('focal', 'moderate', 'large/extensive')
      - side: laterality ('left', 'right', 'central/mediastinal', 'unspecified')
    Call this tool AFTER classify_xray to get spatial context for report generation.
    """
    t0 = time.time()
    model_dense, _, _ = get_models()
    try:
        if disease_name not in CLASSES or disease_name == "No Finding":
            return json.dumps(_DEFAULT)
        if not os.path.exists(image_path):
            return json.dumps({"error": f"Image not found: {image_path}"})

        img_pil = Image.open(image_path).convert("L")
        img_pil_224 = img_pil.resize((224, 224))
        img_np  = np.array(img_pil_224)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        img_vis = img_rgb.astype(np.float32) / 255.0

        # Dùng cùng transform với classify
        img_tensor = _transform_224(img_pil).unsqueeze(0).to(DEVICE)

        # Index theo CLASSES (output của model fine-tuned)
        target_idx = CLASSES.index(disease_name)
        _, gray_cam = generate_heatmap(
            model_dense, img_tensor, img_vis, target_idx
        )

        mask = segment_from_cam(gray_cam, threshold=gray_cam.max() * 0.6)
        loc, size, side = get_location_text(mask, 224, 224)

        logger.info(f"[segment_lesion] loc={loc}, size={size}, side={side} | {(time.time()-t0)*1000:.0f}ms")
        return json.dumps({"location": loc, "size": size, "side": side})

    except Exception as e:
        logger.error(f"[segment_lesion] {e}", exc_info=True)
        return json.dumps(_DEFAULT)
