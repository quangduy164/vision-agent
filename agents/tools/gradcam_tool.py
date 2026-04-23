# agents/tools/gradcam_tool.py
"""
LangChain Tool: generate_gradcam
Tạo heatmap Grad-CAM từ DenseNet121 cho bệnh được chỉ định.
"""
import os
import json
import time
import logging
import numpy as np
import torch
import cv2
from PIL import Image
from datetime import datetime
from langchain.tools import tool

import torchxrayvision as xrv
from models.classifier import CLASSES, _transform_224
from models.gradcam import generate_heatmap
from agents.tools.model_registry import get_models

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@tool
def generate_gradcam(image_path: str, disease_name: str, output_dir: str = "outputs") -> str:
    """
    Generate a Grad-CAM heatmap for a specific disease on a chest X-ray.
    Input:
      - image_path: path to the X-ray image
      - disease_name: one of the 17 disease classes (e.g. 'Cardiomegaly')
      - output_dir: directory to save the heatmap (default: 'outputs')
    Output: JSON string with key 'heatmap_path'.
    Call this tool AFTER classify_xray when diagnosis is not 'No Finding'.
    """
    t0 = time.time()
    model_dense, _, _ = get_models()
    try:
        if disease_name not in CLASSES:
            return json.dumps({"error": f"Invalid disease: {disease_name}"})
        if not os.path.exists(image_path):
            return json.dumps({"error": f"Image not found: {image_path}"})

        os.makedirs(output_dir, exist_ok=True)

        img_pil = Image.open(image_path).convert("L")
        img_pil_224 = img_pil.resize((224, 224))
        img_np  = np.array(img_pil_224)

        # show_cam_on_image cần RGB [0,1] - convert grayscale sang RGB
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        img_vis = img_rgb.astype(np.float32) / 255.0

        # Dùng cùng transform với classify (giống Kaggle)
        img_tensor = _transform_224(img_pil).unsqueeze(0).to(DEVICE)

        # Dùng index theo CLASSES (thứ tự output của model đã fine-tune)
        # KHÔNG dùng model_dense.pathologies vì lớp classifier đã được thay
        target_idx = CLASSES.index(disease_name)
        heatmap_vis, _ = generate_heatmap(
            model_dense, img_tensor, img_vis, target_idx
        )

        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"heatmap_{disease_name}_{ts}.png")
        cv2.imwrite(save_path, cv2.cvtColor(heatmap_vis, cv2.COLOR_RGB2BGR))
        logger.info(f"[generate_gradcam] saved={save_path} | {(time.time()-t0)*1000:.0f}ms")
        return json.dumps({"heatmap_path": save_path})

    except Exception as e:
        logger.error(f"[generate_gradcam] {e}", exc_info=True)
        return json.dumps({"error": str(e)})
