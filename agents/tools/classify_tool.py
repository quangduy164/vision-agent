# agents/tools/classify_tool.py
"""
LangChain Tool: classify_xray
Ensemble DenseNet121 (224x224) + ResNet50 (512x512) → 17 nhãn bệnh.
"""
import os
import json
import time
import logging
from PIL import Image
from langchain.tools import tool

from models.classifier import predict_ensemble, get_threshold
from agents.tools.model_registry import get_models

logger = logging.getLogger(__name__)


@tool
def classify_xray(image_path: str) -> str:
    """
    Classify a chest X-ray image using ensemble DenseNet121 + ResNet50.
    Input: path to the X-ray image file (PNG/JPG).
    Output: JSON string with keys:
      - diagnoses: list of diseases exceeding their optimal threshold
      - top_disease: disease with highest probability
      - top_prob: probability of top disease (float 0-1)
      - all_probabilities: dict of all 17 disease probabilities
    Call this tool FIRST before any other tool.
    """
    t0 = time.time()
    model_dense, model_res, class_mapping_res = get_models()
    try:
        if not os.path.exists(image_path):
            return json.dumps({"error": f"Image not found: {image_path}"})

        img_pil = Image.open(image_path).convert("L")
        probs = predict_ensemble(model_dense, model_res, class_mapping_res, img_pil)

        sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

        # top_disease = bệnh có prob cao nhất tuyệt đối (giống Kaggle)
        top_disease = list(sorted_probs.keys())[0]
        top_prob    = list(sorted_probs.values())[0]

        # diagnoses = tất cả bệnh vượt ngưỡng tối ưu (không tính No Finding)
        diagnoses = [
            d for d, p in sorted_probs.items()
            if p >= get_threshold(d) and d != "No Finding"
        ]

        # Nếu không có bệnh nào vượt ngưỡng → No Finding
        if not diagnoses:
            top_disease = "No Finding"
            top_prob    = sorted_probs.get("No Finding", 0.0)
            diagnoses   = ["No Finding"]

        logger.info(f"[classify_xray] top={top_disease} ({top_prob:.2%}) | {(time.time()-t0)*1000:.0f}ms")
        return json.dumps({
            "diagnoses": diagnoses,
            "top_disease": top_disease,
            "top_prob": round(top_prob, 4),
            "all_probabilities": {k: round(v, 4) for k, v in sorted_probs.items()},
        })
    except Exception as e:
        logger.error(f"[classify_xray] {e}", exc_info=True)
        return json.dumps({"error": str(e)})
