# agent.py
import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchxrayvision as xrv

from models.classifier import load_model, predict, NIH_LABELS
from models.gradcam import generate_heatmap
from models.segmenter import segment_from_cam
from models.vit_attention import generate_vit_attention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================
# Crop lung region (heuristic)
# ===============================
def crop_lung_region(img: Image.Image):
    w, h = img.size
    return img.crop((
        int(0.15 * w),
        int(0.18 * h),
        int(0.85 * w),
        int(0.92 * h)
    ))


# ===============================
# MAIN PIPELINE
# ===============================
def run(image_path, model_name="densenet121_nih"):
    os.makedirs("outputs", exist_ok=True)

    model = load_model(model_name)
    is_vit = model_name == "vit_base_nih"
    is_resnet = "resnet50" in model_name

    IMG_SIZE = 512 if is_resnet else 224

    # ===============================
    # LOAD & PREPROCESS IMAGE
    # ===============================
    if is_vit:
        # ---- ViT NIH (RGB) ----
        img = Image.open(image_path).convert("RGB")
        img = crop_lung_region(img)
        img = img.resize((224, 224))

        img_np = np.array(img).astype(np.float32) / 255.0

        image_tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(DEVICE)
        )

        img_rgb = img_np

    else:
        # ---- CNN NIH (Grayscale) ----
        img = Image.open(image_path).convert("L")
        img = crop_lung_region(img)
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_np = np.array(img).astype(np.float32)
        img_np = xrv.datasets.normalize(img_np, maxval=255)

        image_tensor = (
            torch.from_numpy(img_np)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )

        img_vis = (img_np / 1024.0 + 1.0) / 2.0
        img_vis = np.clip(img_vis, 0, 1)
        img_rgb = np.stack([img_vis] * 3, axis=-1)

    # ===============================
    # PREDICTION
    # ===============================
    if is_vit:
        with torch.no_grad():
            logits, attentions = model(image_tensor)

        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        all_probs = dict(zip(NIH_LABELS, probs.tolist()))

    else:
        pred = predict(model, image_tensor)
        all_probs = pred["all_probs"]
        attentions = None

    # ===============================
    # POSITIVE FINDINGS
    # ===============================
    positives = [
        {"disease": k, "probability": float(v)}
        for k, v in all_probs.items()
        if v >= 0.5
    ]

    if len(positives) == 0:
        return {
            "model": model_name,
            "positive_findings": [],
            "all_probabilities": all_probs,
            "note": "No confident abnormal findings detected."
        }

    # ===============================
    # TOP DISEASE
    # ===============================
    top_label, top_prob = max(all_probs.items(), key=lambda x: x[1])
    class_idx = NIH_LABELS.index(top_label)

    # ===============================
    # HEATMAP
    # ===============================
    if is_vit:
        heatmap, gray_cam = generate_vit_attention(
            attentions,
            img_rgb
        )
    else:
        heatmap = generate_heatmap(
            model,
            image_tensor,
            img_rgb,
            class_idx
        )
        gray_cam = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY) / 255.0

    cv2.imwrite(
        "outputs/heatmap.png",
        cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    )

    mask = segment_from_cam(gray_cam)
    cv2.imwrite("outputs/mask.png", mask)

    # ===============================
    # FINAL RESULT
    # ===============================
    return {
        "model": model_name,
        "positive_findings": positives,
        "all_probabilities": all_probs,
        "top_disease": {
            "label": top_label,
            "probability": float(top_prob)
        },
        "heatmap": "outputs/heatmap.png",
        "roi_mask": "outputs/mask.png",
        "note": "For clinical decision support only. Not a medical diagnosis."
    }
