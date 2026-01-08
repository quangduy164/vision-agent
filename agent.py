import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchxrayvision as xrv

from models.classifier import load_model, predict
from models.gradcam import generate_heatmap
from models.segmenter import segment_from_cam

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def crop_lung_region(img: Image.Image):
    """
    Crop bỏ cổ, chữ, mép trên/dưới
    (heuristic, KHÔNG phải segmentation)
    """
    w, h = img.size
    return img.crop((
        int(0.15 * w),
        int(0.18 * h),
        int(0.85 * w),
        int(0.92 * h)
    ))


def run(image_path, model_name="densenet121_nih"):
    os.makedirs("outputs", exist_ok=True)

    # ===============================
    # LOAD MODEL
    # ===============================
    model = load_model(model_name).to(DEVICE)
    model.eval()

    # ===============================
    # MODEL-SPECIFIC CONFIG
    # ===============================
    if "resnet50" in model_name:
        IMG_SIZE = 512
    else:
        IMG_SIZE = 224

    # ===============================
    # LOAD IMAGE (NIH: GRAYSCALE)
    # ===============================
    img = Image.open(image_path).convert("L")
    img = crop_lung_region(img)
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # PIL → numpy
    img_np = np.array(img).astype(np.float32)

    # ===============================
    # NIH NORMALIZATION (VERY IMPORTANT)
    # [0,255] → [-1024,1024]
    # ===============================
    img_np = xrv.datasets.normalize(img_np, maxval=255)

    # numpy → torch tensor [1,1,H,W]
    image_tensor = torch.from_numpy(img_np)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

    # ===============================
    # VIS IMAGE FOR GRAD-CAM
    # [-1024,1024] → [0,1]
    # ===============================
    img_vis = (img_np / 1024.0 + 1.0) / 2.0
    img_vis = np.clip(img_vis, 0, 1)
    img_rgb = np.stack([img_vis] * 3, axis=-1)

    # ===============================
    # PREDICTION
    # ===============================
    pred = predict(model, image_tensor)

    if len(pred["positive_findings"]) == 0:
        return {
            "model": model_name,
            "positive_findings": [],
            "all_probabilities": pred["all_probs"],
            "note": "No confident abnormal findings detected."
        }

    # ===============================
    # SELECT TOP DISEASE
    # ===============================
    top_label, top_prob = max(
        pred["all_probs"].items(),
        key=lambda x: x[1]
    )
    class_idx = list(pred["all_probs"].keys()).index(top_label)

    # ===============================
    # GRAD-CAM
    # ===============================
    heatmap = generate_heatmap(
        model,
        image_tensor,
        img_rgb,
        class_idx
    )

    heatmap_path = "outputs/heatmap.png"
    cv2.imwrite(
        heatmap_path,
        cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    )

    # ===============================
    # ROI MASK (OPTIONAL)
    # ===============================
    gray_cam = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY) / 255.0
    mask = segment_from_cam(gray_cam)

    mask_path = "outputs/mask.png"
    cv2.imwrite(mask_path, mask)

    return {
        "model": model_name,
        "positive_findings": pred["positive_findings"],
        "all_probabilities": pred["all_probs"],
        "top_disease": {
            "label": top_label,
            "probability": float(top_prob)
        },
        "heatmap": heatmap_path,
        "roi_mask": mask_path,
        "note": "For clinical decision support only. Not a medical diagnosis."
    }
