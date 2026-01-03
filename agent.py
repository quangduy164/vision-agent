import cv2
import torch
import numpy as np
import os
from PIL import Image


from models.classifier import load_model, predict
from models.gradcam import generate_heatmap
from models.segmenter import segment_from_cam
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def run(image_path, model_name="densenet121"):
    os.makedirs("outputs", exist_ok=True)

    # Load model
    model = load_model(model_name).to(device)

    # Load image
    img_bgr = cv2.imread(image_path)
    # OpenCV → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # numpy → PIL Image (BẮT BUỘC)
    img_pil = Image.fromarray(img_rgb)

    # dùng cho Grad-CAM overlay
    img_norm = np.array(img_pil).astype("float32") / 255.0

    # FIX: resize về 224x224
    img_norm = cv2.resize(img_norm, (224, 224))

    # transform chuẩn torchvision
    image_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Prediction
    pred = predict(model, image_tensor)

    # Grad-CAM
    heatmap = generate_heatmap(model, image_tensor, img_norm)
    heatmap_path = "outputs/heatmap.png"
    cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

    # ROI mask
    gray_cam = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY) / 255.0
    mask = segment_from_cam(gray_cam)
    mask_path = "outputs/mask.png"
    cv2.imwrite(mask_path, mask)

    return {
        "confidence": pred["confidence"],
        "predicted_class": pred["predicted_class"],
        "heatmap": heatmap_path,
        "roi_mask": mask_path,
        "note": "For clinical decision support only. Not a diagnosis."
    }
