# models/gradcam.py
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import torch
import cv2

def get_target_layer(model):
    """
    Lấy layer cuối của DenseNet121 để soi.
    """
    if hasattr(model, "features"):
        return model.features.denseblock4[-1]
    raise ValueError("Không tìm thấy layer features (Chỉ hỗ trợ DenseNet/ResNet)")

def generate_heatmap(model, image_tensor, image_origin_np, target_category_idx=None):
    # 1. Xác định Layer
    target_layers = [get_target_layer(model)]

    # 2. Khởi tạo GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # 3. Xác định mục tiêu
    targets = None
    if target_category_idx is not None:
        targets = [ClassifierOutputTarget(target_category_idx)]

    # 4. Chạy Grad-CAM
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]

    # 5. Phủ màu
    visualization = show_cam_on_image(image_origin_np, grayscale_cam, use_rgb=True)
    
    return visualization, grayscale_cam