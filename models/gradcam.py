from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np


def get_target_layer(model):
    """
    TorchXRayVision wraps torchvision backbone inside model.model
    """
    backbone = model.model

    # DenseNet
    if hasattr(backbone, "features"):
        return backbone.features.denseblock4

    # ResNet
    if hasattr(backbone, "layer4"):
        return backbone.layer4[-1]

    raise ValueError("Unsupported model architecture for Grad-CAM")


def generate_heatmap(model, image_tensor, image_np, class_idx):
    target_layer = get_target_layer(model)

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    targets = [ClassifierOutputTarget(class_idx)]

    grayscale_cam = cam(
        input_tensor=image_tensor,
        targets=targets
    )[0]

    heatmap = show_cam_on_image(
        image_np.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    # VERY IMPORTANT: cleanup hooks (FastAPI ổn định)
    cam.activations_and_grads.release()

    return heatmap
