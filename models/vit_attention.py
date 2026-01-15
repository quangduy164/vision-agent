# models/vit_attention.py
import numpy as np
import cv2
import math


def generate_vit_attention(attentions, image_rgb):
    """
    attentions: tuple(num_layers) of (1, heads, N, N)
    image_rgb: [H,W,3] in [0,1]
    """
    if attentions is None:
        raise ValueError("ViT attentions is None. Did you enable output_attentions?")

    # Lấy attention layer cuối
    last_attn = attentions[-1]  # (1, heads, N, N)

    # Average heads
    attn = last_attn.mean(dim=1)[0]  # (N, N)

    # CLS token → patches
    cls_attn = attn[0, 1:]

    num_patches = cls_attn.shape[0]
    size = int(math.sqrt(num_patches))

    attn_map = cls_attn.reshape(size, size).cpu().numpy()

    attn_map = cv2.resize(
        attn_map,
        (image_rgb.shape[1], image_rgb.shape[0])
    )

    attn_map = (attn_map - attn_map.min()) / (
        attn_map.max() - attn_map.min() + 1e-8
    )

    heatmap = cv2.applyColorMap(
        np.uint8(255 * attn_map),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (
        0.6 * image_rgb * 255 + 0.4 * heatmap
    ).astype(np.uint8)

    return overlay, attn_map
