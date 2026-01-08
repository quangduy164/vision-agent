import cv2
import numpy as np


def segment_from_cam(cam_gray, threshold=0.6):
    """
    cam_gray: [0,1]
    """
    mask = (cam_gray > threshold).astype("uint8") * 255

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask
