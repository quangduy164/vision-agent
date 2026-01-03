import torch

def segment_from_cam(cam):
    # cam: [H, W] trong [0,1]
    mask = (cam > 0.5).astype("uint8") * 255
    return mask
