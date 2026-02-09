# models/segmenter.py
import cv2
import numpy as np

def segment_from_cam(cam_gray, threshold=0.5):
    """
    Chuyển Heatmap thành Mask nhị phân.
    """
    mask = (cam_gray > threshold).astype("uint8") * 255

    # Khử nhiễu
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def find_contours(mask):
    """
    Tìm đường viền để vẽ.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_location_text(mask, img_width=224, img_height=224):
    """
    Phân tích Mask để trả về text vị trí chuẩn y khoa (IU Style).
    """
    # Tìm vùng lớn nhất
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "diffuse area", "small", "unspecified"

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cx, cy = x + w/2, y + h/2

    # 1. Xác định Trái/Phải (Lưu ý: X-quang ngược với ảnh)
    # Ảnh: [0..224]. < 74: Phải (của ảnh) = Phổi Trái (BN)
    # Tuy nhiên để đơn giản hoá cho AI, ta cứ dùng Left/Right của ảnh, 
    # BioGPT thông minh sẽ tự hiểu nếu được prompt đúng context.
    # Nhưng chuẩn y khoa: Bên Phải ảnh là Phổi Trái bệnh nhân.
    
    side = ""
    if cx < img_width * 0.4:
        side = "right" # Phổi phải (bên trái ảnh)
    elif cx > img_width * 0.6:
        side = "left"  # Phổi trái (bên phải ảnh)
    else:
        side = "central/mediastinal"

    # 2. Xác định Trên/Giữa/Dưới
    vertical = ""
    if cy < img_height * 0.35:
        vertical = "upper zone" # Đỉnh phổi
    elif cy > img_height * 0.65:
        vertical = "lower zone/base" # Đáy phổi
    else:
        vertical = "mid-lung zone"

    location_str = f"{side} lung {vertical}" if "lung" not in side else f"{side} {vertical}"
    
    if side == "central/mediastinal":
        location_str = "mediastinum"

    # 3. Xác định kích thước
    area = cv2.contourArea(c)
    total_area = img_width * img_height
    ratio = area / total_area

    size_str = "small"
    if ratio > 0.15: size_str = "large/extensive"
    elif ratio > 0.05: size_str = "moderate"
    else: size_str = "focal"

    return location_str, size_str, side