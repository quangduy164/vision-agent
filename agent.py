# agent.py
import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchxrayvision as xrv

# Import các module vệ tinh
from models.classifier import load_model, predict
from models.gradcam import generate_heatmap
from models.segmenter import segment_from_cam, find_contours, get_location_text # <--- Đã thêm get_location_text
from models.bridge import generate_prompt
from models.decoder import BioGPTDecoder

# --- CẤU HÌNH ---
CONFIDENCE_THRESHOLD = 0.63 
MODEL_NAME = "densenet121_all"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MedicalVisionAgent:
    def __init__(self):
        print("\n" + "="*40)
        print(f"🤖 INITIALIZING AI SYSTEM ON {DEVICE}")
        print("="*40)
        
        # 1. Load Mắt (DenseNet)
        self.vision_model = load_model(MODEL_NAME)
        
        # 2. Load Miệng (BioGPT)
        self.language_decoder = BioGPTDecoder()
        
        print("\n✅ SYSTEM READY TO SERVE.\n")

    def analyze(self, image_path, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        
        # --- BƯỚC 1: XỬ LÝ ẢNH ---
        try:
            img_pil = Image.open(image_path).convert("L").resize((224, 224))
            img_np = np.array(img_pil)
            img_vis = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR) # Ảnh để vẽ
            
            # Chuẩn hóa cho Model
            img_norm = xrv.datasets.normalize(img_np, maxval=255)
            img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
        except Exception as e:
            return {"error": f"Lỗi đọc ảnh: {str(e)}"}

        # --- BƯỚC 2: CHẨN ĐOÁN (Vision) ---
        results = predict(self.vision_model, img_tensor)
        sorted_probs = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
        top_disease = list(sorted_probs.keys())[0]
        top_prob = sorted_probs[top_disease]

        # Xác định trạng thái
        if top_prob < CONFIDENCE_THRESHOLD:
            final_diagnosis = "No Finding"
            status = "Normal"
        else:
            final_diagnosis = top_disease
            status = "Abnormal"
        
        print(f"🔍 Analyzed {filename}: {final_diagnosis} ({top_prob:.2%})")

        # --- BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG THỊ GIÁC (Visual Feature Extraction) ---
        # Khởi tạo giá trị mặc định (cho trường hợp Normal)
        loc_text = "chest"
        size_text = "moderate"
        side_text = "unspecified"
        heatmap_vis = None
        mask = None

        if status == "Abnormal":
            # 1. Tìm layer bệnh
            target_idx = self.vision_model.pathologies.index(final_diagnosis)
            
            # 2. Chạy Grad-CAM (Chỉ chạy 1 lần duy nhất ở đây)
            heatmap_vis, gray_cam = generate_heatmap(
                self.vision_model, img_tensor, img_vis.astype(np.float32)/255.0, target_idx
            )
            
            # 3. Phân vùng (Segmentation)
            mask = segment_from_cam(gray_cam, threshold=gray_cam.max() * 0.6)
            
            # 4. Phân tích vị trí & kích thước (để nạp cho BioGPT)
            loc_text, size_text, side_text = get_location_text(mask, 224, 224)
            print(f"📍 AI Vision: Found {size_text} area in {loc_text}")

        # --- BƯỚC 4: TẠO PROMPT THÔNG MINH (Bridge) ---
        # Truyền toàn bộ thông tin thị giác vào cầu nối
        prompt = generate_prompt(
            diagnosis=final_diagnosis,
            confidence=top_prob,
            location=loc_text,
            size=size_text,
            side=side_text,
            threshold=CONFIDENCE_THRESHOLD
        )
        print(f"🌉 Bridge Prompt: {prompt[:80]}...")

        # --- BƯỚC 5: SINH BÁO CÁO CHI TIẾT (Language Generation) ---
        report_text = self.language_decoder.generate_report(prompt)
        print(f"📝 BioGPT Report: {report_text[:100]}...")

        # --- BƯỚC 6: VẼ VÀ LƯU ẢNH (Visualization) ---
        save_path = os.path.join(output_dir, f"result_{filename}")
        
        if status == "Abnormal" and mask is not None:
            # Vẽ đường bao vùng bệnh
            contours = find_contours(mask)
            cv2.drawContours(img_vis, contours, -1, (0, 255, 255), 2)
            
            # Ghi tên bệnh + %
            label = f"{final_diagnosis}: {top_prob*100:.1f}%"
            cv2.putText(img_vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Ghi thêm vị trí tìm được (Debug)
            loc_label = f"Loc: {loc_text}"
            cv2.putText(img_vis, loc_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            # Ghi nhãn bình thường
            cv2.putText(img_vis, "Normal / No Findings", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(save_path, img_vis)

        return {
            "image": filename,
            "status": status,
            "diagnosis": final_diagnosis,
            "confidence": float(top_prob),
            "visual_findings": {
                "location": loc_text,
                "size": size_text,
                "side": side_text
            },
            "report": report_text,
            "output_image": save_path,
            "all_probabilities": sorted_probs
        }

# --- TEST LOCAL ---
if __name__ == "__main__":
    agent = MedicalVisionAgent()
    # Thử một ảnh mẫu
    test_img = "uploads/00000001_000.png"
    if os.path.exists(test_img):
        agent.analyze(test_img)
    else:
        print("Vui lòng tải ảnh vào thư mục uploads/ để test.")