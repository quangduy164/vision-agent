# models/classifier.py
import torch
import torch.nn as nn
import torchxrayvision as xrv
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Đưa danh sách 17 nhãn CHUẨN từ Kaggle về đây
CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Fracture', 'Hernia', 'Infiltration', 
    'Lung Opacity', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 
    'Pneumonia', 'Pneumothorax'
]

def load_model(model_path: str = "best_multilabel_xrv.pth"):
    """
    Load custom model đã Fine-tune đa nhãn từ Kaggle.
    """
    print(f"🔄 Loading Custom Vision Model on {DEVICE}...")
    
    # Khởi tạo bộ khung (backbone)
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    
    # 🌟 "Ghép não": Xây lại lớp classifier y hệt như lúc Train trên Kaggle
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 512),
        nn.BatchNorm1d(512), 
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, len(CLASSES))
    )
    
    # Kiểm tra xem file .pth đã được copy vào đúng chỗ chưa
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file '{model_path}'. Hãy đảm bảo bạn đã copy file tải từ Kaggle vào cùng thư mục chạy code!")
        
    # Nạp trọng số (trí nhớ) vào mô hình
    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    model.op_threshs = None
    
    model = model.to(DEVICE)
    model.eval() 
    print("✅ Custom Vision Model loaded successfully.")
    return model

def predict(model, image_tensor):
    """
    Dự đoán bệnh từ tensor ảnh.
    Trả về Dictionary chứa xác suất của 17 bệnh.
    """
    with torch.no_grad():
        # Đảm bảo tensor ảnh ở đúng thiết bị (CPU/GPU)
        image_tensor = image_tensor.to(DEVICE)
        
        # Chạy suy luận
        logits = model(image_tensor)
        
        # 🌟 Bắt buộc phải dùng Sigmoid để ép điểm số về khoảng [0, 1]
        probs = torch.sigmoid(logits)

    probs = probs.squeeze().cpu().numpy()
    
    # Ghép tên bệnh với xác suất tương ứng
    # Kết quả: { "Pneumonia": 0.85, "Edema": 0.12, ... }
    results = dict(zip(CLASSES, probs.tolist()))
    return results