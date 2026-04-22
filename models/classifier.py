# models/classifier.py
import torch
import torch.nn as nn
import torchxrayvision as xrv
import numpy as np
import os
from torchvision import transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Fracture', 'Hernia', 'Infiltration',
    'Lung Opacity', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening',
    'Pneumonia', 'Pneumothorax'
]

# Ngưỡng chẩn đoán tối ưu - lấy trực tiếp từ kết quả tìm ngưỡng trên tập Validation
BEST_THRESHOLDS = {
    'Atelectasis':        0.47,
    'Cardiomegaly':       0.49,
    'Consolidation':      0.15,  # ép cứng - bệnh hiếm
    'Edema':              0.54,
    'Effusion':           0.53,
    'Emphysema':          0.58,
    'Fibrosis':           0.15,  # ép cứng - bệnh hiếm
    'Fracture':           0.61,
    'Hernia':             0.59,
    'Infiltration':       0.61,
    'Lung Opacity':       0.45,
    'Mass':               0.61,
    'No Finding':         0.36,
    'Nodule':             0.54,
    'Pleural_Thickening': 0.56,
    'Pneumonia':          0.61,
    'Pneumothorax':       0.54,
}

# Trọng số ensemble theo từng bệnh (DenseNet, ResNet)
# Bệnh hiếm giao nhiều quyền hơn cho ResNet50
_WEIGHT_DENSE = np.full(len(CLASSES), 0.6)
_WEIGHT_RES   = np.full(len(CLASSES), 0.4)
for _c in ['Fibrosis', 'Consolidation']:
    if _c in CLASSES:
        _i = CLASSES.index(_c)
        _WEIGHT_DENSE[_i] = 0.1
        _WEIGHT_RES[_i]   = 0.9


def _build_densenet(num_classes: int):
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model


def load_model(model_path: str = "best_multilabel_xrv.pth"):
    """Load DenseNet đơn lẻ (backward-compat). Trả về model."""
    print(f"🔄 Loading DenseNet Vision Model on {DEVICE}...")
    model = _build_densenet(len(CLASSES))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Không tìm thấy '{model_path}'.")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    model.op_threshs = None
    model = model.to(DEVICE).eval()
    print("✅ DenseNet loaded.")
    return model


def load_ensemble_models(densenet_path: str = "best_multilabel_xrv.pth"):
    """
    Load cả DenseNet (224) lẫn ResNet50 (512).
    Trả về (model_dense, model_res, class_mapping_res).
    """
    # --- DenseNet ---
    print(f"🏗️ Loading DenseNet121 (finetuned 224x224) on {DEVICE}...")
    model_dense = _build_densenet(len(CLASSES))
    if not os.path.exists(densenet_path):
        raise FileNotFoundError(f"❌ Không tìm thấy '{densenet_path}'.")
    model_dense.load_state_dict(torch.load(densenet_path, map_location=DEVICE), strict=False)
    model_dense.op_threshs = None
    model_dense = model_dense.to(DEVICE).eval()
    print("✅ DenseNet121 loaded.")

    # --- ResNet50 ---
    print(f"🏗️ Loading ResNet50 (pretrained 512x512) on {DEVICE}...")
    model_res = xrv.models.ResNet(weights="resnet50-res512-all")
    model_res.op_threshs = None
    model_res = model_res.to(DEVICE).eval()
    print("✅ ResNet50 loaded.")

    # Ánh xạ CLASSES -> index trong ResNet pathologies
    xrv_pathologies = model_res.pathologies
    class_mapping_res = [
        xrv_pathologies.index(c) if c in xrv_pathologies else -1
        for c in CLASSES
    ]

    return model_dense, model_res, class_mapping_res


# --- Transform riêng cho từng model ---
_transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(lambda x: x * 1024),
])

_transform_512 = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(lambda x: x * 1024),
])


def _pil_to_tensor(img_pil: Image.Image, transform) -> torch.Tensor:
    """Chuyển PIL grayscale -> tensor [1, 1, H, W]."""
    return transform(img_pil).unsqueeze(0).to(DEVICE)


def predict(model, image_tensor) -> dict:
    """Dự đoán với DenseNet đơn lẻ. Trả về {disease: prob}."""
    with torch.no_grad():
        probs = torch.sigmoid(model(image_tensor.to(DEVICE)))
    probs = probs.squeeze().cpu().numpy()
    return dict(zip(CLASSES, probs.tolist()))


def predict_ensemble(
    model_dense, model_res, class_mapping_res, img_pil: Image.Image
) -> dict:
    """
    Ensemble DenseNet (224) + ResNet50 (512) với trọng số động theo từng bệnh.
    Áp dụng ngưỡng tối ưu từ BEST_THRESHOLDS.
    Trả về {disease: prob} với xác suất đã được ensemble.
    """
    t224 = _pil_to_tensor(img_pil, _transform_224)  # [1,1,224,224]
    t512 = _pil_to_tensor(img_pil, _transform_512)  # [1,1,512,512]

    with torch.no_grad():
        # Nhánh DenseNet
        probs_dense = torch.sigmoid(model_dense(t224)).squeeze().cpu().numpy()  # [17]

        # Nhánh ResNet
        out_res = torch.sigmoid(model_res(t512)).squeeze().cpu().numpy()        # [xrv_classes]
        probs_res = np.zeros(len(CLASSES))
        for i, xrv_idx in enumerate(class_mapping_res):
            if xrv_idx != -1:
                probs_res[i] = out_res[xrv_idx]
            elif CLASSES[i] == 'No Finding':
                # ResNet không có "No Finding" -> ước lượng ngược
                probs_res[i] = 1.0 - float(np.max(out_res))

    # Gộp theo trọng số động
    ensemble_probs = probs_dense * _WEIGHT_DENSE + probs_res * _WEIGHT_RES  # [17]

    return dict(zip(CLASSES, ensemble_probs.tolist()))


def get_threshold(disease: str) -> float:
    return BEST_THRESHOLDS.get(disease, 0.45)
