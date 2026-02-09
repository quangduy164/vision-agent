# models/classifier.py
import torch
import torchxrayvision as xrv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_name: str):
    """
    Load model pre-trained từ torchxrayvision.
    """
    print(f"🔄 Loading Vision Model: {model_name} on {DEVICE}...")
    
    if model_name == "densenet121_all":
        # Model tổng hợp (NIH, CheXpert, v.v.)
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
    elif model_name == "densenet121_nih":
        # Model chuyên biệt cho NIH
        model = xrv.models.DenseNet(weights="densenet121-res224-nih")
    elif model_name == "resnet50_nih":
        model = xrv.models.ResNet(weights="resnet50-res512-all")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model.to(DEVICE)
    model.eval() 
    print("✅ Vision Model loaded successfully.")
    return model

def predict(model, image_tensor):
    """
    Dự đoán bệnh từ tensor ảnh.
    """
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)

    probs = probs.squeeze().cpu().numpy()
    labels = model.pathologies
    
    # Kết quả: { "Pneumonia": 0.85, ... }
    results = dict(zip(labels, probs.tolist()))
    return results