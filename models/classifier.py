# models/classifier.py
import torch
import torchxrayvision as xrv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]


def load_model(model_name: str):
    if model_name == "densenet121_nih":
        model = xrv.models.DenseNet(
            weights="densenet121-res224-all"
        )

    elif model_name == "resnet50_nih":
        model = xrv.models.ResNet(
            weights="resnet50-res512-all"
        )

    else:
        raise ValueError("Unsupported model")

    model = model.to(DEVICE)
    model.eval()
    return model


def predict(model, image_tensor, threshold=0.5):
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)

    probs = probs.squeeze().cpu().numpy()

    positives = [
        {"disease": label, "probability": float(p)}
        for label, p in zip(NIH_LABELS, probs)
        if p >= threshold
    ]

    return {
        "positive_findings": positives,
        "all_probs": dict(zip(NIH_LABELS, probs.tolist()))
    }