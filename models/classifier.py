# models/classifier.py
import torch
import torch.nn as nn
import torchxrayvision as xrv
from transformers import ViTForImageClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NIH ChestX-ray14 labels
NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]


# ===============================
# ViT NIH (HuggingFace pretrained)
# ===============================
class ViT_NIH(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = ViTForImageClassification.from_pretrained(
            "taheera/vit-in1k-chestxray14",
            output_attentions=True
        )

    def forward(self, x):
        outputs = self.model(
            pixel_values=x,
            output_attentions=True
        )
        return outputs.logits, outputs.attentions


# ===============================
# LOAD MODEL
# ===============================
def load_model(model_name: str):

    if model_name == "densenet121_nih":
        model = xrv.models.DenseNet(
            weights="densenet121-res224-all"
        )

    elif model_name == "resnet50_nih":
        model = xrv.models.ResNet(
            weights="resnet50-res512-all"
        )

    elif model_name == "vit_base_nih":
        model = ViT_NIH()

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model.to(DEVICE)
    model.eval()
    return model


# ===============================
# PREDICT (CNN ONLY)
# ===============================
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
