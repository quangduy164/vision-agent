# models/classifier.py
import torch
import torch.nn as nn
import torchxrayvision as xrv
from transformers import ViTModel, ViTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]


# ===============================
# ViT Wrapper (14-label NIH)
# ===============================
class ViT_NIH(nn.Module):
    def __init__(self):
        super().__init__()

        config = ViTConfig.from_pretrained(
            "google/vit-base-patch16-224",
            output_attentions=True   # ⭐ BẮT BUỘC
        )

        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224",
            config=config
        )

        self.classifier = nn.Linear(
            self.vit.config.hidden_size,
            len(NIH_LABELS)
        )

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)

        return logits, outputs.attentions

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
