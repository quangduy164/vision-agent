import torch
from torchvision import models

NUM_CLASSES = 3  # ví dụ: normal / pneumonia / covid

def load_model(model_name: str):
    if model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        model.classifier = torch.nn.Linear(
            model.classifier.in_features,
            NUM_CLASSES
        )

    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features,
            NUM_CLASSES
        )

    else:
        raise ValueError("Unsupported model")

    model.eval()
    return model


def predict(model, image_tensor):
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, cls = probs.max(dim=1)

    return {
        "predicted_class": int(cls.item()),
        "confidence": float(confidence.item()),
        "all_probs": probs.squeeze().tolist()
    }
