# models/bridge.py
import random

# TỪ ĐIỂN MẪU CÂU (IU X-RAY STYLE)
DISEASE_TEMPLATES = {
    "No Finding": [
        "The lungs are clear. Heart size is normal. No pneumothorax or effusion.",
        "No acute cardiopulmonary abnormality. Mediastinal contours are within normal limits.",
        "Clear chest. No focal consolidation, effusion, or pneumothorax."
    ],
    "Cardiomegaly": [
        "Cardiomegaly is present. The cardiac silhouette is enlarged.",
        "The heart is enlarged. Pulmonary vascularity is normal.",
        "Enlargement of the cardiac silhouette without evidence of failure."
    ],
    "Effusion": [
        "Blunting of the {side} costophrenic angle, consistent with {size} pleural effusion.",
        "There is a {size} pleural effusion on the {side}. No pneumothorax."
    ],
    "Pneumonia": [
        "Focal opacity in the {location}, suggestive of pneumonia.",
        "Airspace disease in the {location}. Clinical correlation recommended."
    ],
    "Infiltration": [
        "Patchy infiltrates noted in the {location}.",
        "Ill-defined opacities in the {location}, likely representing infiltration."
    ],
    "Atelectasis": [
        "Linear density in the {location}, consistent with atelectasis.",
        "Subsegmental atelectasis observed in the {location}."
    ],
    "Pneumothorax": [
        "Small pneumothorax seen in the {location}. No tension physiology.",
        "Visible pleural line in the {location}, suggestive of pneumothorax."
    ],
    "Nodule": [
        "A {size} nodule is noted in the {location}.",
        "Nodular opacity in the {location}. Comparison with prior films suggested."
    ],
    "Mass": [
        "A {size} mass-like density is observed in the {location}.",
        "Large opacity in the {location}, concerning for a mass."
    ],
    "Edema": [
        "Diffuse interstitial opacities bilaterally, consistent with pulmonary edema.",
        "Vascular congestion and indistinct markings, suggestive of mild edema."
    ],
    "Consolidation": [
        "Dense consolidation in the {location} with air bronchograms.",
        "Alveolar consolidation noted in the {location}."
    ],
    "Emphysema": [
        "Hyperinflated lungs with flattened diaphragms, consistent with emphysema.",
        "Signs of COPD with hyperlucency and flattened diaphragms."
    ],
    "Fibrosis": [
        "Reticular markings in the {location}, suggestive of fibrosis.",
        "Chronic scarring and fibrosis noted in the {location}."
    ],
    "Pleural_Thickening": [
        "Pleural thickening noted in the {location}.",
        "Apical pleural scarring/thickening observed."
    ],
    "Hernia": [
        "Hiatal hernia visible behind the cardiac silhouette.",
        "Gas-filled structure over the diaphragm, consistent with hernia."
    ]
}

def generate_prompt(diagnosis: str, confidence: float, location: str="chest", size: str="moderate", side: str="unspecified", threshold: float = 0.64) -> str:
    """
    Sinh Prompt "điền vào chỗ trống" dựa trên thông tin từ Grad-CAM.
    """
    
    # 1. Xử lý trường hợp Bình thường
    if diagnosis == "No Finding" or confidence < threshold:
        base_sentence = random.choice(DISEASE_TEMPLATES["No Finding"])
        return f"Chest X-ray Report:\nFindings: {base_sentence}"

    # 2. Xử lý trường hợp Bệnh
    # Lấy template gốc
    if diagnosis in DISEASE_TEMPLATES:
        templates = DISEASE_TEMPLATES[diagnosis]
        selected_template = random.choice(templates)
        
        # Điền thông tin vào chỗ trống {}
        # Dùng .format() an toàn (tránh lỗi nếu template ko có placeholder)
        filled_sentence = selected_template.replace("{location}", location)
        filled_sentence = filled_sentence.replace("{size}", size)
        filled_sentence = filled_sentence.replace("{side}", side)
        
        # Thêm câu mồi cho BioGPT sáng tạo tiếp
        prompt = f"Chest X-ray Findings:\n{filled_sentence} The surrounding osseous structures are intact. "
        return prompt
    
    # Fallback cho bệnh lạ
    return f"Chest X-ray Findings:\nAbnormalities observed in the {location}. suggestive of {diagnosis}."