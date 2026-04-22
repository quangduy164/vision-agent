# models/bridge.py
import random

# Template theo style IU X-Ray ground truth - đầy đủ findings + impression
DISEASE_TEMPLATES = {
    "No Finding": [
        "The lungs are clear. The heart size is normal. The mediastinal contours are unremarkable. No pleural effusion or pneumothorax is identified. The osseous structures are intact.",
        "No acute cardiopulmonary abnormality. The lungs are well expanded and clear. Heart size and mediastinal contours are within normal limits. No pleural effusion or pneumothorax.",
        "Clear lungs bilaterally. Normal cardiac silhouette. No focal consolidation, pleural effusion, or pneumothorax. Bony thorax is intact.",
    ],
    "Cardiomegaly": [
        "The cardiac silhouette is enlarged. The lungs are clear without focal consolidation. No pleural effusion or pneumothorax. The mediastinal contours are stable. Osseous structures are intact.",
        "Cardiomegaly is present. Pulmonary vascularity is within normal limits. No acute airspace disease. No pleural effusion identified. The bony thorax is unremarkable.",
        "Enlarged cardiac silhouette consistent with cardiomegaly. The lungs are clear. No evidence of pulmonary edema. No pneumothorax or pleural effusion.",
    ],
    "Effusion": [
        "There is a {size} pleural effusion on the {side}. The {side} costophrenic angle is blunted. The lungs are otherwise clear. No pneumothorax. Heart size is normal.",
        "Blunting of the {side} costophrenic angle consistent with {size} pleural effusion. No focal consolidation. Cardiac silhouette is within normal limits.",
        "{size} pleural effusion is noted on the {side}. There is associated compressive atelectasis. No pneumothorax identified.",
    ],
    "Pneumonia": [
        "There is a focal airspace opacity in the {location} consistent with pneumonia. No pleural effusion. Heart size is normal. The remaining lung fields are clear.",
        "Focal consolidation in the {location} suggestive of pneumonia or airspace disease. Clinical correlation is recommended. No pneumothorax or large effusion.",
        "Airspace disease in the {location}, most likely representing pneumonia. The cardiac silhouette is normal in size. No pleural effusion.",
    ],
    "Infiltration": [
        "Patchy airspace opacities are noted in the {location}, consistent with infiltration. No pleural effusion or pneumothorax. Heart size is normal.",
        "Ill-defined opacities in the {location} likely representing infiltrates. The cardiac silhouette is normal. No acute osseous abnormality.",
        "Bilateral patchy infiltrates are present, most prominent in the {location}. No pneumothorax. Heart size is stable.",
    ],
    "Atelectasis": [
        "Linear atelectasis is noted in the {location}. No focal consolidation or pleural effusion. Heart size is normal. No pneumothorax.",
        "Subsegmental atelectasis in the {location}. The lungs are otherwise clear. Cardiac silhouette is within normal limits.",
        "Plate-like atelectasis in the {location}. No pneumothorax or pleural effusion identified. Heart size is normal.",
    ],
    "Pneumothorax": [
        "There is a pneumothorax on the {side}. A visible pleural line is noted in the {location}. No tension physiology. The contralateral lung is clear.",
        "Small pneumothorax identified in the {location}. No mediastinal shift. Heart size is normal. No pleural effusion on the contralateral side.",
        "Pneumothorax is present in the {location}. The lung is partially collapsed. Clinical correlation and follow-up recommended.",
    ],
    "Nodule": [
        "A {size} pulmonary nodule is noted in the {location}. No other focal consolidation. Heart size is normal. Comparison with prior imaging is recommended.",
        "Nodular opacity in the {location}. No pleural effusion or pneumothorax. The cardiac silhouette is normal. Follow-up CT may be warranted.",
        "A {size} nodule is identified in the {location}. The remaining lung fields are clear. No acute cardiopulmonary abnormality.",
    ],
    "Mass": [
        "A {size} mass-like opacity is identified in the {location}. Further evaluation with CT is recommended. No pleural effusion. Heart size is normal.",
        "Large opacity in the {location} concerning for a mass lesion. No pneumothorax. Cardiac silhouette is within normal limits. CT correlation advised.",
        "A {size} mass is noted in the {location}. The mediastinum is not widened. No pleural effusion or pneumothorax identified.",
    ],
    "Edema": [
        "Diffuse bilateral interstitial opacities consistent with pulmonary edema. The cardiac silhouette is enlarged. Vascular congestion is present. No pneumothorax.",
        "Pulmonary vascular congestion and perihilar haziness consistent with pulmonary edema. Bilateral pleural effusions may be present. Cardiomegaly noted.",
        "Interstitial edema pattern bilaterally. Indistinct pulmonary vasculature. The heart is enlarged. No pneumothorax identified.",
    ],
    "Consolidation": [
        "Dense consolidation with air bronchograms in the {location}. No pleural effusion. Heart size is normal. The remaining lung fields are clear.",
        "Alveolar consolidation noted in the {location}. No pneumothorax or large effusion. Cardiac silhouette is within normal limits.",
        "Lobar consolidation in the {location} consistent with pneumonia or aspiration. No pleural effusion. Heart size is stable.",
    ],
    "Emphysema": [
        "Hyperinflated lungs with flattened diaphragms consistent with emphysema. Increased AP diameter. No focal consolidation or pleural effusion. Heart size is normal.",
        "Signs of chronic obstructive pulmonary disease with hyperlucency and flattened hemidiaphragms. No acute airspace disease. No pneumothorax.",
        "Emphysematous changes bilaterally. Barrel-shaped chest. No focal consolidation. Cardiac silhouette is within normal limits.",
    ],
    "Fibrosis": [
        "Reticular interstitial markings in the {location} suggestive of pulmonary fibrosis. No acute airspace disease. Heart size is normal.",
        "Chronic interstitial changes and fibrosis noted in the {location}. No pleural effusion or pneumothorax. Cardiac silhouette is stable.",
        "Bilateral reticular opacities consistent with fibrosis. Honeycombing may be present. No acute cardiopulmonary process identified.",
    ],
    "Pleural_Thickening": [
        "Pleural thickening is noted in the {location}. No acute airspace disease. Heart size is normal. No pneumothorax or effusion.",
        "Apical pleural thickening and scarring observed. The lungs are otherwise clear. Cardiac silhouette is within normal limits.",
        "Chronic pleural thickening in the {location}. No new focal consolidation. No pleural effusion. Heart size is stable.",
    ],
    "Hernia": [
        "A hiatal hernia is visible behind the cardiac silhouette. The lungs are clear. No pleural effusion or pneumothorax. Heart size is normal.",
        "Gas-filled structure projecting over the diaphragm consistent with hiatal hernia. No acute cardiopulmonary abnormality.",
        "Hiatal hernia noted. The lungs are clear bilaterally. Cardiac silhouette is within normal limits. No pleural effusion.",
    ],
    "Fracture": [
        "Rib fracture is noted in the {location}. No pneumothorax identified. The lungs are clear. Heart size is normal.",
        "Acute rib fracture in the {location}. No associated pneumothorax or hemothorax. Cardiac silhouette is within normal limits.",
        "Fracture of the {location} ribs. The lungs are otherwise clear. No pleural effusion or pneumothorax.",
    ],
    "Lung Opacity": [
        "Airspace opacity in the {location}. Differential includes pneumonia, atelectasis, or aspiration. No pleural effusion. Heart size is normal.",
        "Focal lung opacity noted in the {location}. Clinical correlation recommended. No pneumothorax. Cardiac silhouette is stable.",
        "Opacity in the {location} of uncertain etiology. No large pleural effusion. Heart size is within normal limits.",
    ],
}

# Câu impression theo từng bệnh
IMPRESSION_TEMPLATES = {
    "No Finding":          "No acute cardiopulmonary process.",
    "Cardiomegaly":        "Cardiomegaly. No acute pulmonary process.",
    "Effusion":            "{size} pleural effusion on the {side}.",
    "Pneumonia":           "Airspace disease in the {location}, pneumonia not excluded.",
    "Infiltration":        "Pulmonary infiltrates in the {location}.",
    "Atelectasis":         "Atelectasis in the {location}.",
    "Pneumothorax":        "Pneumothorax on the {side}. Clinical correlation advised.",
    "Nodule":              "Pulmonary nodule in the {location}. Follow-up recommended.",
    "Mass":                "Mass lesion in the {location}. CT recommended.",
    "Edema":               "Pulmonary edema with cardiomegaly.",
    "Consolidation":       "Consolidation in the {location}.",
    "Emphysema":           "Emphysema / COPD changes.",
    "Fibrosis":            "Pulmonary fibrosis in the {location}.",
    "Pleural_Thickening":  "Pleural thickening in the {location}.",
    "Hernia":              "Hiatal hernia.",
    "Fracture":            "Rib fracture in the {location}.",
    "Lung Opacity":        "Lung opacity in the {location}. Clinical correlation recommended.",
}


def _fill(template: str, location: str, size: str, side: str) -> str:
    return (template
            .replace("{location}", location)
            .replace("{size}", size)
            .replace("{side}", side))


def generate_prompt(
    diagnosis: str,
    confidence: float,
    location: str = "chest",
    size: str = "moderate",
    side: str = "unspecified",
    threshold: float = 0.45,
) -> str:
    """
    Sinh report hoàn chỉnh theo style IU X-Ray (Findings + Impression).
    BioGPT sẽ dùng đây làm prefix để tiếp tục sinh hoặc trả về thẳng.
    """
    key = diagnosis if diagnosis in DISEASE_TEMPLATES else "No Finding"

    findings = _fill(random.choice(DISEASE_TEMPLATES[key]), location, size, side)
    impression = _fill(IMPRESSION_TEMPLATES.get(key, "No acute cardiopulmonary process."), location, size, side)

    prompt = (
        f"Findings: {findings} "
        f"Impression: {impression}"
    )
    return prompt
