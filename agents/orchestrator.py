# agents/orchestrator.py
"""
Orchestrator - Điều phối luồng đa tác nhân:
  Vision Agent → Explanation Agent → Safety Agent

Expose interface analyze() tương thích với app.py hiện tại.
"""
import os
import logging
import uuid
import cv2
import numpy as np
from PIL import Image

import config
from models.classifier import load_ensemble_models
from agents.tools import init_models
from agents.vision_agent import VisionAgent
from agents.explanation_agent import ExplanationAgent
from agents.safety_agent import SafetyAgent
from models.segmenter import find_contours
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)

MODEL_NAME = config.MODEL_PATH


def _build_llm():
    """Khởi tạo LLM từ config. Trả về None nếu không cấu hình."""
    provider   = config.LLM_PROVIDER.lower()
    model_name = config.LLM_MODEL_NAME
    api_key    = config.LLM_API_KEY

    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name or "gpt-4o-mini", api_key=api_key, temperature=0)

        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_name or "gemini-1.5-flash",
                                          google_api_key=api_key, temperature=0)

        if provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model=model_name or "llama3", temperature=0)

    except Exception as e:
        logger.warning(f"Orchestrator: LLM init failed ({e}). Using fallback pipeline.")

    logger.warning("Orchestrator: LLM_PROVIDER not set or invalid. Using fallback pipeline.")
    return None


class MedicalAgentOrchestrator:
    """
    Hệ thống đa tác nhân phân tích ảnh X-quang.
    Interface analyze() tương thích với MedicalVisionAgent cũ.
    """

    def __init__(self):
        session_id = str(uuid.uuid4())[:8]
        logger.info(f"[{session_id}] Initializing Multi-Agent System...")

        # 1. Load models một lần duy nhất
        model_dense, model_res, class_mapping_res = load_ensemble_models(MODEL_NAME)

        # 2. Inject vào Vision Agent tools (shared state)
        init_models(model_dense, model_res, class_mapping_res)
        self._model_dense = model_dense  # giữ ref để vẽ ảnh

        # 3. Khởi tạo LLM (optional)
        llm = _build_llm()

        # 4. Khởi tạo 3 agents
        self.vision_agent      = VisionAgent(llm=llm)
        self.explanation_agent = ExplanationAgent()
        self.safety_agent      = SafetyAgent()

        logger.info(f"[{session_id}] Multi-Agent System ready.")

    def analyze(self, image_path: str, output_dir: str = "outputs", lang: str = "en") -> dict:
        """
        Phân tích ảnh X-quang qua 3 agents.
        Trả về dict tương thích với interface cũ + thêm key 'safety_reviewed'.
        """
        session_id = str(uuid.uuid4())[:8]
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)

        logger.info(f"[{session_id}] START analyze: {filename}")

        # ── AGENT 1: Vision ──────────────────────────────────────────────
        try:
            vision_out = self.vision_agent.run(image_path, output_dir)
        except Exception as e:
            logger.error(f"[{session_id}] VisionAgent failed: {e}", exc_info=True)
            return {"error": f"Vision analysis failed: {str(e)}"}

        if "error" in vision_out:
            return vision_out

        top_disease = vision_out.get("top_disease", "No Finding")
        top_prob    = vision_out.get("confidence", 0.0)
        status      = "Normal" if top_disease == "No Finding" else "Abnormal"
        visual_findings = vision_out.get("visual_findings", {})
        heatmap_path    = vision_out.get("heatmap_path")

        logger.info(f"[{session_id}] VisionAgent done: {top_disease} ({top_prob:.2%})")

        # ── AGENT 2: Explanation ─────────────────────────────────────────
        try:
            explanation_out = self.explanation_agent.run(vision_out, lang=lang)
        except Exception as e:
            logger.error(f"[{session_id}] ExplanationAgent failed: {e}", exc_info=True)
            explanation_out = {"report": "Report generation failed. Please consult a physician."}

        logger.info(f"[{session_id}] ExplanationAgent done.")

        # ── AGENT 3: Safety ──────────────────────────────────────────────
        try:
            safety_out = self.safety_agent.run(explanation_out, lang=lang)
        except Exception as e:
            logger.error(f"[{session_id}] SafetyAgent failed: {e}", exc_info=True)
            safety_out = {
                "report": explanation_out.get("report", ""),
                "safety_reviewed": False,
                "violations_found": [],
            }

        logger.info(f"[{session_id}] SafetyAgent done. Violations: {safety_out.get('violations_found')}")

        # ── Vẽ và lưu ảnh kết quả ────────────────────────────────────────
        save_path = os.path.join(output_dir, f"result_{filename}")
        try:
            self._draw_result(image_path, save_path, top_disease, top_prob,
                              status, visual_findings)
        except Exception as e:
            logger.warning(f"[{session_id}] Draw result failed: {e}")
            save_path = heatmap_path or image_path

        logger.info(f"[{session_id}] END analyze: {filename}")

        return {
            "image":            filename,
            "status":           status,
            "diagnosis":        top_disease,
            "confidence":       float(top_prob),
            "visual_findings":  visual_findings,
            "report":           safety_out["report"],
            "output_image":     save_path,
            "heatmap_path":     heatmap_path,
            "all_probabilities": vision_out.get("all_probabilities", {}),
            "safety_reviewed":  safety_out.get("safety_reviewed", False),
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    def _draw_result(self, image_path, save_path, disease, prob, status, visual_findings):
        import torch
        from models.gradcam import generate_heatmap
        from models.segmenter import segment_from_cam
        from models.classifier import _transform_224, CLASSES

        img_pil = Image.open(image_path).convert("L")
        img_pil_224 = img_pil.resize((224, 224))
        img_np  = np.array(img_pil_224)
        img_vis = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        if status == "Abnormal" and disease in CLASSES:
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            img_tensor = _transform_224(img_pil).unsqueeze(0).to(DEVICE)
            
            # RGB [0,1] cho show_cam_on_image
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            
            # Index theo CLASSES (output của model fine-tuned)
            target_idx = CLASSES.index(disease)
            
            _, gray_cam = generate_heatmap(
                self._model_dense, img_tensor, img_rgb, target_idx
            )

            _, gray_cam = generate_heatmap(
                self._model_dense, img_tensor, img_rgb, target_idx
            )
            mask     = segment_from_cam(gray_cam, threshold=gray_cam.max() * 0.6)
            contours = find_contours(mask)
            cv2.drawContours(img_vis, contours, -1, (0, 255, 255), 2)

            label = f"{disease}: {prob*100:.1f}%"
            cv2.putText(img_vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            loc = visual_findings.get("location", "")
            cv2.putText(img_vis, f"Loc: {loc}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(img_vis, "Normal / No Findings", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(save_path, img_vis)
