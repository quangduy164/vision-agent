# agents/vision_agent.py
"""
Vision Agent - Điều phối 3 tools thị giác qua ReAct.
Fallback về pipeline cứng nếu LLM không khả dụng.
"""
import json
import logging
from langchain_core.prompts import PromptTemplate

from models.classifier import get_threshold
from agents.tools import classify_xray, generate_gradcam, segment_lesion

# AgentExecutor đã bị bỏ trong langchain >= 1.x, dùng langgraph thay thế
try:
    from langgraph.prebuilt import create_react_agent as _create_react_agent
    _LANGGRAPH = True
except ImportError:
    _LANGGRAPH = False

logger = logging.getLogger(__name__)


class VisionAgent:
    def __init__(self, llm=None):
        self._tools = [classify_xray, generate_gradcam, segment_lesion]
        self._agent = None

        if llm is not None and _LANGGRAPH:
            try:
                self._agent = _create_react_agent(llm, self._tools)
                logger.info("VisionAgent: langgraph ReAct agent ready.")
            except Exception as e:
                logger.warning(f"VisionAgent: agent init failed ({e}). Will use fallback.")
        elif llm is not None:
            logger.warning("VisionAgent: langgraph not installed. Using fallback pipeline.")

    def run(self, image_path: str, output_dir: str = "outputs") -> dict:
        if self._agent:
            return self._run_with_llm(image_path, output_dir)
        return self._run_fallback(image_path, output_dir)

    def _run_with_llm(self, image_path: str, output_dir: str) -> dict:
        try:
            result = self._agent.invoke({
                "messages": [("human",
                    f"Analyze this chest X-ray: {image_path} (output_dir={output_dir}). "
                    "Step 1: classify_xray. "
                    "Step 2: if not No Finding, call segment_lesion. "
                    "Step 3: if not No Finding, call generate_gradcam. "
                    "Return JSON with: top_disease, confidence, diagnoses, visual_findings, heatmap_path."
                )]
            })
            # Lấy nội dung message cuối
            last = result["messages"][-1].content
            s, e = last.find("{"), last.rfind("}") + 1
            if s != -1 and e > s:
                return json.loads(last[s:e])
        except Exception as ex:
            logger.warning(f"VisionAgent LLM run failed ({ex}), falling back.")
        return self._run_fallback(image_path, output_dir)

    def _run_fallback(self, image_path: str, output_dir: str) -> dict:
        logger.info("VisionAgent: fallback pipeline.")

        cls = json.loads(classify_xray.invoke({"image_path": image_path}))
        if "error" in cls:
            return cls

        top_disease = cls["top_disease"]
        top_prob    = cls["top_prob"]
        diagnoses   = cls["diagnoses"]
        visual      = {"location": "chest", "size": "moderate", "side": "unspecified"}
        heatmap     = None

        if top_disease != "No Finding" and top_prob >= get_threshold(top_disease):
            seg = json.loads(segment_lesion.invoke({
                "image_path": image_path, "disease_name": top_disease
            }))
            if "error" not in seg:
                visual = seg

            cam = json.loads(generate_gradcam.invoke({
                "image_path": image_path,
                "disease_name": top_disease,
                "output_dir": output_dir,
            }))
            if "error" in cam:
                logger.warning(f"VisionAgent: GradCAM failed for {top_disease}: {cam['error']}")
            heatmap = cam.get("heatmap_path")

        return {
            "top_disease":     top_disease,
            "confidence":      top_prob,
            "diagnoses":       diagnoses,
            "visual_findings": visual,
            "heatmap_path":    heatmap,
            "all_probabilities": cls.get("all_probabilities", {}),
        }
