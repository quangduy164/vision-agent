# agents/tools/report_tool.py
"""
LangChain Tool: generate_report
Sinh báo cáo y tế (Findings + Impression) từ kết quả chẩn đoán.
"""
import json
import logging
from langchain.tools import tool

from models.bridge import generate_prompt
from models.decoder import BioGPTDecoder

logger = logging.getLogger(__name__)

_decoder: BioGPTDecoder | None = None

CLINICAL_DISCLAIMER = "Clinical correlation and physician review are recommended."


def init_decoder(decoder: BioGPTDecoder):
    """Inject decoder instance từ ExplanationAgent (load 1 lần)."""
    global _decoder
    _decoder = decoder


@tool
def generate_report(
    diagnosis: str,
    confidence: float,
    location: str = "chest",
    size: str = "moderate",
    side: str = "unspecified",
) -> str:
    """
    Generate a structured radiology report (Findings + Impression) for a chest X-ray.
    Input:
      - diagnosis: disease name (e.g. 'Cardiomegaly') or 'No Finding'
      - confidence: probability score (float 0-1)
      - location: anatomical location from segment_lesion (default: 'chest')
      - size: lesion size from segment_lesion (default: 'moderate')
      - side: laterality from segment_lesion (default: 'unspecified')
    Output: JSON string with key 'report' containing the full radiology report text.
    Call this tool LAST, after classify_xray and segment_lesion.
    """
    try:
        prompt = generate_prompt(
            diagnosis=diagnosis,
            confidence=confidence,
            location=location,
            size=size,
            side=side,
        )
        report = _decoder.generate_report(prompt) if _decoder else prompt

        if CLINICAL_DISCLAIMER not in report:
            report = f"{report} {CLINICAL_DISCLAIMER}"

        logger.info(f"[generate_report] report generated for '{diagnosis}'.")
        return json.dumps({"report": report})

    except Exception as e:
        logger.error(f"[generate_report] {e}", exc_info=True)
        fallback = (
            f"Findings: Imaging findings noted. "
            f"Impression: Further evaluation recommended. "
            f"{CLINICAL_DISCLAIMER}"
        )
        return json.dumps({"report": fallback})
