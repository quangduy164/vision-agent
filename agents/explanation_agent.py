# agents/explanation_agent.py
"""
Explanation Agent - Sinh báo cáo y tế từ kết quả Vision Agent.
Dùng generate_report tool nội bộ.
"""
import json
import logging
from models.decoder import BioGPTDecoder
from agents.tools import generate_report, init_decoder

logger = logging.getLogger(__name__)


class ExplanationAgent:
    def __init__(self):
        decoder = BioGPTDecoder()
        init_decoder(decoder)  # inject vào report_tool
        logger.info("ExplanationAgent initialized.")

    def run(self, vision_output: dict) -> dict:
        findings = vision_output.get("visual_findings", {})
        result = json.loads(generate_report.invoke({
            "diagnosis":  vision_output.get("top_disease", "No Finding"),
            "confidence": vision_output.get("confidence", 0.0),
            "location":   findings.get("location", "chest"),
            "size":       findings.get("size", "moderate"),
            "side":       findings.get("side", "unspecified"),
        }))
        logger.info("ExplanationAgent: report done.")
        return result
