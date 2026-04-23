# agents/safety_agent.py
"""
Safety Agent - Kiểm soát nội dung đầu ra y tế.
Đảm bảo hệ thống không đưa ra chẩn đoán xác định, kê đơn,
khuyến nghị điều trị hay tiên lượng trực tiếp.
"""
import re
import logging

logger = logging.getLogger(__name__)

SAFETY_DISCLAIMER = (
    "This is an AI-assisted analysis for research purposes only. "
    "It is not a medical diagnosis. "
    "Please consult a qualified radiologist or physician."
)

SAFETY_DISCLAIMER_VI = (
    "Đây là phân tích hỗ trợ bởi AI chỉ dành cho mục đích nghiên cứu. "
    "Đây không phải là chẩn đoán y tế. "
    "Vui lòng tham khảo ý kiến bác sĩ X-quang hoặc bác sĩ chuyên khoa."
)

# Các pattern vi phạm đạo đức y tế
_VIOLATION_PATTERNS = [
    # Chẩn đoán xác định
    (r"\byou have\b", "definitive diagnosis"),
    (r"\bpatient has\b", "definitive diagnosis"),
    (r"\bdiagnosed with\b", "definitive diagnosis"),
    (r"\bconfirmed\b.{0,30}\b(disease|condition|diagnosis)\b", "definitive diagnosis"),
    # Kê đơn / liều lượng
    (r"\btake\s+\d+\s*(mg|ml|tablet|pill)\b", "prescription"),
    (r"\bprescribe\b", "prescription"),
    (r"\bdosage\b", "prescription"),
    # Khuyến nghị điều trị trực tiếp
    (r"\byou should (take|use|start|stop|avoid)\b", "treatment recommendation"),
    (r"\btreatment is\b", "treatment recommendation"),
    (r"\bmedication (is|are)\b", "treatment recommendation"),
    # Tiên lượng
    (r"\byou will\b.{0,30}\b(die|recover|worsen|improve)\b", "prognosis"),
    (r"\bthis will lead to\b", "prognosis"),
    (r"\blife expectancy\b", "prognosis"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), label) for p, label in _VIOLATION_PATTERNS]


def _detect_violations(text: str) -> list[str]:
    """Trả về danh sách loại vi phạm tìm thấy."""
    found = []
    for pattern, label in _COMPILED:
        if pattern.search(text):
            found.append(label)
    return list(set(found))


def _sanitize(text: str) -> str:
    """Thay thế nội dung vi phạm bằng cụm từ an toàn."""
    text = re.sub(r"\byou have\b", "imaging findings suggest", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpatient has\b", "imaging findings suggest", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdiagnosed with\b", "findings consistent with", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou should (take|use|start|stop|avoid)\b",
                  r"physician may consider \1ing", text, flags=re.IGNORECASE)
    text = re.sub(r"\btreatment is\b", "management options include", text, flags=re.IGNORECASE)
    return text


class SafetyAgent:
    def run(self, explanation_output: dict, lang: str = "en") -> dict:
        report = explanation_output.get("report", "")
        disclaimer = SAFETY_DISCLAIMER_VI if lang == "vi" else SAFETY_DISCLAIMER

        violations = _detect_violations(report)
        if violations:
            logger.warning(
                f"SafetyAgent: violations detected {violations}. "
                f"Original report: {report[:200]}"
            )
            report = _sanitize(report)

        if SAFETY_DISCLAIMER not in report and SAFETY_DISCLAIMER_VI not in report:
            report = f"{report}\n\n{disclaimer}"

        logger.info(f"SafetyAgent: review complete. Violations found: {violations or 'none'}")
        return {
            "report": report,
            "safety_reviewed": True,
            "violations_found": violations,
        }
        """
        Review report từ Explanation Agent.
        - Phát hiện và sanitize nội dung vi phạm
        - Thêm SAFETY_DISCLAIMER vào cuối
        - Trả về dict với key 'report' và 'safety_reviewed'
        """
        report = explanation_output.get("report", "")

        violations = _detect_violations(report)
        if violations:
            logger.warning(
                f"SafetyAgent: violations detected {violations}. "
                f"Original report: {report[:200]}"
            )
            report = _sanitize(report)

        # Đảm bảo disclaimer cuối
        if SAFETY_DISCLAIMER not in report:
            report = f"{report}\n\n{SAFETY_DISCLAIMER}"

        logger.info(f"SafetyAgent: review complete. Violations found: {violations or 'none'}")
        return {
            "report": report,
            "safety_reviewed": True,
            "violations_found": violations,
        }
