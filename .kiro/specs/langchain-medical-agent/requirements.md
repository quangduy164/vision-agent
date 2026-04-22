# Requirements Document

## Introduction

Tái kiến trúc hệ thống phân tích ảnh X-quang phổi thành một hệ thống **đa tác nhân (Multi-Agent System)** sử dụng LangChain. Hệ thống gồm 3 Agent chuyên biệt phối hợp với nhau: **Vision Agent** (phân tích ảnh), **Explanation Agent** (sinh báo cáo ngôn ngữ tự nhiên), và **Safety Agent** (kiểm soát nội dung y tế). Các model đã có (DenseNet121 ensemble + ResNet50, Grad-CAM, Segmentation, BioGPT/template) được giữ nguyên và wrap thành LangChain tools.

## Glossary

- **Vision Agent**: Agent chuyên phân tích ảnh X-quang, điều phối các tool kỹ thuật (classify, gradcam, segment)
- **Explanation Agent**: Agent chuyên sinh báo cáo y tế ngôn ngữ tự nhiên từ kết quả của Vision Agent
- **Safety Agent**: Agent kiểm soát nội dung đầu ra, đảm bảo không vi phạm đạo đức y tế
- **Orchestrator**: LangChain Agent cấp cao điều phối luồng giữa 3 Agent chuyên biệt
- **Tool**: Một LangChain tool wrapping một chức năng phân tích ảnh X-quang cụ thể
- **ReAct**: Reasoning + Acting pattern - mỗi bước gồm Thought → Action → Observation
- **LLM**: Language model điều phối Agent (OpenAI / Google / Ollama local)
- **CLASSES**: 17 nhãn bệnh: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Fracture, Hernia, Infiltration, Lung Opacity, Mass, No Finding, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax
- **BEST_THRESHOLDS**: Ngưỡng chẩn đoán tối ưu theo từng bệnh, xác định từ tập validation
- **IU X-Ray**: Tập dữ liệu Indiana University X-Ray dùng để đánh giá chất lượng báo cáo
- **Safety Violation**: Nội dung vi phạm đạo đức y tế: chẩn đoán xác định, kê đơn thuốc, khuyến nghị điều trị trực tiếp

---

## Requirements

### Requirement 1: Vision Agent - Tool Classify X-Ray

**User Story:** As a Vision Agent, I want a classify tool that runs ensemble inference on an X-ray image, so that I can identify potential pathologies with calibrated probabilities.

#### Acceptance Criteria

1. Tool SHALL nhận image file path làm input, trả về JSON chứa toàn bộ 17 nhãn kèm xác suất và danh sách nhãn vượt ngưỡng `BEST_THRESHOLDS`
2. Tool SHALL sử dụng ensemble DenseNet121 (224x224) + ResNet50 (512x512) với trọng số động theo từng bệnh
3. WHEN image không tồn tại hoặc không đọc được, THEN tool SHALL trả về `{"error": "<mô tả lỗi>"}` thay vì raise exception
4. Tool SHALL được đăng ký như LangChain `@tool` với docstring đủ rõ để LLM biết khi nào gọi

### Requirement 2: Vision Agent - Tool GradCAM

**User Story:** As a Vision Agent, I want a gradcam tool that generates visual heatmaps, so that I can show which regions of the X-ray influenced the classification decision.

#### Acceptance Criteria

1. Tool SHALL nhận image path và disease name, trả về JSON chứa đường dẫn file heatmap đã lưu
2. Tool SHALL lưu heatmap vào `outputs/` với tên file gồm tên bệnh và timestamp
3. WHEN disease name không thuộc CLASSES, THEN tool SHALL trả về `{"error": "invalid disease"}` 
4. Tool SHALL chỉ chạy trên DenseNet121 vì ResNet50 không có `pathologies` map tương thích

### Requirement 3: Vision Agent - Tool Segment Lesion

**User Story:** As a Vision Agent, I want a segmentation tool that extracts lesion location and size, so that the Explanation Agent has spatial context for report generation.

#### Acceptance Criteria

1. Tool SHALL nhận image path và disease name, trả về JSON `{"location": ..., "size": ..., "side": ...}` theo text y khoa chuẩn
2. Tool SHALL nội bộ chạy Grad-CAM để lấy heatmap trước khi segment, không yêu cầu gọi GradCAM tool trước
3. WHEN không tìm thấy contour, THEN tool SHALL trả về `{"location": "diffuse area", "size": "small", "side": "unspecified"}`

### Requirement 4: Vision Agent - Orchestration

**User Story:** As a developer, I want the Vision Agent to orchestrate classify/gradcam/segment tools using ReAct, so that it can reason about which tools to call based on intermediate results.

#### Acceptance Criteria

1. Vision Agent SHALL được khởi tạo với 3 tools: classify, gradcam, segment và một LLM làm reasoning engine
2. WHEN nhận image path, Vision Agent SHALL tự reasoning để quyết định thứ tự gọi tools
3. Vision Agent SHALL trả về structured output: `{"diagnoses": [...], "top_disease": ..., "confidence": ..., "visual_findings": {...}, "heatmap_path": ...}`
4. IF LLM không khả dụng, THEN Vision Agent SHALL fallback về pipeline cứng: classify → segment → gradcam

### Requirement 5: Explanation Agent - Sinh báo cáo y tế

**User Story:** As an Explanation Agent, I want to generate structured radiology reports from Vision Agent output, so that findings are communicated in clear clinical language.

#### Acceptance Criteria

1. Explanation Agent SHALL nhận output của Vision Agent làm input và sinh báo cáo theo format Findings + Impression
2. Explanation Agent SHALL sử dụng bridge template (models/bridge.py) với khả năng mở rộng sang BioGPT khi `USE_TEMPLATE_ONLY = False`
3. Explanation Agent SHALL KHÔNG đưa ra chẩn đoán xác định hay khuyến nghị điều trị - chỉ mô tả findings quan sát được
4. Báo cáo SHALL luôn kết thúc bằng câu nhắc nhở: *"Clinical correlation and physician review are recommended."*
5. WHEN diagnosis là "No Finding", Explanation Agent SHALL sinh báo cáo bình thường không có findings bất thường

### Requirement 6: Safety Agent - Kiểm soát nội dung y tế

**User Story:** As a Safety Agent, I want to review all outputs before they are returned to the user, so that the system never makes definitive diagnoses or treatment recommendations that could harm patients.

#### Acceptance Criteria

1. Safety Agent SHALL kiểm tra output của Explanation Agent trước khi trả về user, phát hiện các vi phạm Safety Violation
2. Safety Agent SHALL từ chối (block) output chứa bất kỳ nội dung nào trong danh sách cấm:
   - Chẩn đoán xác định dạng "You have [disease]" hoặc "Patient has [disease]"
   - Kê đơn thuốc hoặc liều lượng cụ thể
   - Khuyến nghị điều trị trực tiếp ("You should take...", "Treatment is...")
   - Tuyên bố tiên lượng ("You will...", "This will lead to...")
3. WHEN Safety Agent phát hiện vi phạm, THEN SHALL thay thế nội dung vi phạm bằng disclaimer chuẩn và log cảnh báo
4. Safety Agent SHALL luôn thêm disclaimer vào cuối mọi output: *"This is an AI-assisted analysis for research purposes only. It is not a medical diagnosis. Please consult a qualified radiologist or physician."*
5. Safety Agent SHALL log mọi lần phát hiện vi phạm kèm nội dung gốc để audit

### Requirement 7: Orchestrator - Điều phối đa tác nhân

**User Story:** As a developer, I want an Orchestrator that coordinates the three agents in sequence, so that the full analysis pipeline runs end-to-end with proper handoffs.

#### Acceptance Criteria

1. Orchestrator SHALL điều phối luồng: Vision Agent → Explanation Agent → Safety Agent theo thứ tự
2. Orchestrator SHALL truyền output của Vision Agent làm input cho Explanation Agent
3. Orchestrator SHALL truyền output của Explanation Agent cho Safety Agent để review trước khi trả về
4. WHEN bất kỳ Agent nào gặp lỗi, Orchestrator SHALL log lỗi và trả về partial result thay vì crash
5. Orchestrator SHALL expose interface `analyze(image_path, output_dir)` tương thích với `app.py` hiện tại
6. Orchestrator SHALL trả về dict với đầy đủ keys: `image`, `status`, `diagnosis`, `confidence`, `visual_findings`, `report`, `output_image`, `all_probabilities`, `safety_reviewed`

### Requirement 8: Quản lý Model State

**User Story:** As a developer, I want models to be loaded once and shared across all agents and tools, so that the system avoids redundant loading overhead.

#### Acceptance Criteria

1. Orchestrator SHALL load tất cả models (DenseNet121, ResNet50) một lần khi khởi tạo và inject vào Vision Agent
2. Tất cả tools trong Vision Agent SHALL dùng chung model instances đã load, không tự load lại
3. WHEN một Session kết thúc, memory của Session đó SHALL được xóa để tránh context leak

### Requirement 9: Cấu hình LLM linh hoạt

**User Story:** As a developer, I want to configure the LLM backend via environment variables, so that the system works with different providers or runs fully offline.

#### Acceptance Criteria

1. Hệ thống SHALL đọc cấu hình từ `.env`: `LLM_PROVIDER` (openai/google/ollama), `LLM_MODEL_NAME`, `LLM_API_KEY`
2. WHERE `LLM_PROVIDER=ollama`, hệ thống SHALL kết nối Ollama local server để chạy offline không cần API key
3. WHERE `LLM_PROVIDER=openai`, hệ thống SHALL dùng OpenAI ChatGPT API
4. IF `LLM_PROVIDER` không hợp lệ hoặc thiếu, THEN hệ thống SHALL log warning và dùng fallback pipeline không cần LLM

### Requirement 10: Logging và Observability

**User Story:** As a developer, I want detailed logging of each agent's reasoning steps, so that I can debug and monitor the multi-agent decision-making process.

#### Acceptance Criteria

1. Mỗi Agent SHALL log các bước ReAct (Thought, Action, Observation) với timestamp và session ID
2. WHEN một Tool được gọi, hệ thống SHALL log tên tool, input và thời gian thực thi (ms)
3. Safety Agent SHALL log riêng mọi lần phát hiện Safety Violation kèm nội dung gốc
4. Hệ thống SHALL hỗ trợ LangChain callbacks để tích hợp LangSmith khi được cấu hình

### Requirement 11: Tương thích Evaluation

**User Story:** As a researcher, I want the multi-agent system to be compatible with the existing evaluate.py script, so that I can compare BLEU/ROUGE metrics against the baseline pipeline.

#### Acceptance Criteria

1. Orchestrator SHALL tương thích với `evaluate.py` hiện tại - không cần sửa evaluate.py
2. WHEN chạy evaluation, mỗi ảnh SHALL được xử lý độc lập và trả về `report` text để tính BLEU/ROUGE
3. Chất lượng báo cáo SHALL không thấp hơn baseline pipeline (đo bằng ROUGE-L)
