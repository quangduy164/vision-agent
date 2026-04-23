# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Đường dẫn dữ liệu ---
BASE_DATA_DIR    = "data/iu_xray"
REPORTS_PATH     = os.path.join(BASE_DATA_DIR, "indiana_reports.csv")
PROJECTIONS_PATH = os.path.join(BASE_DATA_DIR, "indiana_projections.csv")
IMAGES_DIR       = os.path.join(BASE_DATA_DIR, "images")

# --- Cấu hình đánh giá ---
START_INDEX = 0   # Bắt đầu từ mẫu nào
BATCH_SIZE  = 50  # Số mẫu mỗi lần chạy

# --- Cấu hình LLM (đọc từ .env) ---
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "")       # openai | google | ollama | (trống = fallback)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")
LLM_API_KEY    = os.getenv("LLM_API_KEY", "")

# --- Model ---
MODEL_PATH = "best_densenet_finetuned.pth"
