# config.py
import os

# Cấu hình đường dẫn dữ liệu
BASE_DATA_DIR = "data/iu_xray" # Đổi lại cho đúng với máy/Kaggle của bạn
REPORTS_PATH = os.path.join(BASE_DATA_DIR, "indiana_reports.csv")
PROJECTIONS_PATH = os.path.join(BASE_DATA_DIR, "indiana_projections.csv")
IMAGES_DIR = os.path.join(BASE_DATA_DIR, "images")

# Cấu hình đánh giá
START_INDEX = 0   # Bắt đầu từ mẫu nào (Index bắt đầu từ 0)
BATCH_SIZE = 50    # Số lượng mẫu muốn chạy trong đợt này