# download_data.py
"""
Tải toàn bộ ảnh Frontal từ IU X-Ray dataset về data/iu_xray/images/
Số lượng khớp với tập validation trên Kaggle (train_test_split 80/20, random_state=42).
Yêu cầu: KAGGLE_USERNAME và KAGGLE_KEY trong file .env
"""
import os
import shutil
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATASET_NAME = "raddar/chest-xrays-indiana-university"
TARGET_DIR   = "data/iu_xray/images"
CSV_PATH     = "data/iu_xray/indiana_projections.csv"
LIMIT        = 1000  # Số ảnh Frontal muốn tải


def get_filenames():
    df = pd.read_csv(CSV_PATH)
    return df[df['projection'] == 'Frontal']['filename'].head(LIMIT).tolist()


def download_subset():
    os.makedirs(TARGET_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f"❌ Không tìm thấy {CSV_PATH}")
        return

    val_files = get_filenames()
    total = len(val_files)
    print(f"📋 Sẽ tải {total} ảnh Frontal...")

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    downloaded = 0
    skipped    = 0
    failed     = 0

    for i, file_name in enumerate(val_files, 1):
        save_path = os.path.join(TARGET_DIR, file_name)
        if os.path.exists(save_path):
            skipped += 1
            print(f"⏩ [{i}/{total}] Đã có: {file_name}")
            continue
        try:
            print(f"⬇️  [{i}/{total}] Tải: {file_name}")
            api.dataset_download_file(
                DATASET_NAME,
                file_name=f"images/images_normalized/{file_name}",
                path=TARGET_DIR,
            )
            downloaded += 1
        except Exception as e:
            print(f"❌ Lỗi {file_name}: {e}")
            failed += 1

    # Giải nén zip nếu có
    for item in os.listdir(TARGET_DIR):
        if item.endswith(".zip"):
            shutil.unpack_archive(os.path.join(TARGET_DIR, item), TARGET_DIR)
            os.remove(os.path.join(TARGET_DIR, item))

    print(f"\n✅ Hoàn tất: {downloaded} tải mới | {skipped} đã có | {failed} lỗi")
    print(f"📂 Thư mục: {TARGET_DIR}")


if __name__ == "__main__":
    download_subset()
