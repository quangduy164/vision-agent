import pandas as pd
import os
import shutil

# --- CẤU HÌNH ---
os.environ['KAGGLE_USERNAME'] = "duy164"
os.environ['KAGGLE_KEY'] = "KGAT_e170510dd461f3822dee8ee5ad7bc4ce"

from kaggle.api.kaggle_api_extended import KaggleApi

# Cấu hình tải về
DATASET_NAME = "raddar/chest-xrays-indiana-university"
TARGET_DIR = "data/iu_xray/images"
CSV_PATH = "data/iu_xray/indiana_projections.csv" # Đường dẫn file CSV bạn đã có
LIMIT = 200 # Số lượng ảnh muốn tải

def download_subset():
    # 1. Tạo thư mục chứa ảnh
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 2. Đọc file CSV để lấy danh sách tên ảnh
    if not os.path.exists(CSV_PATH):
        print(f"❌ Không tìm thấy file {CSV_PATH}. Hãy đặt file này vào đúng chỗ.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # 3. Lọc lấy ảnh Frontal (Chụp thẳng)
    frontal_df = df[df['projection'] == 'Frontal']
    
    # Lấy 200 ảnh đầu tiên
    subset_files = frontal_df['filename'].head(LIMIT).tolist()
    
    print(f"📋 Đã tìm thấy {len(frontal_df)} ảnh Frontal. Sẽ tải {len(subset_files)} ảnh...")

    # 4. Kết nối Kaggle API
    api = KaggleApi()
    api.authenticate()

    # 5. Tải từng file
    downloaded_count = 0
    for file_name in subset_files:
        # Đường dẫn file trong file zip trên Kaggle thường nằm trong folder 'images/images_normalized' 
        # hoặc nằm thẳng ở ngoài tùy dataset. 
        # Với dataset của 'raddar', cấu trúc là images/images_normalized/
        
        save_path = os.path.join(TARGET_DIR, file_name)
        
        if os.path.exists(save_path):
            print(f"⏩ Đã có: {file_name}")
            downloaded_count += 1
            continue
            
        try:
            print(f"⬇️ Đang tải ({downloaded_count + 1}/{LIMIT}): {file_name}")
            
            # Hàm này tải file về thư mục hiện tại
            api.dataset_download_file(
                DATASET_NAME,
                file_name=f"images/images_normalized/{file_name}", # Đường dẫn cụ thể trên Kaggle
                path=TARGET_DIR
            )
            downloaded_count += 1
            
        except Exception as e:
            print(f"❌ Lỗi tải {file_name}: {e}")

    print("\n✅ HOÀN TẤT! Hãy kiểm tra thư mục:", TARGET_DIR)
    
    # Lưu ý: Kaggle API khi tải file lẻ có thể để nó trong file zip riêng lẻ.
    # Code dưới đây giải nén nếu cần
    for item in os.listdir(TARGET_DIR):
        if item.endswith(".zip"):
            file_path = os.path.join(TARGET_DIR, item)
            # Giải nén
            shutil.unpack_archive(file_path, TARGET_DIR)
            # Xóa file zip
            os.remove(file_path)

if __name__ == "__main__":
    download_subset()