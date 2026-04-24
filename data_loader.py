# prepare_data.py
import pandas as pd
import os
import config

def extract_multi_labels(problems_str):
    """
    Ánh xạ cột Problems sang 17 nhãn chuẩn.
    Logic giống hệt script Kaggle: dùng `if` độc lập, check trên toàn bộ chuỗi.
    """
    problems_str = str(problems_str).lower()

    if problems_str == 'nan' or 'normal' in problems_str:
        return ['No Finding']

    labels = set()
    if "cardiomegaly" in problems_str or "enlarged heart" in problems_str:
        labels.add("Cardiomegaly")
    if "atelectasis" in problems_str:
        labels.add("Atelectasis")
    if "effusion" in problems_str or "fluid" in problems_str:
        labels.add("Effusion")
    if "pneumonia" in problems_str:
        labels.add("Pneumonia")
    if "pneumothorax" in problems_str:
        labels.add("Pneumothorax")
    if "consolidation" in problems_str:
        labels.add("Consolidation")
    if "edema" in problems_str or "congestion" in problems_str:
        labels.add("Edema")
    if "nodule" in problems_str:
        labels.add("Nodule")
    if "mass" in problems_str:
        labels.add("Mass")
    if "hernia" in problems_str:
        labels.add("Hernia")
    if "fibrosis" in problems_str or "scar" in problems_str:
        labels.add("Fibrosis")
    if "thickening" in problems_str:                        # bỏ "pleural" tránh nhầm với effusion
        labels.add("Pleural_Thickening")
    if "emphysema" in problems_str or "copd" in problems_str:
        labels.add("Emphysema")
    if "fracture" in problems_str:
        labels.add("Fracture")
    if "infiltrat" in problems_str:
        labels.add("Infiltration")
    if "opacity" in problems_str:
        labels.add("Lung Opacity")

    if len(labels) == 0:
        return ['No Finding']
    return list(labels)

def load_and_process_data():
    """
    Đọc 2 file CSV, nối chúng lại và tạo danh sách dữ liệu test ĐA NHÃN.
    """
    # Kiểm tra file tồn tại
    if not os.path.exists(config.REPORTS_PATH) or not os.path.exists(config.PROJECTIONS_PATH):
        print(f"❌ Lỗi: Không tìm thấy file CSV trong {config.BASE_DATA_DIR}")
        return []

    print("🔄 Đang đọc dữ liệu CSV...")
    try:
        reports_df = pd.read_csv(config.REPORTS_PATH)
        projections_df = pd.read_csv(config.PROJECTIONS_PATH)
    except Exception as e:
        print(f"❌ Lỗi đọc file CSV: {e}")
        return []

    # Nối (Merge) 2 bảng dựa trên 'uid'
    merged_df = pd.merge(reports_df, projections_df, on="uid")

    # Lọc dữ liệu: Chỉ lấy ảnh chụp thẳng (Frontal)
    frontal_df = merged_df[merged_df['projection'] == 'Frontal']
    
    # Xử lý dữ liệu rỗng
    frontal_df = frontal_df.fillna({
        'findings': 'No findings reported.',
        'impression': '',
        'Problems': 'normal'
    })

    test_data = []
    print(f"🔄 Đang xử lý {len(frontal_df)} ảnh chụp thẳng (Frontal)...")

    for index, row in frontal_df.iterrows():
        filename = str(row['filename'])
        full_image_path = os.path.join(config.IMAGES_DIR, filename)
        
        # Nếu file gốc không tồn tại, thử hoán đổi đuôi .png <-> .dcm.png
        if not os.path.exists(full_image_path):
            alt_name = filename.replace(".dcm.png", ".png") if filename.endswith(".dcm.png") else filename.replace(".png", ".dcm.png")
            full_image_path = os.path.join(config.IMAGES_DIR, alt_name)
            
            # Nếu vẫn không thấy thì bỏ qua
            if not os.path.exists(full_image_path):
                continue

        raw_problems = str(row['Problems'])
        true_labels = extract_multi_labels(raw_problems)

        findings = str(row['findings']).replace("XXXX", "")
        impression = str(row['impression']).replace("XXXX", "")
        full_report = f"Findings: {findings}\nImpression: {impression}".strip()
        full_report = " ".join(full_report.split())

        if len(full_report) > 10:
            test_data.append({
                "filename": os.path.basename(full_image_path),
                "image_path": full_image_path,
                "true_labels": true_labels, 
                "true_report": full_report
            })

    print(f"✅ Đã chuẩn bị xong {len(test_data)} mẫu dữ liệu hợp lệ.")
    return test_data

if __name__ == "__main__":
    # Test nhanh xem file có chạy đúng không
    data = load_and_process_data()
    if data:
        print("\n" + "="*70)
        print(f"📄 HIỂN THỊ MẪU KIỂM TRA ĐẦU TIÊN:")
        print("="*70)
        sample = data[0]
        print(f"📂 File:   {sample['filename']}")
        print(f"🏷️ Labels: {', '.join(sample['true_labels'])}") 
        print(f"📝 Report: {sample['true_report'][:150]}...")