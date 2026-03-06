import pandas as pd
import os

def load_and_process_data(base_dir="data/iu_xray"):
    """
    Đọc 2 file CSV, nối chúng lại và tạo danh sách dữ liệu test ĐA NHÃN.
    """
    # 1. Định nghĩa đường dẫn
    reports_path = os.path.join(base_dir, "indiana_reports.csv")
    projections_path = os.path.join(base_dir, "indiana_projections.csv")
    images_dir = os.path.join(base_dir, "images")

    # Kiểm tra file tồn tại
    if not os.path.exists(reports_path) or not os.path.exists(projections_path):
        print(f"❌ Lỗi: Không tìm thấy file CSV trong {base_dir}")
        return []

    # 2. Đọc dữ liệu
    print("🔄 Đang đọc dữ liệu CSV...")
    try:
        reports_df = pd.read_csv(reports_path)
        projections_df = pd.read_csv(projections_path)
    except Exception as e:
        print(f"❌ Lỗi đọc file CSV: {e}")
        return []

    # 3. Nối (Merge) 2 bảng dựa trên 'uid'
    merged_df = pd.merge(reports_df, projections_df, on="uid")

    # 4. Lọc dữ liệu: Chỉ lấy ảnh chụp thẳng (Frontal)
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
        # A. Xử lý đường dẫn ảnh (Tối ưu hóa gọn gàng hơn)
        filename = str(row['filename'])
        full_image_path = os.path.join(images_dir, filename)
        
        # Nếu file gốc không tồn tại, thử hoán đổi đuôi .png <-> .dcm.png
        if not os.path.exists(full_image_path):
            alt_name = filename.replace(".dcm.png", ".png") if filename.endswith(".dcm.png") else filename.replace(".png", ".dcm.png")
            full_image_path = os.path.join(images_dir, alt_name)
            
            # Nếu vẫn không thấy thì bỏ qua (skip) mẫu này
            if not os.path.exists(full_image_path):
                continue

        # B. Xử lý Nhãn bệnh (SỬA LỖI: Lấy toàn bộ danh sách ĐA NHÃN)
        raw_problems = str(row['Problems'])
        true_labels = extract_multi_labels(raw_problems)

        # C. Xử lý Báo cáo
        findings = str(row['findings']).replace("XXXX", "")
        impression = str(row['impression']).replace("XXXX", "")
        full_report = f"Findings: {findings}\nImpression: {impression}".strip()
        full_report = " ".join(full_report.split()) # Xóa khoảng trắng thừa

        # Chỉ lấy những mẫu có báo cáo đủ dài (> 10 ký tự)
        if len(full_report) > 10:
            test_data.append({
                "filename": os.path.basename(full_image_path),
                "image_path": full_image_path,
                "true_labels": true_labels,  # Đổi thành list chứa đa nhãn
                "true_report": full_report
            })

    print(f"✅ Đã chuẩn bị xong {len(test_data)} mẫu dữ liệu hợp lệ.")
    return test_data

def extract_multi_labels(problems_str):
    """Ánh xạ toàn bộ danh sách bệnh từ cột Problems sang chuẩn 17 nhãn"""
    problems_str = str(problems_str).lower()
    
    if problems_str == 'nan' or 'normal' in problems_str:
        return ['No Finding']
    
    labels = set()
    # Duyệt qua TẤT CẢ các bệnh được phân cách bởi dấu chấm phẩy
    for problem in problems_str.split(';'):
        p = problem.strip()
        if "cardiomegaly" in p or "enlarged heart" in p: labels.add("Cardiomegaly")
        elif "atelectasis" in p: labels.add("Atelectasis")
        elif "effusion" in p or "fluid" in p: labels.add("Effusion")
        elif "pneumonia" in p: labels.add("Pneumonia")
        elif "pneumothorax" in p: labels.add("Pneumothorax")
        elif "consolidation" in p: labels.add("Consolidation")
        elif "edema" in p or "congestion" in p: labels.add("Edema")
        elif "nodule" in p: labels.add("Nodule")
        elif "mass" in p: labels.add("Mass")
        elif "hernia" in p: labels.add("Hernia")
        elif "fibrosis" in p or "scar" in p: labels.add("Fibrosis")
        elif "thickening" in p or "pleural" in p: labels.add("Pleural_Thickening")
        elif "emphysema" in p or "copd" in p: labels.add("Emphysema")
        elif "fracture" in p: labels.add("Fracture")
        elif "infiltrat" in p: labels.add("Infiltration")
        elif "opacity" in p: labels.add("Lung Opacity")
        
    if len(labels) == 0:
        return ['No Finding']
    
    return list(labels)

# --- TEST LOCAL ---
if __name__ == "__main__":
    # Thay đổi đường dẫn này trỏ tới thư mục chứa dữ liệu của bạn trên máy/Kaggle
    all_data = load_and_process_data(base_dir="data/iu_xray")
    
    if all_data:
        num_samples = min(20, len(all_data))
        samples = all_data[:num_samples]
        
        print("\n" + "="*70)
        print(f"📄 HIỂN THỊ {num_samples} MẪU DỮ LIỆU KIỂM TRA (TEST DATA)")
        print("="*70)
        
        for i, sample in enumerate(samples):
            print(f"🔹 MẪU [{i+1}/{num_samples}]")
            print(f"   📂 File:   {sample['filename']}")
            # In ra danh sách các nhãn (Ví dụ: ['Cardiomegaly', 'Effusion'])
            print(f"   🏷️ Labels: {', '.join(sample['true_labels'])}") 
            short_report = sample['true_report'][:120] + "..." if len(sample['true_report']) > 120 else sample['true_report']
            print(f"   📝 Report: {short_report}")
            print("-" * 70)
    else:
        print("⚠️ Không tải được dữ liệu.")