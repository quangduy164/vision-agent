# sample_images.py
"""
Copy 5 ảnh mỗi loại bệnh từ dataset IU X-Ray vào uploads/test_samples/
để tiện test trên web UI.
CLASSES được import thẳng từ models/classifier.py để đảm bảo đồng nhất.
"""
import os
import shutil
import pandas as pd
from data_loader import extract_multi_labels
from models.classifier import CLASSES
import config

DEST_DIR          = "uploads/test_samples"
SAMPLES_PER_CLASS = 5


def main():
    os.makedirs(DEST_DIR, exist_ok=True)

    reports_df     = pd.read_csv(config.REPORTS_PATH)
    projections_df = pd.read_csv(config.PROJECTIONS_PATH)
    merged  = pd.merge(reports_df, projections_df, on="uid")
    frontal = merged[merged['projection'] == 'Frontal'].fillna({'Problems': 'normal'})

    # Build danh sách (image_path, labels) - dùng đúng logic của evaluate.py
    samples = []
    for _, row in frontal.iterrows():
        fname = str(row['filename'])
        path  = os.path.join(config.IMAGES_DIR, fname)
        if not os.path.exists(path):
            alt = fname.replace(".dcm.png", ".png") if fname.endswith(".dcm.png") \
                  else fname.replace(".png", ".dcm.png")
            path = os.path.join(config.IMAGES_DIR, alt)
            if not os.path.exists(path):
                continue
        # Dùng đúng hàm extract_multi_labels từ prepare_data (giống evaluate.py)
        labels = extract_multi_labels(str(row['Problems']))
        samples.append((path, labels))

    # Lấy tối đa SAMPLES_PER_CLASS ảnh mỗi class
    collected = {c: [] for c in CLASSES}
    for path, labels in samples:
        for label in labels:
            if label in collected and len(collected[label]) < SAMPLES_PER_CLASS:
                collected[label].append(path)

    # Copy và đổi tên theo class
    total = 0
    print(f"\n📋 CLASSES từ classifier.py ({len(CLASSES)} nhãn):\n")
    for cls, paths in collected.items():
        cls_dir = os.path.join(DEST_DIR, cls.replace(" ", "_").replace("/", "-"))
        os.makedirs(cls_dir, exist_ok=True)
        for i, src in enumerate(paths, 1):
            ext  = os.path.splitext(src)[1]
            dest = os.path.join(cls_dir, f"{i}{ext}")
            shutil.copy2(src, dest)
            total += 1
        found  = len(paths)
        status = "✅" if found == SAMPLES_PER_CLASS else f"⚠️  chỉ có {found}"
        print(f"  {status}  {cls:<22} → {found} ảnh")

    print(f"\n✅ Đã copy {total} ảnh vào {DEST_DIR}/")


if __name__ == "__main__":
    main()
