# evaluate.py
import torch
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

# Import hàm load dữ liệu
from prepare_data import load_and_process_data

# Tải dữ liệu NLTK (Fix lỗi phiên bản mới)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab') 
except LookupError:
    print("⬇️ Đang tải dữ liệu NLTK cần thiết...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Import Agent
from agent import MedicalVisionAgent

# --- 1. HÀM TÍNH ĐIỂM NGÔN NGỮ ---
def calculate_text_metrics(reference_text, candidate_text):
    """
    Tính điểm BLEU và ROUGE giữa báo cáo thật và báo cáo AI sinh ra.
    """
    if not candidate_text or not reference_text:
        return {"BLEU": 0, "ROUGE-1": 0, "ROUGE-L": 0}

    # Chuẩn hóa văn bản
    ref_tokens = nltk.word_tokenize(reference_text.lower())
    cand_tokens = nltk.word_tokenize(candidate_text.lower())

    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return {"BLEU": 0, "ROUGE-1": 0, "ROUGE-L": 0}

    # Tính BLEU (Smoothing để tránh lỗi chia cho 0)
    cc = SmoothingFunction()
    bleu_score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=cc.method1)

    # Tính ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_text, candidate_text)
    
    return {
        "BLEU": bleu_score,
        "ROUGE-1": rouge_scores['rouge1'].fmeasure,
        "ROUGE-L": rouge_scores['rougeL'].fmeasure
    }

# --- 2. HÀM CHẠY ĐÁNH GIÁ ---
def run_evaluation(test_data):
    agent = MedicalVisionAgent()
    
    # DANH SÁCH CÁC CẶP BỆNH TƯƠNG ĐỒNG (Định nghĩa 1 lần ở đây)
    SIMILAR_DISEASES = {
        "Cardiomegaly": ["Enlarged Cardiomediastinum"],
        "Enlarged Cardiomediastinum": ["Cardiomegaly"],
        "Effusion": ["Atelectasis", "Edema", "Infiltration", "Pleural_Thickening"],
        "Atelectasis": ["Effusion", "Infiltration", "Consolidation", "Lung Opacity"],
        "Pneumonia": ["Consolidation", "Infiltration", "Lung Opacity"],
        "Infiltration": ["Pneumonia", "Lung Opacity", "Atelectasis", "Effusion"],
        "Lung Opacity": ["Pneumonia", "Infiltration", "Consolidation", "Atelectasis"],
        "Mass": ["Nodule", "Lung Lesion"],
        "Nodule": ["Mass", "Lung Lesion"]
    }
    
    y_true_cls = []  # Nhãn thật
    y_pred_cls = []  # Nhãn dự đoán
    
    text_metrics_sum = {"BLEU": 0, "ROUGE-1": 0, "ROUGE-L": 0}
    
    total_samples = len(test_data)
    print(f"\n🚀 BẮT ĐẦU ĐÁNH GIÁ TRÊN {total_samples} ẢNH...")
    print("="*60)
    
    success_count = 0

    for i, item in enumerate(test_data):
        img_name = os.path.basename(item["image_path"])
        print(f"📸 [{i+1}/{total_samples}] Đang xử lý: {img_name}...", end=" ")
        
        try:
            # A. Chạy Agent
            result = agent.analyze(item["image_path"], output_dir="eval_outputs")
            
            # Kiểm tra lỗi từ Agent
            if "error" in result:
                print(f"❌ Lỗi Agent: {result['error']}")
                continue

            # B. Đánh giá Vision
            pred_label = result["diagnosis"]
            
            # ĐÃ SỬA: Lấy danh sách đa nhãn (có chữ 's')
            true_labels = item["true_labels"] 
            
            # Logic so sánh Đa nhãn
            is_correct = 0
            
            # 1. So khớp chính xác: Dự đoán có nằm trong danh sách bệnh thực tế không?
            if pred_label in true_labels:
                is_correct = 1
            # 2. So khớp gần đúng (Fuzzy Matching) quét qua từng bệnh
            else:
                for t_label in true_labels:
                    if pred_label in SIMILAR_DISEASES and t_label in SIMILAR_DISEASES[pred_label]:
                        is_correct = 1
                        print(f"(⚠️ Chấp nhận gần đúng: {pred_label} ~ {t_label})", end=" ")
                        break # Chỉ cần trúng 1 bệnh là tính điểm đúng luôn
            
            y_true_cls.append(1) 
            y_pred_cls.append(is_correct)
            
            # C. Đánh giá Language
            gen_report = result["report"]
            true_report = item["true_report"]
            
            scores = calculate_text_metrics(true_report, gen_report)
            for k, v in scores.items():
                text_metrics_sum[k] += v
                
            print(f"✅ Xong. (Vision: {'Đúng' if is_correct else 'Sai'} | BLEU: {scores['BLEU']:.4f})")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Lỗi ngoại lệ: {str(e)}")

    # --- 3. TỔNG HỢP KẾT QUẢ ---
    if success_count == 0:
        print("\n❌ Không có mẫu nào chạy thành công. Hãy kiểm tra lại đường dẫn ảnh.")
        return

    print("\n" + "="*60)
    print("📊 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (EVALUATION REPORT)")
    print("="*60)
    print(f"Số lượng mẫu đánh giá thành công: {success_count}/{total_samples}")

    # 3.1. Vision Metrics
    vision_acc = sum(y_pred_cls) / len(y_pred_cls)
    print(f"\n[1] MÔ HÌNH THỊ GIÁC (VISION - DENSENET)")
    print(f"   ► Top-1 Accuracy (Bao gồm gần đúng): {vision_acc:.2%}")

    # 3.2. Language Metrics
    avg_bleu = text_metrics_sum["BLEU"] / success_count
    avg_rouge1 = text_metrics_sum["ROUGE-1"] / success_count
    avg_rougeL = text_metrics_sum["ROUGE-L"] / success_count
    
    print(f"\n[2] MÔ HÌNH NGÔN NGỮ (LANGUAGE - BIOGPT)")
    print(f"   ► BLEU Score    : {avg_bleu:.4f}  (Độ chính xác từ vựng)")
    print(f"   ► ROUGE-1       : {avg_rouge1:.4f} (Độ phủ từ đơn)")
    print(f"   ► ROUGE-L       : {avg_rougeL:.4f} (Độ trôi chảy/Cấu trúc câu)")
    print("="*60)

if __name__ == "__main__":
    # 1. Load toàn bộ dữ liệu
    all_data = load_and_process_data()
    
    if not all_data:
        print("❌ Không tìm thấy dữ liệu. Hãy chạy 'python prepare_data.py' để kiểm tra trước.")
        import sys
        sys.exit()

    # --- CẤU HÌNH CHẠY TIẾP THEO ---
    start_index = 51           # Bắt đầu từ mẫu thứ 51 (index 50 trong lập trình)
    batch_size = 50            # Số lượng mẫu muốn chạy
    end_index = start_index + batch_size

    # Kiểm tra tổng số ảnh
    total_images = len(all_data)
    
    # Xử lý trường hợp hết ảnh
    if start_index >= total_images:
        print(f"⚠️ Dữ liệu chỉ có {total_images} ảnh. Không thể bắt đầu từ index {start_index}.")
    else:
        # Cắt dữ liệu từ 50 -> 100 (hoặc hết nếu không đủ 100)
        real_end_index = min(end_index, total_images)
        subset_data = all_data[start_index:real_end_index]
        
        print(f"\n📋 Tổng dữ liệu kho: {total_images} ảnh.")
        print(f"🚀 Đang chạy đánh giá đợt 2: Từ mẫu {start_index + 1} đến {real_end_index}...")
        
        # 3. Chạy
        run_evaluation(subset_data)