# evaluate.py
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

import config
from prepare_data import load_and_process_data
from agent import MedicalVisionAgent

# Tải dữ liệu NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab') 
except LookupError:
    print("⬇️ Đang tải dữ liệu NLTK cần thiết...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

def calculate_text_metrics(reference_text, candidate_text):
    """Tính điểm BLEU và ROUGE"""
    if not candidate_text or not reference_text:
        return {"BLEU": 0, "ROUGE-1": 0, "ROUGE-L": 0}

    ref_tokens = nltk.word_tokenize(reference_text.lower())
    cand_tokens = nltk.word_tokenize(candidate_text.lower())

    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return {"BLEU": 0, "ROUGE-1": 0, "ROUGE-L": 0}

    cc = SmoothingFunction()
    bleu_score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=cc.method1)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_text, candidate_text)
    
    return {
        "BLEU": bleu_score,
        "ROUGE-1": rouge_scores['rouge1'].fmeasure,
        "ROUGE-L": rouge_scores['rougeL'].fmeasure
    }

def run_evaluation(test_data):
    agent = MedicalVisionAgent()
    
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
    
    y_pred_cls = []
    text_metrics_sum = {"BLEU": 0, "ROUGE-1": 0, "ROUGE-L": 0}
    
    total_samples = len(test_data)
    print(f"\n🚀 BẮT ĐẦU ĐÁNH GIÁ TRÊN {total_samples} ẢNH...")
    print("="*60)
    
    success_count = 0

    for i, item in enumerate(test_data):
        img_name = item["filename"]
        print(f"📸 [{i+1}/{total_samples}] {img_name}...", end=" ")
        
        try:
            result = agent.analyze(item["image_path"], output_dir="eval_outputs")
            
            if "error" in result:
                print(f"❌ Lỗi Agent: {result['error']}")
                continue

            pred_label = result["diagnosis"]
            true_labels = item["true_labels"] 
            
            is_correct = 0
            if pred_label in true_labels:
                is_correct = 1
            else:
                for t_label in true_labels:
                    if pred_label in SIMILAR_DISEASES and t_label in SIMILAR_DISEASES.get(pred_label, []):
                        is_correct = 1
                        print(f"(⚠️ Chấp nhận gần đúng: {pred_label} ~ {t_label})", end=" ")
                        break 
            
            y_pred_cls.append(is_correct)
            
            gen_report = result["report"]
            true_report = item["true_report"]
            
            scores = calculate_text_metrics(true_report, gen_report)
            for k, v in scores.items():
                text_metrics_sum[k] += v
                
            print(f"✅ Xong. (Vision: {'Đúng' if is_correct else 'Sai'} | BLEU: {scores['BLEU']:.4f})")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")

    if success_count == 0:
        print("\n❌ Không có mẫu nào chạy thành công.")
        return

    print("\n" + "="*60)
    print("📊 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ")
    print("="*60)
    print(f"Số lượng mẫu: {success_count}/{total_samples}")

    vision_acc = sum(y_pred_cls) / len(y_pred_cls) if y_pred_cls else 0
    print(f"\n[1] MÔ HÌNH THỊ GIÁC (VISION - DENSENET)")
    print(f"   ► Accuracy (Bao gồm gần đúng): {vision_acc:.2%}")

    avg_bleu = text_metrics_sum["BLEU"] / success_count
    avg_rouge1 = text_metrics_sum["ROUGE-1"] / success_count
    avg_rougeL = text_metrics_sum["ROUGE-L"] / success_count
    
    print(f"\n[2] MÔ HÌNH NGÔN NGỮ (LANGUAGE - BIOGPT)")
    print(f"   ► BLEU Score : {avg_bleu:.4f}")
    print(f"   ► ROUGE-1    : {avg_rouge1:.4f}")
    print(f"   ► ROUGE-L    : {avg_rougeL:.4f}")
    print("="*60)

if __name__ == "__main__":
    all_data = load_and_process_data()
    
    if not all_data:
        print("❌ Lỗi dữ liệu.")
        import sys
        sys.exit()

    start_idx = config.START_INDEX
    end_idx = start_idx + config.BATCH_SIZE
    total = len(all_data)
    
    if start_idx >= total:
        print(f"⚠️ Dữ liệu chỉ có {total} ảnh.")
    else:
        real_end = min(end_idx, total)
        subset_data = all_data[start_idx:real_end]
        
        print(f"\n📋 Tổng kho: {total} ảnh.")
        print(f"🚀 Đang chạy từ mẫu {start_idx + 1} đến {real_end}...")
        
        run_evaluation(subset_data)