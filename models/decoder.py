# models/decoder.py
import torch
from transformers import BioGptTokenizer, BioGptForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BioGPTDecoder:
    def __init__(self):
        print(f"🧠 Loading Language Decoder (BioGPT) on {DEVICE}...")
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        self.model.to(DEVICE)
        self.model.eval()
        print("✅ BioGPT loaded.")

    def generate_report(self, prompt: str) -> str:
        """
        Sinh báo cáo từ prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                min_length=20,
                max_length=100,        # Độ dài báo cáo tối đa
                num_beams=5,           # Beam Search cho câu mượt hơn
                no_repeat_ngram_size=2,# Tránh lặp từ
                early_stopping=True
            )
            
        report = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return report