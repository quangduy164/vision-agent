# models/decoder.py
import torch
from transformers import BioGptTokenizer, BioGptForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Nếu True: trả thẳng report từ bridge (không qua BioGPT generate)
# Tắt khi bạn có GPU mạnh hoặc đã fine-tune BioGPT
USE_TEMPLATE_ONLY = True

class BioGPTDecoder:
    def __init__(self):
        if USE_TEMPLATE_ONLY:
            print("📝 Language Decoder: Template-only mode (no BioGPT generation).")
            self.model = None
            self.tokenizer = None
            return

        print(f"🧠 Loading Language Decoder (BioGPT) on {DEVICE}...")
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        self.model.to(DEVICE)
        self.model.eval()
        print("✅ BioGPT loaded.")

    def generate_report(self, prompt: str) -> str:
        # Template-only: trả thẳng prompt đã được bridge sinh ra
        if USE_TEMPLATE_ONLY or self.model is None:
            return prompt

        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                min_new_tokens=30,
                max_new_tokens=80,       # Chỉ sinh thêm phần mới, không tính prompt
                num_beams=8,
                length_penalty=1.2,      # Khuyến khích câu dài hơn
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        # Chỉ decode phần BioGPT sinh thêm (bỏ phần prompt)
        new_tokens = output_ids[0][input_len:]
        generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Ghép prompt gốc + phần sinh thêm
        return f"{prompt} {generated}".strip()
