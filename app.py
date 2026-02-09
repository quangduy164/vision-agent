# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
from agent import MedicalVisionAgent

app = FastAPI(title="AI Medical Vision Agent API")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Khởi tạo Agent 1 lần duy nhất (Load Model tốn khoảng 10-20s)
print("⏳ Starting Server & Loading Models...")
ai_agent = MedicalVisionAgent()

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    # 1. Lưu ảnh
    image_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2. Gọi Agent phân tích (Pipeline: Vision -> Bridge -> Text)
    try:
        result = ai_agent.analyze(image_path, OUTPUT_DIR)
        return JSONResponse({
            "success": True,
            "data": result
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)