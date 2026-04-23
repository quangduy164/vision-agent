# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from agents.orchestrator import MedicalAgentOrchestrator

app = FastAPI(title="AI Medical Vision Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve ảnh output (heatmap, result)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Serve React build (nếu đã build)
FRONTEND_BUILD = "frontend/build"
if os.path.exists(FRONTEND_BUILD):
    app.mount("/static", StaticFiles(directory=f"{FRONTEND_BUILD}/static"), name="static")

print("⏳ Starting Server & Loading Multi-Agent System...")
ai_agent = MedicalAgentOrchestrator()


@app.get("/")
async def serve_frontend():
    index = f"{FRONTEND_BUILD}/index.html"
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "API is running. Frontend not built yet."}

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