from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
import agent

app = FastAPI(title="Vision Agent API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    model_name: str = "vit_base_nih"
):
    image_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = agent.run(image_path, model_name)

    return JSONResponse({
        "image": file.filename,
        "model": model_name,
        "result": result,
        "warning": "For research support only. Not a medical diagnosis."
    })
