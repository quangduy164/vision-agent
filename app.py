from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
import agent

app = FastAPI(title="Vision Agent API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    model_name: str = "densenet121"
):
    image_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = agent.run(image_path, model_name)

    return {
        "image": file.filename,
        "model": model_name,
        "result": result,
        "warning": "For research support only. Not a medical diagnosis."
    }
