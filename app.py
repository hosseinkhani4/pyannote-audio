from fastapi import FastAPI, UploadFile, File
import shutil
import os
from model.diarization import pipeline

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/diarize")
async def diarize_audio(file: UploadFile = File(...)):
    # 1. ذخیره فایل
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. اجرای diarization
    diarization = pipeline(file_path)

    # 3. تبدیل خروجی به JSON
    results = []
    for turn, speaker in diarization.speaker_diarization:
        results.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": f"speaker_{speaker}"
        })

    return {
        "file": file.filename,
        "segments": results
    }
