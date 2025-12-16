import os
import shutil
import uuid
import torch

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pydub import AudioSegment

# ---------- تنظیمات ----------
BASE_DIR = "outputs"
SEGMENTS_DIR = os.path.join(BASE_DIR, "segments")
MERGED_DIR = os.path.join(BASE_DIR, "merged")

os.makedirs(SEGMENTS_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- بارگذاری pipeline (یک بار) ----------
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token="YOUR_hugingface_key"
)
pipeline.to(DEVICE)

# ---------- FastAPI ----------
app = FastAPI(title="Speaker Diarization API")

@app.post("/diarize")
async def diarize_audio(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())

    input_path = f"{request_id}_{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    audio = AudioSegment.from_file(input_path)

    with ProgressHook() as hook:
        diarization = pipeline(input_path, hook=hook)

    speaker_segments = {}
    segment_files = []

    # ---------- برش و ذخیره ----------
    for turn, speaker in diarization.speaker_diarization:
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)

        segment = audio[start_ms:end_ms]

        speaker_segments.setdefault(speaker, []).append(segment)

        seg_path = os.path.join(
            SEGMENTS_DIR,
            f"{speaker}_{start_ms}_{end_ms}.wav"
        )
        segment.export(seg_path, format="wav")

        segment_files.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
            "file": seg_path
        })

    # ---------- تشخیص گوینده اول ----------
    first_speaker = min(
        diarization.speaker_diarization,
        key=lambda x: x[0].start
    )[1]

    # ---------- ادغام صدای گوینده اول ----------
    merged_audio = AudioSegment.empty()
    for seg in speaker_segments[first_speaker]:
        merged_audio += seg

    merged_path = os.path.join(
        MERGED_DIR,
        f"{first_speaker}_ALL.wav"
    )
    merged_audio.export(merged_path, format="wav")

    return JSONResponse({
        "request_id": request_id,
        "first_speaker": first_speaker,
        "merged_file": merged_path,
        "segments": segment_files
    })
