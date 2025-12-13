# models/diarization.py
import torch
from pyannote.audio import Pipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token="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"
)

pipeline.to(DEVICE)
