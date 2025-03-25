import os
import io
import gc
import torch
import whisperx
import soundfile as sf
from typing import List, Optional, Dict, Any
from functools import lru_cache
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

app = FastAPI(title="WhisperX API", description="API for audio transcription using WhisperX")

# Environment variables
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")
DEVICE = os.environ.get("DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")
MODEL_DIR = os.environ.get("MODEL_DIR", None)
MODEL_SIZE = os.environ.get("MODEL_SIZE", "small")
MODEL_CACHE_SIZE = int(os.environ.get("MODEL_CACHE_SIZE", "3"))

# Pydantic models for request validation
class TranscriptionRequest(BaseModel):
    batch_size: int = Field(default=16, description="Batch size for transcription. Reduce if low on GPU memory")
    diarize: bool = Field(default=False, description="Whether to perform speaker diarization")
    align_words: bool = Field(default=False, description="Whether to align words")
    min_speakers: Optional[int] = Field(default=None, description="Minimum number of speakers for diarization")
    max_speakers: Optional[int] = Field(default=None, description="Maximum number of speakers for diarization")
    allowed_languages: Optional[List[str]] = Field(default=None, description="List of allowed languages for transcription")

# Model caches
@lru_cache(maxsize=MODEL_CACHE_SIZE)
def get_transcription_model():
    return whisperx.load_model(
        MODEL_SIZE,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        download_root=MODEL_DIR
    )

@lru_cache(maxsize=MODEL_CACHE_SIZE)
def get_alignment_model(language_code):
    return whisperx.load_align_model(language_code=language_code, device=DEVICE)

@lru_cache(maxsize=1)
def get_diarization_model():
    if not HF_AUTH_TOKEN:
        raise ValueError("HF_TOKEN environment variable is required for diarization")
    return whisperx.DiarizationPipeline(use_auth_token=HF_AUTH_TOKEN, device=DEVICE)

def load_audio_from_bytes(audio_bytes):
    """Load audio from bytes using soundfile"""
    with io.BytesIO(audio_bytes) as audio_buffer:
        audio_data, sample_rate = sf.read(audio_buffer)
        # Convert to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)
        return audio_data.astype(np.float32), sample_rate

@app.post("/transcribe", response_model=Dict[str, Any])
async def transcribe(
    file: UploadFile = File(...),
    request: TranscriptionRequest = Depends()
):
    """
    Transcribe audio file using WhisperX

    The audio file is processed in memory without saving to disk.
    """
    try:
        # Load audio from bytes
        audio_bytes = await file.read()
        audio_data, _ = load_audio_from_bytes(audio_bytes)

        # Get transcription model
        model = get_transcription_model()

        # Transcribe audio
        result = model.transcribe(
            audio_data,
            batch_size=request.batch_size
        )

        # Filter language if allowed languages are specified
        if request.allowed_languages and result["language"] not in request.allowed_languages:
            raise HTTPException(
                status_code=400,
                detail=f"Detected language {result['language']} is not in allowed languages: {request.allowed_languages}"
            )

        # Align words if requested
        if request.align_words:
            try:
                model_a, metadata = get_alignment_model(result["language"])
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio_data,
                    DEVICE,
                    return_char_alignments=False
                )
            except Exception as e:
                print(f"Warning: Word alignment failed: {str(e)}")
                # Continue with unaligned result

        # Perform diarization if requested
        if request.diarize:
            try:
                if not HF_AUTH_TOKEN:
                    raise HTTPException(
                        status_code=400,
                        detail="HF_TOKEN environment variable is required for diarization"
                    )

                diarize_model = get_diarization_model()

                # Add min/max number of speakers if specified
                diarize_kwargs = {}
                if request.min_speakers is not None:
                    diarize_kwargs["min_speakers"] = request.min_speakers
                if request.max_speakers is not None:
                    diarize_kwargs["max_speakers"] = request.max_speakers

                diarize_segments = diarize_model(audio_data, **diarize_kwargs)

                # Assign speaker labels to words
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                print(f"Warning: Diarization failed: {str(e)}")
                # Continue with non-diarized result

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gc", response_model=Dict[str, str])
async def garbage_collect():
    """
    Clear model caches and run garbage collection
    """
    get_transcription_model.cache_clear()
    get_alignment_model.cache_clear()
    get_diarization_model.cache_clear()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"status": "cleaned"}

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

