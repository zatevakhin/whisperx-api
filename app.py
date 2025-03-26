import os
import io
import gc
import torch
import whisperx
import soundfile as sf
from typing import Optional, Dict, Any
from functools import lru_cache
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import numpy as np
from typing_extensions import TypedDict

# NOTE: https://github.com/pyannote/pyannote-audio/issues/1370
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

app = FastAPI(title="WhisperX API", description="API for audio transcription using WhisperX")

# Environment variables
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", None)
DEVICE = os.environ.get("DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")
MODEL_DIR = os.environ.get("MODEL_DIR", None)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
MODEL_CACHE_SIZE = int(os.environ.get("MODEL_CACHE_SIZE", "3"))

class ResponseData(TypedDict):
    transcription: Optional[TypedDict] = None
    align: Optional[TypedDict] = None
    diarized: dict = None
    words: dict = None

# Model caches
@lru_cache(maxsize=MODEL_CACHE_SIZE)
def get_transcription_model(model: Optional[str] = None):
    print(f"Loading model =: {model}, {WHISPER_MODEL}")
    model = model if model is not None else WHISPER_MODEL

    print(f"Loading model: {model}")
    return whisperx.load_model(
        model, DEVICE, compute_type=COMPUTE_TYPE, download_root=MODEL_DIR
    )

@lru_cache(maxsize=MODEL_CACHE_SIZE)
def get_alignment_model(language_code):
    return whisperx.load_align_model(language_code=language_code, device=DEVICE)

@lru_cache(maxsize=1)
def get_diarization_model():
    if not HF_AUTH_TOKEN:
        raise ValueError("HF_AUTH_TOKEN environment variable is required for diarization")
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
    batch_size: int = Form(16, description="Batch size for transcription. Reduce if low on GPU memory"),
    diarize: bool = Form(False, description="Whether to perform speaker diarization"),
    align_words: bool = Form(False, description="Whether to align words"),
    whisper_model: Optional[str] = Form(default=None, description=f"Model used for transcription instead of (default: ${WHISPER_MODEL})"),
    min_speakers: Optional[int] = Form(None, description="Minimum number of speakers for diarization"),
    max_speakers: Optional[int] = Form(None, description="Maximum number of speakers for diarization"),
    allowed_languages: Optional[str] = Form(None, description="Comma-separated list of allowed languages for transcription")
):
    """
    Transcribe audio file using WhisperX

    The audio file is processed in memory without saving to disk.
    """
    try:
        # Load audio from bytes
        audio_bytes = await file.read()
        audio_data, _ = load_audio_from_bytes(audio_bytes)

        model = get_transcription_model(whisper_model)

        result = ResponseData()

        result["transcription"] = model.transcribe(
            audio_data,
            batch_size=batch_size
        )

        if allowed_languages and result["transcription"]["language"] not in allowed_languages:
            raise HTTPException(
                status_code=400,
                detail=f"Detected language {result['transcription']['language']} is not in allowed languages: {allowed_languages}"
            )

        if align_words:
            try:
                model_a, metadata = get_alignment_model(result["transcription"]["language"])
                result["align"] = whisperx.align(
                    result["transcription"]["segments"],
                    model_a,
                    metadata,
                    audio_data,
                    DEVICE,
                    return_char_alignments=False
                )

            except Exception as e:
                print(f"Warning: Word alignment failed: {str(e)}")
                # Continue with unaligned result

        if diarize:
            try:
                if not HF_AUTH_TOKEN:
                    raise HTTPException(
                        status_code=400,
                        detail="HF_TOKEN environment variable is required for diarization"
                    )

                diarize_model = get_diarization_model()

                diarize_kwargs = {}
                if min_speakers is not None:
                    diarize_kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    diarize_kwargs["max_speakers"] = max_speakers

                diarized = diarize_model(audio_data, **diarize_kwargs)
                result["diarized"] = diarized.to_dict(orient='records')
                if align_words:
                    result["words"] = whisperx.assign_word_speakers(diarized, result["align"])

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

