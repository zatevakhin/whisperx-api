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

from typing import Dict, List, Optional, Any
from typing_extensions import TypedDict
from collections import defaultdict

from typing import Dict, List, Optional, Any
from typing_extensions import TypedDict
from collections import defaultdict

class Word(TypedDict):
    word: str
    start: float
    end: float

class Speaker(TypedDict):
    id: str
    label: str
    total_time: float

class Segment(TypedDict):
    text: str
    start: float
    end: float
    speaker: Optional[str]
    words: List[Word]

class FormattedTranscript(TypedDict):
    transcript: str
    language: str
    duration: float
    speakers: List[Speaker]
    segments: List[Segment]

class TranscriptFormatter:
    @staticmethod
    def format(
        transcription: Dict[str, Any],
        aligned_segments: Optional[Dict[str, Any]] = None,
        diarized_segments: Optional[List[Dict[str, Any]]] = None,
        word_speakers: Optional[Dict[str, Any]] = None
    ) -> FormattedTranscript:
        """
        Format the WhisperX output into a cleaner structure

        Args:
            transcription: The raw transcription output
            aligned_segments: Word-aligned segments if available
            diarized_segments: Speaker diarization information if available
            word_speakers: Word-level speaker assignments if available

        Returns:
            A formatted transcript in a cleaner structure
        """
        # Initialize result
        result: FormattedTranscript = {
            "transcript": "",
            "language": transcription.get("language", ""),
            "duration": 0.0,
            "speakers": [],
            "segments": []
        }

        # Get full transcript text
        if "segments" in transcription:
            full_text = " ".join([segment.get("text", "").strip() for segment in transcription["segments"]])
            result["transcript"] = full_text

            # Set duration to the end time of the last segment
            if transcription["segments"]:
                result["duration"] = transcription["segments"][-1].get("end", 0.0)

        # Process speakers if diarization is available
        speakers_dict = {}
        if diarized_segments:
            speaker_times = defaultdict(float)
            speaker_labels = {}

            for segment in diarized_segments:
                speaker_id = segment.get("speaker", "")
                label = segment.get("label", "")
                start_time = segment.get("start", 0.0)
                end_time = segment.get("end", 0.0)

                if speaker_id:
                    duration = end_time - start_time
                    speaker_times[speaker_id] += duration
                    speaker_labels[speaker_id] = label

            for speaker_id, total_time in speaker_times.items():
                speakers_dict[speaker_id] = {
                    "id": speaker_id,
                    "label": speaker_labels.get(speaker_id, ""),
                    "total_time": round(total_time, 3)
                }

            result["speakers"] = list(speakers_dict.values())

        # Use aligned_segments if available, otherwise use transcription segments
        source_segments = aligned_segments if aligned_segments and "segments" in aligned_segments else transcription

        if "segments" in source_segments:
            for segment in source_segments["segments"]:
                formatted_segment: Segment = {
                    "text": segment.get("text", "").strip(),
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                    "speaker": None,
                    "words": []
                }

                # Add words if available
                if "words" in segment:
                    for word in segment["words"]:
                        formatted_segment["words"].append({
                            "word": word.get("word", ""),
                            "start": word.get("start", 0.0),
                            "end": word.get("end", 0.0)
                        })

                # Add speaker information if word_speakers is available
                if word_speakers and "segments" in word_speakers:
                    for ws_segment in word_speakers["segments"]:
                        if (ws_segment.get("start") == segment.get("start") and
                            ws_segment.get("end") == segment.get("end")):
                            formatted_segment["speaker"] = ws_segment.get("speaker", None)
                            break
                # If no speaker was assigned and diarized segments are available, try to assign a speaker
                if formatted_segment["speaker"] is None and diarized_segments:
                    segment_start = formatted_segment["start"]
                    segment_end = formatted_segment["end"]

                    # Find the diarized segment with the most overlap
                    best_overlap = 0
                    best_speaker = None

                    for d_segment in diarized_segments:
                        d_start = d_segment.get("start", 0.0)
                        d_end = d_segment.get("end", 0.0)

                        # Calculate overlap
                        overlap_start = max(segment_start, d_start)
                        overlap_end = min(segment_end, d_end)
                        overlap = max(0, overlap_end - overlap_start)

                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_speaker = d_segment.get("speaker", None)

                    if best_speaker:
                        formatted_segment["speaker"] = best_speaker

                result["segments"].append(formatted_segment)

        return result

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

        aligned_segments = None
        diarized = None
        word_speakers = None
        transcription = model.transcribe(
            audio_data,
            batch_size=batch_size
        )

        if allowed_languages and transcription["language"] not in allowed_languages:
            raise HTTPException(
                status_code=400,
                detail=f"Detected language {transcription['language']} is not in allowed languages: {allowed_languages}"
            )

        if align_words:
            try:
                model_a, metadata = get_alignment_model(transcription["language"])
                aligned_segments = whisperx.align(
                    transcription["segments"],
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

                diarized_segments = diarize_model(audio_data, **diarize_kwargs)
                diarized = diarized_segments.to_dict(orient='records')

                if align_words:
                    word_speakers = whisperx.assign_word_speakers(diarized_segments, aligned_segments)

            except Exception as e:
                print(f"Warning: Diarization failed: {str(e)}")
                # Continue with non-diarized result

        # Format the output using TranscriptFormatter
        formatted_transcript = TranscriptFormatter.format(
            transcription=transcription,
            aligned_segments=aligned_segments,
            diarized_segments=diarized,
            word_speakers=word_speakers
        )

        return formatted_transcript

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

