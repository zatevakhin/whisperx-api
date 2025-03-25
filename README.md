# WhisperX API Service

A FastAPI-based service for audio transcription, alignment, and speaker diarization using the WhisperX library.

## Features

- Audio transcription with WhisperX
- Word alignment
- Speaker diarization
- Language filtering
- GPU memory optimization with LRU caching and garbage collection
- Audio processing without file storage (in-memory via BytesIO)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_AUTH_TOKEN` | Hugging Face token for accessing diarization models | *Required for diarization* |
| `DEVICE` | Device to run models on (`cuda` or `cpu`) | `cpu` |
| `COMPUTE_TYPE` | Computation precision (`float16` or `int8`) | `int8` |
| `MODEL_DIR` | Directory to save models (optional) | None |
| `WHISPER_MODEL` | WhisperX model size | `small` |
| `MODEL_CACHE_SIZE` | Number of models to keep in LRU cache | `3` |

## API Endpoints

### POST /transcribe

Transcribes an audio file with optional alignment and diarization.

**Request Parameters:**
- `file`: Audio file (multipart/form-data)
- `batch_size`: Batch size for transcription (default: 16)
- `diarize`: Enable speaker diarization (default: false)
- `align_words`: Enable word alignment (default: false)
- `whisper_model`: Model to use for transcription (overrides default)
- `min_speakers`: Minimum number of speakers for diarization (optional)
- `max_speakers`: Maximum number of speakers for diarization (optional)
- `allowed_languages`: Comma-separated list of allowed languages for transcription (optional)

### POST /gc

Clears model caches and runs garbage collection to free up GPU memory.

### GET /health

Health check endpoint.

## Running with Docker

### Build the Docker Image

```bash
docker build -t whisperx-api .
```

### Run the Container

```bash
docker run --gpus all -p 8000:8000 \
  -e HF_AUTH_TOKEN=your_huggingface_token \
  -e DEVICE=cuda \
  -e COMPUTE_TYPE=float16 \
  -e WHISPER_MODEL=large-v2 \
  -e MODEL_CACHE_SIZE=3 \
  whisperx-api
```

## Usage Examples

### Basic Transcription

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.mp3"
```

### Transcription with Word Alignment

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.mp3" \
  -F "align_words=true"
```

### Transcription with Speaker Diarization

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.mp3" \
  -F "diarize=true"
```

### Transcription with Custom Model

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.mp3" \
  -F "whisper_model=large-v2"
```

### Transcription with Language Filtering

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.mp3" \
  -F "allowed_languages=en,fr,de"
```

### Cleaning Up GPU Memory

```bash
curl -X POST "http://localhost:8000/gc"
```

## Python Client Example

```python
import requests

def transcribe_audio(file_path, server_url="http://localhost:8000"):
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {
            "batch_size": 16,
            "diarize": True,  # String form for multipart/form-data
            "align_words": True,  # String form for multipart/form-data
            "whisper_model": "large-v2",  # Override default model
            "min_speakers": 1,
            "max_speakers": 5,
            "allowed_languages": "en,fr,de"  # Comma-separated languages
        }
        response = requests.post(f"{server_url}/transcribe", files=files, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Example usage
result = transcribe_audio("/path/to/audio.mp3")
print(result)
```

## Performance Considerations

- Use `DEVICE=cuda` for GPU acceleration when available
- Adjust `COMPUTE_TYPE` based on your hardware capabilities:
  - `float16`: Better accuracy but requires more memory
  - `int8`: Reduced memory usage with slight accuracy tradeoff
- Set `batch_size` according to your GPU memory (lower for less memory)
- Use the `/gc` endpoint between processing large files to free up memory
- Choose an appropriate `WHISPER_MODEL` size based on accuracy needs and hardware constraints:
  - `tiny`: Fastest, lowest memory usage, lowest accuracy
  - `small`: Good balance for most use cases
  - `medium`: Better accuracy, moderate resource usage
  - `large-v2`: Highest accuracy, highest resource usage


