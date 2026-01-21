#!/usr/bin/env python3
"""
Speech API for LangCoach

FastAPI-based REST API providing endpoints for:
- /transcribe: Speech-to-Text using Whisper
- /synthesize: Text-to-Speech using Orpheus with multi-speaker support

Usage:
    # Start the API server
    python -m src.api.speech_api

    # Or with uvicorn directly
    uvicorn src.api.speech_api:app --host 0.0.0.0 --port 8600
"""

import io
import os
import logging
import tempfile
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Lazy imports for services
_tts_service = None
_stt_service = None


def get_tts_service():
    """Lazy load TTS service."""
    global _tts_service
    if _tts_service is None:
        from src.tts.service import initialize_tts_service
        _tts_service = initialize_tts_service()
    return _tts_service


def get_stt_service():
    """Lazy load STT service."""
    global _stt_service
    if _stt_service is None:
        from src.stt.service import initialize_stt_service
        _stt_service = initialize_stt_service()
    return _stt_service


# Request/Response models
class SynthesizeRequest(BaseModel):
    """Request model for text-to-speech synthesis."""
    text: str
    speaker: str = "Ceylia"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    fast_mode: bool = False  # Use Edge-TTS for faster response


class TranscribeResponse(BaseModel):
    """Response model for speech-to-text transcription."""
    text: str
    language: str


class SpeakersResponse(BaseModel):
    """Response model for listing supported speakers."""
    speakers: List[str]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    tts_initialized: bool
    stt_initialized: bool


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Starting Speech API...")

    # Optionally pre-load models on startup
    preload = os.getenv("PRELOAD_MODELS", "false").lower() == "true"
    if preload:
        logger.info("Pre-loading models...")
        try:
            get_tts_service()
            logger.info("TTS service loaded")
        except Exception as e:
            logger.error(f"Failed to load TTS service: {e}")

        try:
            get_stt_service()
            logger.info("STT service loaded")
        except Exception as e:
            logger.error(f"Failed to load STT service: {e}")

    yield

    logger.info("Shutting down Speech API...")


# Create FastAPI app
app = FastAPI(
    title="LangCoach Speech API",
    description="Speech-to-Text and Text-to-Speech API for LangCoach",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and service status."""
    global _tts_service, _stt_service

    return HealthResponse(
        status="healthy",
        tts_initialized=_tts_service is not None and _tts_service.is_initialized,
        stt_initialized=_stt_service is not None and _stt_service.is_initialized
    )


@app.get("/speakers", response_model=SpeakersResponse)
async def list_speakers():
    """List supported TTS speakers."""
    try:
        service = get_tts_service()
        speakers = service.get_supported_speakers()
        return SpeakersResponse(speakers=speakers)
    except Exception as e:
        logger.error(f"Error listing speakers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from text.

    Args:
        request: SynthesizeRequest with text and speaker

    Returns:
        WAV audio file as streaming response
    """
    try:
        # Fast mode uses Edge-TTS (Microsoft Azure) for near-instant response
        if request.fast_mode:
            return await synthesize_fast(request.text, request.speaker)
        
        service = get_tts_service()

        result = service.synthesize(
            text=request.text,
            speaker=request.speaker,
            temperature=request.temperature,
            top_p=request.top_p
        )

        # Convert to WAV bytes
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, result["audio"], result["sample_rate"], format="WAV")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="speech_{request.speaker}.wav"'
            }
        )

    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Edge-TTS voice mapping for fast mode
EDGE_TTS_VOICES = {
    "Ceylia": "en-US-JennyNeural",  # Female, natural
    "Tifa": "en-US-AriaNeural",     # Female, expressive
    "default": "en-US-JennyNeural"
}


async def synthesize_fast(text: str, speaker: str) -> StreamingResponse:
    """Fast TTS using Edge-TTS (Microsoft Azure)."""
    try:
        import edge_tts
        
        voice = EDGE_TTS_VOICES.get(speaker, EDGE_TTS_VOICES["default"])
        logger.info(f"[Fast TTS] Using Edge-TTS voice: {voice} for speaker: {speaker}")
        
        communicate = edge_tts.Communicate(text, voice)
        buffer = io.BytesIO()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])
        
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="audio/mpeg",  # Edge-TTS outputs MP3
            headers={
                "Content-Disposition": f'attachment; filename="speech_{speaker}.mp3"'
            }
        )
    except ImportError:
        logger.error("edge-tts not installed. Install with: pip install edge-tts")
        raise HTTPException(status_code=500, detail="edge-tts not installed")
    except Exception as e:
        logger.error(f"Edge-TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/json")
async def synthesize_json(request: SynthesizeRequest):
    """
    Synthesize speech and return audio as base64-encoded JSON.

    Args:
        request: SynthesizeRequest with text and speaker

    Returns:
        JSON with base64-encoded audio
    """
    import base64

    try:
        # Fast mode uses Edge-TTS
        if request.fast_mode:
            return await synthesize_fast_json(request.text, request.speaker)
        
        service = get_tts_service()

        result = service.synthesize(
            text=request.text,
            speaker=request.speaker,
            temperature=request.temperature,
            top_p=request.top_p
        )

        # Convert to WAV bytes
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, result["audio"], result["sample_rate"], format="WAV")
        audio_bytes = buffer.getvalue()

        return JSONResponse({
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "sample_rate": result["sample_rate"],
            "speaker": result["speaker"],
            "text": result["text"],
            "format": "wav"
        })

    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def synthesize_fast_json(text: str, speaker: str) -> JSONResponse:
    """Fast TTS using Edge-TTS, returning JSON with base64 audio."""
    import base64
    
    try:
        import edge_tts
        
        voice = EDGE_TTS_VOICES.get(speaker, EDGE_TTS_VOICES["default"])
        logger.info(f"[Fast TTS JSON] Using Edge-TTS voice: {voice} for speaker: {speaker}")
        
        communicate = edge_tts.Communicate(text, voice)
        buffer = io.BytesIO()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])
        
        audio_bytes = buffer.getvalue()
        
        return JSONResponse({
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "sample_rate": 24000,  # Edge-TTS uses 24kHz
            "speaker": speaker,
            "text": text,
            "format": "mp3"
        })
    except ImportError:
        logger.error("edge-tts not installed")
        raise HTTPException(status_code=500, detail="edge-tts not installed")
    except Exception as e:
        logger.error(f"Edge-TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    task: Optional[str] = Form("transcribe")
):
    """
    Transcribe audio to text.

    Args:
        audio: Audio file (WAV, MP3, etc.)
        language: Language code (optional, auto-detect if not specified)
        task: "transcribe" or "translate"

    Returns:
        TranscribeResponse with transcribed text
    """
    import librosa

    try:
        service = get_stt_service()

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Load and transcribe
            audio_data, sr = librosa.load(tmp_path, sr=16000)

            result = service.transcribe(
                audio=audio_data,
                sample_rate=sr,
                language=language,
                task=task
            )

            return TranscribeResponse(
                text=result["text"],
                language=result["language"]
            )

        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe/json")
async def transcribe_json(
    audio_base64: str = Form(...),
    language: Optional[str] = Form(None),
    task: Optional[str] = Form("transcribe")
):
    """
    Transcribe base64-encoded audio to text.

    Args:
        audio_base64: Base64-encoded audio data
        language: Language code (optional)
        task: "transcribe" or "translate"

    Returns:
        JSON with transcribed text
    """
    import base64
    import librosa

    try:
        service = get_stt_service()

        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_base64)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            # Load and transcribe
            audio_data, sr = librosa.load(tmp_path, sr=16000)

            result = service.transcribe(
                audio=audio_data,
                sample_rate=sr,
                language=language,
                task=task
            )

            return JSONResponse({
                "text": result["text"],
                "language": result["language"]
            })

        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the API server."""
    import uvicorn

    host = os.getenv("SPEECH_API_HOST", "0.0.0.0")
    port = int(os.getenv("SPEECH_API_PORT", "8600"))

    logger.info(f"Starting Speech API on {host}:{port}")

    uvicorn.run(
        "src.api.speech_api:app",
        host=host,
        port=port,
        reload=False
    )


if __name__ == "__main__":
    main()
