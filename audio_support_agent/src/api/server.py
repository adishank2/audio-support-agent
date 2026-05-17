"""
FastAPI Server for Audio Customer Support Agent

This module provides REST API endpoints for testing the audio support pipeline.
Students can use this server to test their implementations via HTTP requests.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import logging
import os
import base64

from src.pipeline import AudioSupportPipeline, create_pipeline, PipelineConfig


class TextRequest(BaseModel):
    """Request model for text-based queries."""
    text: str
    parameters: Optional[Dict[str, Any]] = {}


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    components: Dict[str, bool]
    message: str


class TextResponse(BaseModel):
    """Response model for text queries."""
    response_text: str
    audio_available: bool
    processing_time_ms: int


class TranscriptDataModel(BaseModel):
    """Transcript data for audio interactions."""
    user_input: str
    agent_response: str


class EnhancedAudioResponse(BaseModel):
    """Enhanced response model for audio queries with transcript."""
    success: bool
    audio_response: str  # base64 encoded
    transcript: TranscriptDataModel
    processing_time_ms: int


app = FastAPI(
    title="Audio Customer Support Agent API",
    description="REST API for testing the STT -> LLM -> TTS pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[AudioSupportPipeline] = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    global pipeline
    
    try:
        logger.info("Starting Audio Support Agent API server...")
        
        stt_config = {
            "api_key": os.getenv("DEEPGRAM_API_KEY", ""),
            "model": "nova-2"
        }
        
        llm_config = {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        }
        
        tts_config = {
            "api_key": os.getenv("ELEVENLABS_API_KEY", ""),
            "voice_id": "21m00Tcm4TlvDq8ikWAM"
        }
        
        pipeline = await create_pipeline(stt_config, llm_config, tts_config)
        logger.info("Pipeline initialized successfully on server startup!")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        # Don't raise here to allow server to start for debugging


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup pipeline resources on server shutdown."""
    global pipeline
    
    if pipeline:
        logger.info("Shutting down pipeline...")
        await pipeline.cleanup()
        pipeline = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Audio Customer Support Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of all pipeline components.
    """
    global pipeline
    
    if not pipeline:
        return HealthResponse(
            status="unhealthy",
            components={
                "pipeline_initialized": False,
                "stt_ready": False,
                "llm_ready": False,
                "tts_ready": False
            },
            message="Pipeline not initialized"
        )
    
    try:
        components = await pipeline.health_check()
        all_healthy = all(components.values())
        
        return HealthResponse(
            status="healthy" if all_healthy else "unhealthy",
            components=components,
            message="All components ready" if all_healthy else "Some components not ready"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="error",
            components={},
            message=f"Health check failed: {str(e)}"
        )


@app.post("/chat/text", response_model=TextResponse)
async def chat_text(request: TextRequest):
    """
    Process text query through the LLM agent.
    
    This endpoint allows testing the LLM component without audio processing.
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        import time
        start_time = time.time()
        
        response_text, response_audio = await pipeline.process_text(
            request.text, 
            **request.parameters
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return TextResponse(
            response_text=response_text,
            audio_available=len(response_audio) > 0,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Text processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/audio")
async def chat_audio(audio: UploadFile = File(...)):
    """
    Process audio query through the complete pipeline.
    
    Returns JSON with base64-encoded audio, transcript, and processing time.
    """
    global pipeline
    
    if not pipeline:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "Pipeline not initialized",
                "transcript": {"user_input": None, "agent_response": None},
                "processing_time_ms": 0
            }
        )
    
    try:
        audio_bytes = await audio.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        response_audio, transcript_data, processing_time = await pipeline.process_audio_with_transcript(audio_bytes)
        
        # Encode audio to base64
        audio_b64 = base64.b64encode(response_audio).decode("utf-8")
        
        return JSONResponse(
            content={
                "success": True,
                "audio_response": audio_b64,
                "transcript": {
                    "user_input": transcript_data.user_input,
                    "agent_response": transcript_data.agent_response,
                    "processing_metadata": transcript_data.processing_metadata
                },
                "processing_time_ms": processing_time
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "transcript": {"user_input": None, "agent_response": None},
                "processing_time_ms": 0
            }
        )


@app.get("/chat/audio/{text}")
async def text_to_audio(text: str):
    """
    TODO: Convert text to audio using TTS.
    
    Useful for testing TTS component independently.
    
    Args:
        text: Text to convert to speech
        
    Returns:
        Audio file as bytes
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        if not pipeline.tts:
            raise HTTPException(status_code=503, detail="TTS not available")
        
        audio_bytes = await pipeline.tts.synthesize(text)
        
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.mp3"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/stt")
async def debug_stt(audio: UploadFile = File(...)):
    """
    TODO: Debug endpoint for testing STT component independently.
    
    Args:
        audio: Audio file to transcribe
        
    Returns:
        Transcription result
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        audio_bytes = await audio.read()
        
        if not pipeline.stt:
            raise HTTPException(status_code=503, detail="STT not available")
        
        transcription = await pipeline.stt.transcribe(audio_bytes)
        
        return {"transcription": transcription}
        
    except Exception as e:
        logger.error(f"STT debug failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # TODO: Students can modify these settings for development
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )