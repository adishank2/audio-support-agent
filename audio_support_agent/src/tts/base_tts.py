"""
Base Text-to-Speech (TTS) Interface

This module defines the abstract base class for Text-to-Speech implementations.
Students should implement the concrete TTS class by inheriting from this base class.

Recommended implementation: ElevenLabs API (free tier available)
Alternative options: OpenTTS, gTTS, Azure Speech Services, or any other TTS service
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import io


class BaseTTS(ABC):
    """
    Abstract base class for Text-to-Speech implementations.
    
    This class defines the interface that all TTS implementations must follow.
    Students should inherit from this class and implement the abstract methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TTS service.
        
        Args:
            config: Configuration dictionary containing API keys, voice settings, etc.
                   Example: {"api_key": "your_api_key", "voice_id": "voice_id", "model": "eleven_turbo_v2"}
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the TTS service (setup API clients, load models, etc.).
        This method should be called before using the TTS service.
        
        Raises:
            Exception: If initialization fails
        """
        pass
    
    @abstractmethod
    async def synthesize(self, text: str, **kwargs) -> bytes:
        """
        Convert text to speech audio bytes.
        
        Args:
            text: Text to convert to speech
            **kwargs: Additional parameters specific to the TTS implementation
                     (e.g., voice_id, speed, pitch, format)
        
        Returns:
            bytes: Audio data as bytes (typically MP3 or WAV format)
            
        Raises:
            Exception: If synthesis fails
        """
        pass
    
    @abstractmethod
    async def synthesize_stream(self, text: str, **kwargs) -> io.BytesIO:
        """
        Convert text to speech with streaming support.
        
        Args:
            text: Text to convert to speech
            **kwargs: Additional parameters for streaming
        
        Returns:
            io.BytesIO: Streaming audio data
            
        Raises:
            Exception: If streaming synthesis fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup resources (close connections, free memory, etc.).
        This method should be called when the TTS service is no longer needed.
        """
        pass
    
    def is_ready(self) -> bool:
        """
        Check if the TTS service is ready to use.
        
        Returns:
            bool: True if ready, False otherwise
        """
        return self.is_initialized


class TTSService(BaseTTS):
    """
    Generic TTS implementation template.
    
    Students should complete this implementation using their chosen TTS service or pretrained model.
    
    API-based options:
    - ElevenLabs API (free tier, high quality): pip install elevenlabs
    - OpenAI TTS API (high quality): included in openai package
    - Azure Speech Services: pip install azure-cognitiveservices-speech
    - Google Cloud Text-to-Speech: pip install google-cloud-texttospeech
    - Amazon Polly: pip install boto3
    
    Pretrained model options (local inference):
    - Coqui TTS: pip install TTS (various pretrained models: Tacotron2, VITS, etc.)
    - Parler TTS: pip install parler-tts (Hugging Face pretrained models)
    - Bark: pip install bark (Suno's generative audio model)
    - Edge TTS: pip install edge-tts (free Microsoft voices, no training needed)
    - Festival: pip install festival (classic speech synthesis)
    - eSpeak: pip install espeak (lightweight, many languages)
    - Piper: pip install piper-tts (fast neural TTS)
    
    Input: text (str) - Text to convert to speech
    Output: audio_bytes (bytes) - Audio data (typically MP3 or WAV)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = None
        self.voice_id = None
        self.model = None
        # TODO: Initialize your chosen TTS client/model
        # API-based examples:
        # - For ElevenLabs: from elevenlabs import ElevenLabs
        # - For OpenAI: from openai import OpenAI
        # Pretrained model examples:
        # - For Coqui TTS: from TTS.api import TTS
        # - For Parler TTS: from parler_tts import ParlerTTSForConditionalGeneration
        # - For Bark: import bark
        # - For Edge TTS: import edge_tts
    
    async def initialize(self) -> None:
        import os
        api_key = self.config.get("api_key")
        if not api_key or "your_" in api_key:
            api_key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("TTS_API_KEY")
            
        if api_key and "your_" not in api_key:
            self.service_type = "elevenlabs"
            try:
                from elevenlabs import ElevenLabs
                self.client = ElevenLabs(api_key=api_key)
                self.voice_id = self.config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
            except ImportError:
                print("Warning: elevenlabs package not installed, falling back to Edge TTS.")
                self.service_type = "edge_tts"
                self.voice = self.config.get("voice", "en-US-AriaNeural")
        else:
            self.service_type = "edge_tts"
            self.voice = self.config.get("voice", "en-US-AriaNeural")
            print("Info: Using Edge TTS (free Microsoft voices) for speech synthesis.")
            
        self.is_initialized = True
    
    async def synthesize(self, text: str, **kwargs) -> bytes:
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")
            
        if not text.strip():
            raise ValueError("Text cannot be empty")
            
        if self.service_type == "elevenlabs":
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                
                # Execute synchronous generator call in a safe wrapper or iterate directly
                def _call_elevenlabs():
                    audio_stream = self.client.text_to_speech.stream(
                        text=text,
                        voice_id=self.voice_id,
                        model="eleven_turbo_v2_5"
                    )
                    res = b""
                    for chunk in audio_stream:
                        res += chunk
                    return res
                    
                audio_bytes = await loop.run_in_executor(None, _call_elevenlabs)
                return audio_bytes
            except Exception as e:
                print(f"ElevenLabs synthesis failed: {str(e)}. Falling back to Edge TTS.")
                # Fallback to Edge TTS if ElevenLabs errors out
                import edge_tts
                communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
                audio_bytes = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_bytes += chunk["data"]
                return audio_bytes
                
        else:
            # Edge TTS implementation
            import edge_tts
            communicate = edge_tts.Communicate(text, self.voice)
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
            return audio_bytes
    
    async def synthesize_stream(self, text: str, **kwargs) -> io.BytesIO:
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")
            
        audio_buffer = io.BytesIO()
        audio_data = await self.synthesize(text, **kwargs)
        audio_buffer.write(audio_data)
        audio_buffer.seek(0)
        return audio_buffer
    
    async def cleanup(self) -> None:
        self.client = None
        self.is_initialized = False
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")
            
        if self.service_type == "elevenlabs" and hasattr(self.client, 'voices'):
            try:
                voices = self.client.voices.get_all()
                return [{"voice_id": v.voice_id, "name": v.name} for v in voices]
            except Exception:
                return [{"voice_id": self.voice_id, "name": "Default Voice"}]
        else:
            return [{"voice_id": "en-US-AriaNeural", "name": "Aria (Edge TTS)"}]