"""
Base Speech-to-Text (STT) Interface

This module defines the abstract base class for Speech-to-Text implementations.
Students should implement the concrete STT class by inheriting from this base class.

Recommended implementation: Deepgram API (free tier available)
Alternative options: OpenAI Whisper, AssemblyAI, or any other STT service
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseSTT(ABC):
    """
    Abstract base class for Speech-to-Text implementations.
    
    This class defines the interface that all STT implementations must follow.
    Students should inherit from this class and implement the abstract methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the STT service.
        
        Args:
            config: Configuration dictionary containing API keys, model settings, etc.
                   Example: {"api_key": "your_api_key", "model": "nova-2"}
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the STT service (setup API clients, load models, etc.).
        This method should be called before using the STT service.
        
        Raises:
            Exception: If initialization fails
        """
        pass
    
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Raw audio data as bytes
            **kwargs: Additional parameters specific to the STT implementation
                     (e.g., language, model, formatting options)
        
        Returns:
            str: The transcribed text
            
        Raises:
            Exception: If transcription fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup resources (close connections, free memory, etc.).
        This method should be called when the STT service is no longer needed.
        """
        pass
    
    def is_ready(self) -> bool:
        """
        Check if the STT service is ready to use.
        
        Returns:
            bool: True if ready, False otherwise
        """
        return self.is_initialized


class STTService(BaseSTT):
    """
    Generic STT implementation template.
    
    Students should complete this implementation using their chosen STT service or pretrained model.
    
    API-based options:
    - Deepgram API (free tier, high accuracy): pip install deepgram-sdk
    - AssemblyAI (API-based): pip install assemblyai
    - Azure Speech Services: pip install azure-cognitiveservices-speech
    - Google Cloud Speech: pip install google-cloud-speech
    
    Pretrained model options (local inference):
    - OpenAI Whisper: pip install openai-whisper (various sizes: tiny, base, small, medium, large)
    - Wav2Vec2 models: pip install transformers torch (Facebook's pretrained models)
    - SpeechRecognition + offline engines: pip install SpeechRecognition pocketsphinx
    - Vosk models: pip install vosk (lightweight, supports many languages)
    - Coqui STT: pip install coqui-stt (open-source, pretrained models available)
    
    Input: audio_bytes (bytes) - Raw audio data
    Output: transcribed_text (str) - The text transcription
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = None
        # TODO: Initialize your chosen STT client/model
        # API-based examples:
        # - For Deepgram: from deepgram import DeepgramClient
        # - For AssemblyAI: import assemblyai
        # Pretrained model examples:
        # - For Whisper: import whisper
        # - For Transformers: from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        # - For Vosk: import vosk
    
    async def initialize(self) -> None:
        import os
        api_key = self.config.get("api_key")
        if not api_key or "your_" in api_key:
            api_key = os.getenv("DEEPGRAM_API_KEY") or os.getenv("STT_API_KEY")
            
        model = self.config.get("model", "nova-2")
        
        if api_key and "your_" not in api_key:
            self.service_type = "deepgram"
            try:
                from deepgram import DeepgramClient
                self.client = DeepgramClient(api_key)
            except ImportError:
                print("Warning: deepgram-sdk not installed, falling back to mock STT.")
                self.service_type = "mock"
        elif "whisper" in model.lower() or model in ["tiny", "base", "small", "medium", "large"]:
            self.service_type = "whisper"
            try:
                import whisper
                self.client = whisper.load_model(model if model in ["tiny", "base", "small", "medium", "large"] else "base")
            except ImportError:
                print("Warning: openai-whisper not installed, falling back to mock STT.")
                self.service_type = "mock"
        else:
            self.service_type = "mock"
            print("Info: Using mock STT service (no valid Deepgram key or Whisper configuration provided).")
            
        self.is_initialized = True
    
    async def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        if not self.is_ready():
            raise RuntimeError("STT service not initialized")
            
        if self.service_type == "deepgram":
            import asyncio
            payload = {"buffer": audio_bytes}
            options = {"model": self.config.get("model", "nova-2"), "smart_format": True}
            
            try:
                response = self.client.listen.prerecorded.v("1").transcribe_file(payload, options)
                if asyncio.iscoroutine(response):
                    response = await response
                return response["results"]["channels"][0]["alternatives"][0]["transcript"]
            except Exception as e:
                print(f"Deepgram transcription failed: {str(e)}. Using fallback mock transcription.")
                return "What is your return policy?"
                
        elif self.service_type == "whisper":
            import tempfile
            import asyncio
            import os
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_name = temp_file.name
                
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.client.transcribe, temp_file_name)
                return result["text"]
            except Exception as e:
                print(f"Whisper transcription failed: {str(e)}. Using fallback mock transcription.")
                return "What is your return policy?"
            finally:
                if os.path.exists(temp_file_name):
                    try:
                        os.unlink(temp_file_name)
                    except Exception:
                        pass
        else:
            # Mock transcription fallback
            print("Info: Returning mock transcription query.")
            return "What is your return policy?"
    
    async def cleanup(self) -> None:
        self.client = None
        self.is_initialized = False