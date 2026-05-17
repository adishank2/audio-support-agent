"""
Audio Customer Support Agent Pipeline

This module orchestrates the complete STT -> LLM -> TTS pipeline.
Students should complete the implementation to connect all components.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict

from src.stt.base_stt import BaseSTT, STTService
from src.llm.agent import BaseAgent, CustomerSupportAgent
from src.tts.base_tts import BaseTTS, TTSService


@dataclass
class PipelineConfig:
    """Configuration for the audio support pipeline."""
    stt_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    tts_config: Dict[str, Any]
    enable_logging: bool = True


@dataclass
class TranscriptData:
    """Data structure for transcript information."""
    user_input: str = ""
    agent_response: str = ""
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AudioSupportPipeline:
    """
    Main pipeline class that orchestrates STT -> LLM -> TTS flow.
    
    This class manages the entire audio processing pipeline for customer support.
    Students should complete the implementation to make it fully functional.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the audio support pipeline.
        
        Args:
            config: Pipeline configuration containing settings for all components
        """
        self.config = config
        self.stt: Optional[BaseSTT] = None
        self.llm_agent: Optional[BaseAgent] = None
        self.tts: Optional[BaseTTS] = None
        self.is_initialized = False
        
        if config.enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.CRITICAL)
    
    async def initialize(self) -> None:
        try:
            self.logger.info("Initializing Audio Support Pipeline...")
            
            # Initialize STT
            self.logger.info("Initializing STT service...")
            self.stt = STTService(self.config.stt_config)
            await self.stt.initialize()
            
            # Initialize LLM Agent
            self.logger.info("Initializing LLM agent...")
            self.llm_agent = CustomerSupportAgent(self.config.llm_config)
            await self.llm_agent.initialize()
            
            # Initialize TTS
            self.logger.info("Initializing TTS service...")
            self.tts = TTSService(self.config.tts_config)
            await self.tts.initialize()
            
            # Verify all components are ready
            if not all([self.stt.is_ready(), self.llm_agent.is_initialized, self.tts.is_ready()]):
                raise RuntimeError("Some pipeline components failed to initialize")
            
            self.is_initialized = True
            self.logger.info("Pipeline initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {str(e)}")
            await self.cleanup()
            raise
    
    async def process_audio(self, audio_bytes: bytes, **kwargs) -> bytes:
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        try:
            # Step 1 - Speech to Text
            self.logger.info("Converting speech to text...")
            text_input = await self.stt.transcribe(audio_bytes, **kwargs)
            self.logger.info(f"Transcribed text: {text_input}")
            
            # Step 2 - Process with LLM Agent
            self.logger.info("Processing query with LLM agent...")
            agent_response = await self.llm_agent.process_query(text_input, **kwargs)
            self.logger.info(f"Agent response: {agent_response}")
            
            # Step 3 - Text to Speech
            self.logger.info("Converting response to speech...")
            response_audio = await self.tts.synthesize(agent_response, **kwargs)
            self.logger.info("Audio response generated successfully")
            
            return response_audio
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            raise
    
    async def process_audio_with_transcript(self, audio_bytes: bytes, **kwargs) -> Tuple[bytes, TranscriptData, int]:
        """Process audio and capture transcript data with timing.
        
        Returns:
            Tuple[bytes, TranscriptData, int]: (response_audio, transcript_data, processing_time_ms)
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        overall_start = time.time()
        
        try:
            # Step 1 - Speech to Text
            stt_start = time.time()
            self.logger.info("[Transcript] Converting speech to text...")
            user_input = await self.stt.transcribe(audio_bytes, **kwargs)
            stt_time = int((time.time() - stt_start) * 1000)
            self.logger.info(f"[Transcript] Transcribed: {user_input}")
            
            # Step 2 - Process with LLM Agent
            llm_start = time.time()
            self.logger.info("[Transcript] Processing query with LLM agent...")
            agent_response = await self.llm_agent.process_query(user_input, **kwargs)
            llm_time = int((time.time() - llm_start) * 1000)
            self.logger.info(f"[Transcript] Agent response: {agent_response}")
            
            # Step 3 - Text to Speech
            tts_start = time.time()
            self.logger.info("[Transcript] Converting response to speech...")
            response_audio = await self.tts.synthesize(agent_response, **kwargs)
            tts_time = int((time.time() - tts_start) * 1000)
            self.logger.info("[Transcript] Audio response generated")
            
            total_time = int((time.time() - overall_start) * 1000)
            
            transcript = self._create_transcript_data(
                user_input=user_input,
                agent_response=agent_response,
                stt_time_ms=stt_time,
                llm_time_ms=llm_time,
                tts_time_ms=tts_time,
                total_time_ms=total_time
            )
            
            return response_audio, transcript, total_time
            
        except Exception as e:
            total_time = int((time.time() - overall_start) * 1000)
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            raise
    
    async def process_text_with_timing(self, text: str, **kwargs) -> Tuple[str, int]:
        """Process text and capture processing time.
        
        Returns:
            Tuple[str, int]: (agent_response_text, processing_time_ms)
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        start_time = time.time()
        try:
            agent_response = await self.llm_agent.process_query(text, **kwargs)
            processing_time = int((time.time() - start_time) * 1000)
            return agent_response, processing_time
        except Exception as e:
            self.logger.error(f"Text processing with timing failed: {str(e)}")
            raise
    
    def _create_transcript_data(self, user_input: str, agent_response: str, 
                                 stt_time_ms: int = 0, llm_time_ms: int = 0,
                                 tts_time_ms: int = 0, total_time_ms: int = 0) -> TranscriptData:
        """Create structured transcript data."""
        return TranscriptData(
            user_input=user_input,
            agent_response=agent_response,
            processing_metadata={
                "stt_processing_time_ms": stt_time_ms,
                "response_generation_time_ms": llm_time_ms,
                "tts_generation_time_ms": tts_time_ms,
                "total_processing_time_ms": total_time_ms
            }
        )
    
    async def process_text(self, text_input: str, **kwargs) -> Tuple[str, bytes]:
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        try:
            # Process with LLM Agent
            self.logger.info(f"Processing text query: {text_input}")
            agent_response = await self.llm_agent.process_query(text_input, **kwargs)
            
            # Convert to speech
            response_audio = await self.tts.synthesize(agent_response, **kwargs)
            
            return agent_response, response_audio
            
        except Exception as e:
            self.logger.error(f"Text processing failed: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, bool]:
        """
        TODO: Check the health status of all pipeline components.
        
        Returns:
            Dict[str, bool]: Status of each component
        """
        return {
            "pipeline_initialized": self.is_initialized,
            "stt_ready": self.stt.is_ready() if self.stt else False,
            "llm_ready": self.llm_agent.is_initialized if self.llm_agent else False,
            "tts_ready": self.tts.is_ready() if self.tts else False,
        }
    
    async def cleanup(self) -> None:
        """
        TODO: Cleanup all pipeline resources.
        
        This method should be called when the pipeline is no longer needed.
        """
        self.logger.info("Cleaning up pipeline resources...")
        
        try:
            # TODO: Cleanup all components
            if self.stt:
                await self.stt.cleanup()
            if self.llm_agent:
                await self.llm_agent.cleanup()
            if self.tts:
                await self.tts.cleanup()
                
            self.stt = None
            self.llm_agent = None
            self.tts = None
            self.is_initialized = False
            
            self.logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            raise


async def create_pipeline(
    stt_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    tts_config: Dict[str, Any],
    enable_logging: bool = True
) -> AudioSupportPipeline:
    """
    TODO: Factory function to create and initialize a pipeline.
    
    Args:
        stt_config: STT configuration
        llm_config: LLM configuration  
        tts_config: TTS configuration
        enable_logging: Whether to enable logging
        
    Returns:
        AudioSupportPipeline: Initialized pipeline instance
    """
    config = PipelineConfig(
        stt_config=stt_config,
        llm_config=llm_config,
        tts_config=tts_config,
        enable_logging=enable_logging
    )
    
    pipeline = AudioSupportPipeline(config)
    await pipeline.initialize()
    
    return pipeline


if __name__ == "__main__":
    import os
    async def main():
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
        
        print("Creating and testing pipeline...")
        pipeline = await create_pipeline(stt_config, llm_config, tts_config)
        
        response_text, response_audio = await pipeline.process_text("What is your return policy?")
        print(f"Response: {response_text}")
        print(f"Audio response length: {len(response_audio)} bytes")
        
        await pipeline.cleanup()
        print("Pipeline example completed successfully!")
    
    asyncio.run(main())