"""
Configuration management for InstaVoice application.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


class Config:
    """
    Configuration class for InstaVoice application.
    
    Centralizes all configuration values and provides validation.
    """
    
    def __init__(self):
        """Initialize configuration with environment variables."""
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "base")
        self.vad_aggressiveness: int = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
        self.sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
        self.chunk_duration_ms: int = int(os.getenv("CHUNK_DURATION_MS", "30"))
        
        self._validate_config()
        
    def _validate_config(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If configuration values are invalid.
        """
        valid_whisper_models = ["tiny", "base", "small", "medium", "large"]
        if self.whisper_model_size not in valid_whisper_models:
            raise ValueError(
                f"Invalid Whisper model size: {self.whisper_model_size}. "
                f"Must be one of: {valid_whisper_models}"
            )
            
        if not 0 <= self.vad_aggressiveness <= 3:
            raise ValueError(
                f"VAD aggressiveness must be between 0-3, got: {self.vad_aggressiveness}"
            )
            
        if self.sample_rate not in [8000, 16000, 32000, 48000]:
            logger.warning(
                f"Sample rate {self.sample_rate} may not be optimal. "
                "Recommended: 16000 Hz"
            )
            
        logger.info(f"Configuration loaded successfully:")
        logger.info(f"  - Whisper model: {self.whisper_model_size}")
        logger.info(f"  - VAD aggressiveness: {self.vad_aggressiveness}")
        logger.info(f"  - Sample rate: {self.sample_rate} Hz")


# Global configuration instance
config = Config()
