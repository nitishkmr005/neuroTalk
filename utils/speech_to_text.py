"""
Speech-to-Text module using OpenAI Whisper.

This module provides functionality to transcribe audio data using
OpenAI's Whisper speech recognition model.
"""

import whisper
import numpy as np
import tempfile
import os
from typing import Optional, Dict, Any
from loguru import logger
import torch


class SpeechToTextProcessor:
    """
    Speech-to-Text processor using OpenAI Whisper.
    
    This class provides methods to transcribe audio data using various
    Whisper model sizes with configurable options.
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize Speech-to-Text processor.
        
        Args:
            model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device (Optional[str]): Device to run model on ('cpu', 'cuda'). Auto-detected if None
        
        Raises:
            ValueError: If model size is invalid
        """
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if model_size not in valid_models:
            raise ValueError(f"Model size must be one of {valid_models}, got {model_size}")
        
        self.model_size = model_size
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading Whisper model '{model_size}' on device '{self.device}'...")
        
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            logger.info(f"Whisper model '{model_size}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_data: np.ndarray, 
                        sample_rate: int = 16000,
                        language: Optional[str] = None,
                        task: str = "transcribe") -> Dict[str, Any]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data (np.ndarray): Audio data as numpy array (float32)
            sample_rate (int): Sample rate of audio data
            language (Optional[str]): Language code (e.g., 'en', 'es'). Auto-detected if None
            task (str): Task type ('transcribe' or 'translate')
            
        Returns:
            Dict[str, Any]: Transcription result containing text and metadata
        """
        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio to [-1, 1] range
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                logger.warning(f"Resampling from {sample_rate}Hz to 16000Hz")
                # Simple resampling (for production, use proper resampling)
                target_length = int(len(audio_data) * 16000 / sample_rate)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), target_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Transcribe using Whisper
            options = {
                "task": task,
                "fp16": self.device == "cuda",  # Use FP16 on GPU for speed
            }
            
            if language:
                options["language"] = language
            
            result = self.model.transcribe(audio_data, **options)
            
            logger.info(f"Transcription completed: '{result['text'][:100]}...'")
            return result
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return {"text": "", "error": str(e)}
    
    def transcribe_file(self, audio_file_path: str, 
                       language: Optional[str] = None,
                       task: str = "transcribe") -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_file_path (str): Path to audio file
            language (Optional[str]): Language code. Auto-detected if None
            task (str): Task type ('transcribe' or 'translate')
            
        Returns:
            Dict[str, Any]: Transcription result containing text and metadata
        """
        try:
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            options = {
                "task": task,
                "fp16": self.device == "cuda",
            }
            
            if language:
                options["language"] = language
            
            result = self.model.transcribe(audio_file_path, **options)
            
            logger.info(f"File transcription completed: '{result['text'][:100]}...'")
            return result
            
        except Exception as e:
            logger.error(f"Error during file transcription: {e}")
            return {"text": "", "error": str(e)}
    
    def transcribe_with_timestamps(self, audio_data: np.ndarray,
                                 sample_rate: int = 16000,
                                 language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio with word-level timestamps.
        
        Args:
            audio_data (np.ndarray): Audio data as numpy array
            sample_rate (int): Sample rate of audio data
            language (Optional[str]): Language code. Auto-detected if None
            
        Returns:
            Dict[str, Any]: Transcription result with detailed timestamps
        """
        try:
            # Prepare audio
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Resample if needed
            if sample_rate != 16000:
                target_length = int(len(audio_data) * 16000 / sample_rate)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), target_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Transcribe with word timestamps
            options = {
                "task": "transcribe",
                "word_timestamps": True,
                "fp16": self.device == "cuda",
            }
            
            if language:
                options["language"] = language
            
            result = self.model.transcribe(audio_data, **options)
            
            logger.info(f"Timestamped transcription completed")
            return result
            
        except Exception as e:
            logger.error(f"Error during timestamped transcription: {e}")
            return {"text": "", "segments": [], "error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information including size, device, and parameters
        """
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
            
            return {
                "model_size": self.model_size,
                "device": self.device,
                "parameter_count": param_count,
                "model_type": type(self.model).__name__,
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}
