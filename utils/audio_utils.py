"""
Audio utilities for recording, processing, and handling audio data.

This module provides functionality for audio recording, format conversion,
and audio data manipulation for the InstaVoice application.
"""

import numpy as np
import pyaudio
import wave
import tempfile
import os
from typing import Optional, Tuple, Generator
from loguru import logger
import threading
import time
from pydub import AudioSegment
from io import BytesIO


class AudioRecorder:
    """
    Audio recorder for capturing microphone input.
    
    This class provides functionality to record audio from the microphone
    with configurable parameters and real-time processing capabilities.
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, 
                 chunk_size: int = 1024, format_type: int = pyaudio.paInt16):
        """
        Initialize audio recorder.
        
        Args:
            sample_rate (int): Audio sample rate in Hz
            channels (int): Number of audio channels (1 for mono, 2 for stereo)
            chunk_size (int): Size of audio chunks to read at once
            format_type (int): PyAudio format type (paInt16, paFloat32, etc.)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format_type = format_type
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.recorded_frames = []
        
        logger.info(f"AudioRecorder initialized: {sample_rate}Hz, {channels} channels")
    
    def start_recording(self) -> None:
        """
        Start audio recording.
        
        Raises:
            RuntimeError: If recording is already in progress
        """
        if self.is_recording:
            raise RuntimeError("Recording is already in progress")
        
        try:
            self.stream = self.audio.open(
                format=self.format_type,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_recording = True
            self.recorded_frames = []
            
            logger.info("Audio recording started")
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            raise
    
    def stop_recording(self) -> np.ndarray:
        """
        Stop audio recording and return recorded data.
        
        Returns:
            np.ndarray: Recorded audio data as numpy array
            
        Raises:
            RuntimeError: If no recording is in progress
        """
        if not self.is_recording:
            raise RuntimeError("No recording in progress")
        
        try:
            self.is_recording = False
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            # Convert recorded frames to numpy array
            if self.recorded_frames:
                audio_data = b''.join(self.recorded_frames)
                
                # Check if we actually have audio data
                if len(audio_data) == 0:
                    logger.warning("No audio data in recorded frames")
                    return np.array([], dtype=np.float32)
                
                if self.format_type == pyaudio.paInt16:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    # Convert to float32 and normalize
                    audio_array = audio_array.astype(np.float32) / 32768.0
                elif self.format_type == pyaudio.paFloat32:
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                else:
                    raise ValueError(f"Unsupported audio format: {self.format_type}")
                
                # Handle stereo to mono conversion
                if self.channels == 2 and len(audio_array) > 0:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1)
                
                logger.info(f"Recording stopped. Duration: {len(audio_array) / self.sample_rate:.2f}s, Samples: {len(audio_array)}")
                return audio_array
            else:
                logger.warning("No audio frames recorded - microphone may not be working")
                return np.array([], dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            raise
    
    def record_chunk(self) -> Optional[np.ndarray]:
        """
        Record a single chunk of audio data.
        
        Returns:
            Optional[np.ndarray]: Audio chunk as numpy array, None if not recording
        """
        if not self.is_recording or not self.stream:
            return None
        
        try:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            self.recorded_frames.append(data)
            
            # Convert chunk to numpy array for real-time processing
            if self.format_type == pyaudio.paInt16:
                chunk_array = np.frombuffer(data, dtype=np.int16)
                chunk_array = chunk_array.astype(np.float32) / 32768.0
            elif self.format_type == pyaudio.paFloat32:
                chunk_array = np.frombuffer(data, dtype=np.float32)
            else:
                return None
            
            # Handle stereo to mono
            if self.channels == 2:
                chunk_array = chunk_array.reshape(-1, 2).mean(axis=1)
            
            return chunk_array
            
        except Exception as e:
            logger.error(f"Error recording chunk: {e}")
            return None
    
    def record_for_duration(self, duration_seconds: float) -> np.ndarray:
        """
        Record audio for a specified duration.
        
        Args:
            duration_seconds (float): Duration to record in seconds
            
        Returns:
            np.ndarray: Recorded audio data
        """
        self.start_recording()
        
        try:
            time.sleep(duration_seconds)
            return self.stop_recording()
        except Exception as e:
            if self.is_recording:
                self.stop_recording()
            raise e
    
    def cleanup(self) -> None:
        """Clean up audio resources."""
        if self.is_recording:
            self.stop_recording()
        
        if self.audio:
            self.audio.terminate()
            logger.info("Audio resources cleaned up")


class AudioProcessor:
    """
    Audio processing utilities for format conversion and manipulation.
    
    This class provides methods for converting between different audio formats,
    resampling, and other audio processing operations.
    """
    
    @staticmethod
    def save_audio_to_file(audio_data: np.ndarray, filename: str, 
                          sample_rate: int = 16000) -> str:
        """
        Save audio data to a WAV file.
        
        Args:
            audio_data (np.ndarray): Audio data as numpy array
            filename (str): Output filename
            sample_rate (int): Sample rate of audio data
            
        Returns:
            str: Path to saved file
        """
        try:
            # Convert float32 to int16
            if audio_data.dtype == np.float32:
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            logger.info(f"Audio saved to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            raise
    
    @staticmethod
    def load_audio_from_file(filename: str) -> Tuple[np.ndarray, int]:
        """
        Load audio data from a file.
        
        Args:
            filename (str): Path to audio file
            
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        try:
            # Use pydub for broader format support
            audio_segment = AudioSegment.from_file(filename)
            
            # Convert to mono if stereo
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            # Get sample rate
            sample_rate = audio_segment.frame_rate
            
            # Convert to numpy array
            audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Normalize to [-1, 1]
            if audio_segment.sample_width == 2:  # 16-bit
                audio_data = audio_data / 32768.0
            elif audio_segment.sample_width == 4:  # 32-bit
                audio_data = audio_data / 2147483648.0
            
            logger.info(f"Audio loaded from: {filename} ({len(audio_data)} samples, {sample_rate}Hz)")
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise
    
    @staticmethod
    def resample_audio(audio_data: np.ndarray, original_rate: int, 
                      target_rate: int) -> np.ndarray:
        """
        Resample audio data to a different sample rate.
        
        Args:
            audio_data (np.ndarray): Input audio data
            original_rate (int): Original sample rate
            target_rate (int): Target sample rate
            
        Returns:
            np.ndarray: Resampled audio data
        """
        if original_rate == target_rate:
            return audio_data
        
        try:
            # Simple linear interpolation resampling
            # For production use, consider using scipy.signal.resample or librosa
            target_length = int(len(audio_data) * target_rate / original_rate)
            
            resampled = np.interp(
                np.linspace(0, len(audio_data), target_length),
                np.arange(len(audio_data)),
                audio_data
            )
            
            logger.info(f"Audio resampled from {original_rate}Hz to {target_rate}Hz")
            return resampled.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            raise
    
    @staticmethod
    def apply_noise_reduction(audio_data: np.ndarray, 
                            noise_factor: float = 0.1) -> np.ndarray:
        """
        Apply simple noise reduction to audio data.
        
        Args:
            audio_data (np.ndarray): Input audio data
            noise_factor (float): Noise reduction factor (0.0 to 1.0)
            
        Returns:
            np.ndarray: Noise-reduced audio data
        """
        try:
            # Check for empty audio data
            if len(audio_data) == 0:
                logger.warning("Empty audio data provided for noise reduction")
                return audio_data
            
            # Simple noise gate - remove samples below threshold
            threshold = np.std(audio_data) * noise_factor
            mask = np.abs(audio_data) > threshold
            
            cleaned_audio = audio_data.copy()
            cleaned_audio[~mask] = 0
            
            logger.info(f"Noise reduction applied with factor {noise_factor}")
            return cleaned_audio
            
        except Exception as e:
            logger.error(f"Error applying noise reduction: {e}")
            return audio_data
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray, target_level: float = 0.8) -> np.ndarray:
        """
        Normalize audio data to a target level.
        
        Args:
            audio_data (np.ndarray): Input audio data
            target_level (float): Target normalization level (0.0 to 1.0)
            
        Returns:
            np.ndarray: Normalized audio data
        """
        try:
            # Check for empty audio data
            if len(audio_data) == 0:
                logger.warning("Empty audio data provided for normalization")
                return audio_data
            
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                normalized = audio_data * (target_level / max_val)
                logger.info(f"Audio normalized to level {target_level}")
                return normalized
            else:
                return audio_data
                
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return audio_data


def create_temp_audio_file(audio_data: np.ndarray, 
                          sample_rate: int = 16000) -> str:
    """
    Create a temporary audio file from numpy array.
    
    Args:
        audio_data (np.ndarray): Audio data
        sample_rate (int): Sample rate
        
    Returns:
        str: Path to temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_filename = temp_file.name
    temp_file.close()
    
    AudioProcessor.save_audio_to_file(audio_data, temp_filename, sample_rate)
    return temp_filename
