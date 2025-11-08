"""
Voice Activity Detection module using WebRTC VAD.

This module provides functionality to detect speech activity in audio streams
using Google's WebRTC Voice Activity Detection algorithm.
"""

import webrtcvad
import numpy as np
from typing import List, Tuple
from loguru import logger


class VoiceActivityDetector:
    """
    Voice Activity Detection using WebRTC VAD.
    
    This class provides methods to detect speech activity in audio frames
    and process audio streams for voice detection.
    """
    
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000):
        """
        Initialize Voice Activity Detector.
        
        Args:
            aggressiveness (int): VAD aggressiveness level (0-3).
                                0 = least aggressive, 3 = most aggressive
            sample_rate (int): Audio sample rate in Hz. Must be 8000, 16000, 32000, or 48000
        
        Raises:
            ValueError: If sample rate or aggressiveness level is invalid
        """
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Sample rate must be 8000, 16000, 32000, or 48000 Hz, got {sample_rate}")
            
        if not 0 <= aggressiveness <= 3:
            raise ValueError(f"Aggressiveness must be 0-3, got {aggressiveness}")
            
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        
        # Frame duration in milliseconds (10, 20, or 30 ms)
        self.frame_duration_ms = 30
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
        logger.info(f"VAD initialized with aggressiveness={aggressiveness}, sample_rate={sample_rate}")
    
    def is_speech(self, audio_frame: bytes) -> bool:
        """
        Detect if an audio frame contains speech.
        
        Args:
            audio_frame (bytes): Raw audio data (16-bit PCM)
            
        Returns:
            bool: True if speech is detected, False otherwise
        """
        try:
            return self.vad.is_speech(audio_frame, self.sample_rate)
        except Exception as e:
            logger.error(f"Error in speech detection: {e}")
            return False
    
    def process_audio_chunks(self, audio_data: np.ndarray) -> List[Tuple[int, bool]]:
        """
        Process audio data in chunks and detect speech activity.
        
        Args:
            audio_data (np.ndarray): Audio data as numpy array (float32, normalized to [-1, 1])
            
        Returns:
            List[Tuple[int, bool]]: List of (chunk_index, is_speech) tuples
        """
        # Check for empty audio data
        if len(audio_data) == 0:
            logger.warning("Empty audio data provided to VAD")
            return []
        
        # Ensure audio is properly normalized
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            # Normalize to use full dynamic range, but don't exceed [-1, 1]
            if max_val < 0.1:  # Very quiet audio
                audio_data = audio_data / max_val * 0.5  # Boost quiet audio
                logger.info(f"Boosted quiet audio (original max: {max_val:.4f})")
            elif max_val > 1.0:  # Clipped audio
                audio_data = audio_data / max_val * 0.9  # Normalize clipped audio
                logger.info(f"Normalized clipped audio (original max: {max_val:.4f})")
        
        # Convert float32 to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        results = []
        chunk_size = self.frame_size
        
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i + chunk_size]
            
            # Pad chunk if it's too short
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            
            # Convert to bytes
            chunk_bytes = chunk.tobytes()
            
            # Detect speech
            is_speech = self.is_speech(chunk_bytes)
            results.append((i // chunk_size, is_speech))
        
        # Log some statistics for debugging
        speech_frames = sum(1 for _, is_speech in results if is_speech)
        logger.info(f"VAD processed {len(results)} frames, {speech_frames} detected as speech ({speech_frames/len(results)*100:.1f}%)")
        
        return results
    
    def get_speech_segments(self, audio_data: np.ndarray, 
                          min_speech_duration: float = 0.5) -> List[Tuple[float, float]]:
        """
        Extract speech segments from audio data.
        
        Args:
            audio_data (np.ndarray): Audio data as numpy array
            min_speech_duration (float): Minimum duration of speech segment in seconds
            
        Returns:
            List[Tuple[float, float]]: List of (start_time, end_time) tuples in seconds
        """
        vad_results = self.process_audio_chunks(audio_data)
        
        segments = []
        current_start = None
        
        frame_duration_sec = self.frame_duration_ms / 1000.0
        
        for chunk_idx, is_speech in vad_results:
            time_sec = chunk_idx * frame_duration_sec
            
            if is_speech and current_start is None:
                # Start of speech segment
                current_start = time_sec
            elif not is_speech and current_start is not None:
                # End of speech segment
                duration = time_sec - current_start
                if duration >= min_speech_duration:
                    segments.append((current_start, time_sec))
                current_start = None
        
        # Handle case where audio ends during speech
        if current_start is not None:
            end_time = len(vad_results) * frame_duration_sec
            duration = end_time - current_start
            if duration >= min_speech_duration:
                segments.append((current_start, end_time))
        
        logger.info(f"Detected {len(segments)} speech segments")
        return segments
