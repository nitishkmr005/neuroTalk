"""
Tests for Voice Activity Detection module.
"""

import pytest
import numpy as np
from utils.vad import VoiceActivityDetector


class TestVoiceActivityDetector:
    """Test cases for VoiceActivityDetector class."""
    
    def test_vad_initialization(self):
        """Test VAD initialization with valid parameters."""
        vad = VoiceActivityDetector(aggressiveness=2, sample_rate=16000)
        
        assert vad.aggressiveness == 2
        assert vad.sample_rate == 16000
        assert vad.frame_duration_ms == 30
        assert vad.frame_size == 480  # 16000 * 30 / 1000
    
    def test_vad_invalid_sample_rate(self):
        """Test VAD initialization with invalid sample rate."""
        with pytest.raises(ValueError, match="Sample rate must be"):
            VoiceActivityDetector(sample_rate=44100)
    
    def test_vad_invalid_aggressiveness(self):
        """Test VAD initialization with invalid aggressiveness."""
        with pytest.raises(ValueError, match="Aggressiveness must be"):
            VoiceActivityDetector(aggressiveness=5)
    
    def test_process_audio_chunks(self):
        """Test processing audio chunks."""
        vad = VoiceActivityDetector(aggressiveness=2, sample_rate=16000)
        
        # Create dummy audio data (1 second of silence)
        audio_data = np.zeros(16000, dtype=np.float32)
        
        results = vad.process_audio_chunks(audio_data)
        
        # Should have results for each chunk
        assert len(results) > 0
        assert all(isinstance(chunk_idx, int) and isinstance(is_speech, bool) 
                  for chunk_idx, is_speech in results)
    
    def test_get_speech_segments(self):
        """Test extracting speech segments."""
        vad = VoiceActivityDetector(aggressiveness=1, sample_rate=16000)
        
        # Create dummy audio data
        audio_data = np.random.normal(0, 0.1, 16000).astype(np.float32)
        
        segments = vad.get_speech_segments(audio_data, min_speech_duration=0.1)
        
        # Should return a list of tuples
        assert isinstance(segments, list)
        assert all(isinstance(seg, tuple) and len(seg) == 2 for seg in segments)
        assert all(start < end for start, end in segments)
