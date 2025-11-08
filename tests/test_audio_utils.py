"""
Tests for Audio utilities module.
"""

import pytest
import numpy as np
import tempfile
import os
from utils.audio_utils import AudioProcessor


class TestAudioProcessor:
    """Test cases for AudioProcessor class."""
    
    def test_save_and_load_audio(self):
        """Test saving and loading audio files."""
        # Create test audio data
        sample_rate = 16000
        duration = 1.0  # 1 second
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        audio_data = audio_data.astype(np.float32)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            # Save audio
            saved_path = AudioProcessor.save_audio_to_file(audio_data, tmp_filename, sample_rate)
            assert saved_path == tmp_filename
            assert os.path.exists(tmp_filename)
            
            # Load audio back
            loaded_audio, loaded_rate = AudioProcessor.load_audio_from_file(tmp_filename)
            
            assert loaded_rate == sample_rate
            assert isinstance(loaded_audio, np.ndarray)
            assert loaded_audio.dtype == np.float32
            
        finally:
            # Clean up
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_resample_audio(self):
        """Test audio resampling."""
        # Create test audio
        original_rate = 44100
        target_rate = 16000
        duration = 1.0
        
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(original_rate * duration)))
        audio_data = audio_data.astype(np.float32)
        
        # Resample
        resampled = AudioProcessor.resample_audio(audio_data, original_rate, target_rate)
        
        # Check output
        expected_length = int(len(audio_data) * target_rate / original_rate)
        assert len(resampled) == expected_length
        assert resampled.dtype == np.float32
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        # Create test audio with known amplitude
        audio_data = np.array([0.5, -0.8, 0.3, -0.2], dtype=np.float32)
        target_level = 0.5
        
        normalized = AudioProcessor.normalize_audio(audio_data, target_level)
        
        # Check that max amplitude matches target level
        max_amplitude = np.max(np.abs(normalized))
        assert abs(max_amplitude - target_level) < 1e-6
    
    def test_apply_noise_reduction(self):
        """Test noise reduction."""
        # Create noisy audio
        clean_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        noise = np.random.normal(0, 0.1, 16000)
        noisy_audio = (clean_signal + noise).astype(np.float32)
        
        # Apply noise reduction
        cleaned = AudioProcessor.apply_noise_reduction(noisy_audio, noise_factor=0.2)
        
        # Should return same shape
        assert cleaned.shape == noisy_audio.shape
        assert cleaned.dtype == np.float32
