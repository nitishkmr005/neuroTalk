#!/usr/bin/env python3
"""
Interactive Microphone Test - Will prompt you to speak

This script tests:
1. PyAudio microphone access
2. Real-time audio recording
3. Audio level detection
4. VAD (Voice Activity Detection)
5. Speech-to-text transcription
"""

import sys
import time
import numpy as np
import pyaudio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import config
from utils.vad import VoiceActivityDetector
from utils.speech_to_text import SpeechToTextProcessor

print("\n" + "="*70)
print("  🎤 INTERACTIVE MICROPHONE TEST")
print("="*70 + "\n")

# Test 1: Check PyAudio
print("1️⃣  Checking audio devices...")
try:
    pa = pyaudio.PyAudio()
    device_count = pa.get_device_count()
    print(f"   ✅ Found {device_count} audio devices\n")
    
    # List all input devices
    print("   Available input devices:")
    for i in range(device_count):
        info = pa.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"      [{i}] {info['name']} - {info['maxInputChannels']} channels")
    
    default_input = pa.get_default_input_device_info()
    print(f"\n   Default input: {default_input['name']}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 2: Record audio with visual feedback
print("\n2️⃣  Testing live microphone recording...\n")
print("="*70)
print("  🔴 GET READY TO SPEAK IN 3 SECONDS!")
print("="*70)

time.sleep(3)

try:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5
    
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print("\n  🎤 RECORDING NOW - Say: 'Hello, this is a microphone test'\n")
    
    frames = []
    max_amplitude_seen = 0
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        
        # Convert to numpy for amplitude detection
        audio_chunk = np.frombuffer(data, dtype=np.int16)
        amplitude = np.max(np.abs(audio_chunk))
        max_amplitude_seen = max(max_amplitude_seen, amplitude)
        
        # Visual level meter
        level = int((amplitude / 32768.0) * 50)
        bar = "█" * level
        print(f"  Level: [{bar:<50}] {amplitude:5d}", end='\r')
    
    print("\n\n  ⏹️  Recording stopped!")
    
    stream.stop_stream()
    stream.close()
    
    # Convert to numpy array
    audio_data = b''.join(frames)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    print(f"\n   Audio captured:")
    print(f"      - Samples: {len(audio_np)}")
    print(f"      - Duration: {len(audio_np)/RATE:.2f}s")
    print(f"      - Max amplitude: {np.max(np.abs(audio_np)):.4f}")
    print(f"      - RMS amplitude: {np.sqrt(np.mean(audio_np**2)):.4f}")
    
    if max_amplitude_seen < 100:
        print("\n   ⚠️  WARNING: Very low audio levels detected!")
        print("      Please check:")
        print("      - Microphone is not muted")
        print("      - Input volume is turned up")
        print("      - Correct microphone is selected")
    else:
        print(f"\n   ✅ Good audio levels detected (peak: {max_amplitude_seen})")
    
except Exception as e:
    print(f"\n   ❌ Recording error: {e}")
    pa.terminate()
    sys.exit(1)

# Test 3: VAD
print("\n3️⃣  Testing Voice Activity Detection...")
try:
    vad = VoiceActivityDetector(aggressiveness=1, sample_rate=RATE)
    vad_results = vad.process_audio_chunks(audio_np)
    
    speech_frames = sum(1 for _, is_speech in vad_results if is_speech)
    total_frames = len(vad_results)
    speech_ratio = (speech_frames / total_frames * 100) if total_frames > 0 else 0
    
    print(f"   VAD Results:")
    print(f"      - Total frames: {total_frames}")
    print(f"      - Speech frames: {speech_frames}")
    print(f"      - Speech ratio: {speech_ratio:.1f}%")
    
    if speech_frames > 0:
        print(f"   ✅ Speech detected!")
    else:
        print(f"   ⚠️  No speech detected - try speaking louder")
        
except Exception as e:
    print(f"   ❌ VAD error: {e}")

# Test 4: Speech-to-Text
if speech_frames > 0:
    print("\n4️⃣  Testing Speech-to-Text transcription...")
    print("   ⏳ Loading Whisper model (may take a moment)...")
    
    try:
        # Save to temp file
        import tempfile
        import scipy.io.wavfile as wavfile
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            audio_int16 = (audio_np * 32767).astype(np.int16)
            wavfile.write(temp_path, RATE, audio_int16)
        
        stt = SpeechToTextProcessor(model_size="base")
        result = stt.transcribe_file(temp_path)
        
        # Cleanup
        import os
        os.remove(temp_path)
        
        print(f"\n   Transcription Result:")
        print(f"      Language: {result['language']}")
        print(f"\n" + "="*70)
        print(f"  📝 TRANSCRIPTION: {result['text']}")
        print("="*70)
        
        if result['text'].strip():
            print(f"\n   ✅ Transcription successful!")
        else:
            print(f"\n   ⚠️  Empty transcription")
            
    except Exception as e:
        print(f"\n   ❌ STT error: {e}")
else:
    print("\n4️⃣  Skipping Speech-to-Text (no speech detected)")

# Cleanup
pa.terminate()

print("\n" + "="*70)
print("  ✅ MICROPHONE TEST COMPLETE!")
print("="*70 + "\n")

print("If you see low audio levels or no speech detected:")
print("  1. Check System Settings → Sound → Input")
print("  2. Ensure correct microphone is selected")
print("  3. Check input volume level")
print("  4. Test microphone in another app first")
print()

