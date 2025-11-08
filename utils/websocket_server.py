"""
WebSocket server for real-time audio streaming and processing.

This module provides a WebSocket server that receives audio chunks from the browser,
processes them with VAD and speech-to-text, and sends results back in real-time.
"""

import asyncio
import websockets
import json
import numpy as np
import base64
import threading
import queue
from typing import Dict, Any, Optional
from loguru import logger
import time

from .vad import VoiceActivityDetector
from .speech_to_text import SpeechToTextProcessor
from .audio_utils import AudioProcessor


class AudioBuffer:
    """
    Buffer for accumulating audio chunks for speech processing.
    """
    
    def __init__(self, sample_rate: int = 16000, max_duration: float = 30.0):
        """
        Initialize audio buffer.
        
        Args:
            sample_rate (int): Audio sample rate
            max_duration (float): Maximum buffer duration in seconds
        """
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        self.buffer = np.array([], dtype=np.float32)
        self.is_recording = False
        self.last_speech_time = 0
        self.silence_threshold = 1.0  # seconds of silence before processing
        
    def add_chunk(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk to buffer."""
        if len(audio_chunk) == 0:
            return
            
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        
        # Keep buffer within max size
        if len(self.buffer) > self.max_samples:
            # Keep only the last max_samples
            self.buffer = self.buffer[-self.max_samples:]
    
    def get_buffer(self) -> np.ndarray:
        """Get current buffer contents."""
        return self.buffer.copy()
    
    def clear_buffer(self) -> None:
        """Clear the buffer."""
        self.buffer = np.array([], dtype=np.float32)
    
    def get_duration(self) -> float:
        """Get current buffer duration in seconds."""
        return len(self.buffer) / self.sample_rate if len(self.buffer) > 0 else 0.0


class RealTimeVoiceProcessor:
    """
    Real-time voice processing with VAD and speech-to-text.
    """
    
    def __init__(self, 
                 vad_aggressiveness: int = 1,
                 sample_rate: int = 16000,
                 whisper_model: str = "base"):
        """
        Initialize real-time voice processor.
        
        Args:
            vad_aggressiveness (int): VAD aggressiveness level (0-3)
            sample_rate (int): Audio sample rate
            whisper_model (str): Whisper model size
        """
        self.sample_rate = sample_rate
        self.vad = VoiceActivityDetector(vad_aggressiveness, sample_rate)
        self.stt = None  # Lazy load STT model
        self.whisper_model = whisper_model
        
        self.audio_buffer = AudioBuffer(sample_rate)
        self.processing_queue = queue.Queue()
        self.is_processing = False
        
        # Statistics
        self.total_chunks_processed = 0
        self.speech_chunks_detected = 0
        self.last_transcription = ""
        self.last_transcription_time = 0
        
        logger.info(f"RealTimeVoiceProcessor initialized with VAD={vad_aggressiveness}, SR={sample_rate}")
    
    def _load_stt_model(self) -> None:
        """Lazy load speech-to-text model."""
        if self.stt is None:
            logger.info(f"Loading Whisper model '{self.whisper_model}'...")
            self.stt = SpeechToTextProcessor(self.whisper_model)
            logger.info("Whisper model loaded successfully")
    
    def process_audio_chunk(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Process a single audio chunk.
        
        Args:
            audio_data (bytes): Raw audio data (base64 encoded)
            
        Returns:
            Dict[str, Any]: Processing results
        """
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            # Convert to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            if len(audio_float) == 0:
                return {"error": "Empty audio chunk"}
            
            # Add to buffer
            self.audio_buffer.add_chunk(audio_float)
            
            # Process with VAD
            vad_results = self.vad.process_audio_chunks(audio_float)
            
            # Update statistics
            self.total_chunks_processed += 1
            speech_detected = any(is_speech for _, is_speech in vad_results)
            
            if speech_detected:
                self.speech_chunks_detected += 1
                self.audio_buffer.last_speech_time = time.time()
            
            # Calculate metrics
            max_amplitude = np.max(np.abs(audio_float)) if len(audio_float) > 0 else 0.0
            rms_amplitude = np.sqrt(np.mean(audio_float**2)) if len(audio_float) > 0 else 0.0
            
            result = {
                "status": "processed",
                "speech_detected": speech_detected,
                "max_amplitude": float(max_amplitude),
                "rms_amplitude": float(rms_amplitude),
                "chunk_duration": len(audio_float) / self.sample_rate,
                "buffer_duration": self.audio_buffer.get_duration(),
                "total_chunks": self.total_chunks_processed,
                "speech_chunks": self.speech_chunks_detected,
                "timestamp": time.time()
            }
            
            # Check if we should trigger transcription
            current_time = time.time()
            time_since_speech = current_time - self.audio_buffer.last_speech_time
            
            if (speech_detected and 
                self.audio_buffer.get_duration() > 1.0 and 
                time_since_speech < self.audio_buffer.silence_threshold):
                
                # Queue for transcription
                self.processing_queue.put(self.audio_buffer.get_buffer())
                result["transcription_queued"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {"error": str(e)}
    
    def process_transcription_queue(self) -> Optional[Dict[str, Any]]:
        """
        Process queued audio for transcription.
        
        Returns:
            Optional[Dict[str, Any]]: Transcription result or None
        """
        if self.processing_queue.empty():
            return None
        
        try:
            # Get audio from queue
            audio_data = self.processing_queue.get_nowait()
            
            if len(audio_data) == 0:
                return None
            
            # Lazy load STT model
            self._load_stt_model()
            
            # Transcribe (force English language)
            logger.info(f"Transcribing {len(audio_data)} samples ({len(audio_data)/self.sample_rate:.2f}s)")
            
            transcription_result = self.stt.transcribe_audio(
                audio_data,
                sample_rate=self.sample_rate,
                language="en"  # Force English transcription
            )
            
            if 'error' not in transcription_result:
                self.last_transcription = transcription_result.get('text', '')
                self.last_transcription_time = time.time()
                
                return {
                    "status": "transcription_complete",
                    "text": self.last_transcription,
                    "language": transcription_result.get('language', 'unknown'),
                    "duration": len(audio_data) / self.sample_rate,
                    "timestamp": self.last_transcription_time
                }
            else:
                logger.error(f"Transcription error: {transcription_result['error']}")
                return {
                    "status": "transcription_error",
                    "error": transcription_result['error']
                }
                
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Error in transcription processing: {e}")
            return {
                "status": "transcription_error",
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        speech_ratio = (self.speech_chunks_detected / self.total_chunks_processed 
                       if self.total_chunks_processed > 0 else 0.0)
        
        return {
            "total_chunks": self.total_chunks_processed,
            "speech_chunks": self.speech_chunks_detected,
            "speech_ratio": speech_ratio,
            "buffer_duration": self.audio_buffer.get_duration(),
            "last_transcription": self.last_transcription,
            "last_transcription_time": self.last_transcription_time,
            "model_loaded": self.stt is not None
        }
    
    def reset(self) -> None:
        """Reset processor state."""
        self.audio_buffer.clear_buffer()
        self.total_chunks_processed = 0
        self.speech_chunks_detected = 0
        self.last_transcription = ""
        self.last_transcription_time = 0
        
        # Clear queue
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Voice processor reset")


class WebSocketAudioServer:
    """
    WebSocket server for real-time audio processing.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize WebSocket server.
        
        Args:
            host (str): Server host
            port (int): Server port
        """
        self.host = host
        self.port = port
        self.processor = None
        self.clients = set()
        
        logger.info(f"WebSocket server initialized on {host}:{port}")
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection."""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.clients.add(websocket)
        logger.info(f"Client connected: {client_id}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.process_message(data)
                    
                    if response:
                        await websocket.send(json.dumps(response))
                        
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "error": "Invalid JSON message"
                    }))
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await websocket.send(json.dumps({
                        "error": str(e)
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        finally:
            self.clients.discard(websocket)
    
    async def process_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process incoming WebSocket message."""
        message_type = data.get("type")
        
        if message_type == "init":
            # Initialize processor with settings
            settings = data.get("settings", {})
            self.processor = RealTimeVoiceProcessor(
                vad_aggressiveness=settings.get("vad_aggressiveness", 1),
                sample_rate=settings.get("sample_rate", 16000),
                whisper_model=settings.get("whisper_model", "base")
            )
            
            return {
                "type": "init_response",
                "status": "initialized",
                "settings": settings
            }
        
        elif message_type == "audio_chunk":
            if not self.processor:
                return {"error": "Processor not initialized"}
            
            audio_data = data.get("audio_data")
            if not audio_data:
                return {"error": "No audio data provided"}
            
            # Process audio chunk
            result = self.processor.process_audio_chunk(audio_data)
            result["type"] = "audio_processed"
            
            # Check for transcription results
            transcription_result = self.processor.process_transcription_queue()
            if transcription_result:
                # Send transcription as separate message
                transcription_result["type"] = "transcription"
                return transcription_result
            
            return result
        
        elif message_type == "get_stats":
            if not self.processor:
                return {"error": "Processor not initialized"}
            
            stats = self.processor.get_stats()
            stats["type"] = "stats"
            return stats
        
        elif message_type == "reset":
            if self.processor:
                self.processor.reset()
            
            return {
                "type": "reset_response",
                "status": "reset_complete"
            }
        
        else:
            return {"error": f"Unknown message type: {message_type}"}
    
    async def start_server(self):
        """Start the WebSocket server."""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info("WebSocket server started successfully")
            await asyncio.Future()  # Run forever
    
    def run_server(self):
        """Run the WebSocket server in the current thread."""
        asyncio.run(self.start_server())
    
    def run_server_threaded(self) -> threading.Thread:
        """Run the WebSocket server in a separate thread."""
        def server_thread():
            asyncio.run(self.start_server())
        
        thread = threading.Thread(target=server_thread, daemon=True)
        thread.start()
        logger.info("WebSocket server started in background thread")
        return thread


# Global server instance
_websocket_server = None

def get_websocket_server(host: str = "localhost", port: int = 8765) -> WebSocketAudioServer:
    """Get or create global WebSocket server instance."""
    global _websocket_server
    
    if _websocket_server is None:
        _websocket_server = WebSocketAudioServer(host, port)
    
    return _websocket_server

def start_websocket_server_background(host: str = "localhost", port: int = 8765) -> threading.Thread:
    """Start WebSocket server in background thread."""
    server = get_websocket_server(host, port)
    return server.run_server_threaded()
