"""
NeuroTalk Real-time - WebSocket-based Voice Activity Detection and Speech-to-Text.

=================================================================================
                        FUNCTION CALL MINDMAP
=================================================================================

                            [USER OPENS APP]
                                  |
                                  v
                            main() ──────────────────────────────────┐
                              |                                       |
                              |                                       |
            ┌─────────────────┼───────────────────┐                 |
            |                 |                   |                 |
            v                 v                   v                 |
    init_session_state()  render_settings()  start_websocket_server()
    │                     │                   │                     |
    │ Creates:            │ Returns:          │ Starts:             |
    │ - transcriptions[]  │ - vad_level       │ - WebSocket         |
    │ - audio_stats{}     │ - whisper_model   │   server thread     |
    │ - connection_status │ - sample_rate     │ - Port 8765         |
                          │                   │                     |
                          │                   v                     |
                          │         start_websocket_server_background()
                          │         (websocket_server.py)           |
                          │                   │                     |
                          │                   v                     |
                          │         WebSocketAudioServer.__init__() |
                          │                   │                     |
                          │                   v                     |
                          │         WebSocketAudioServer.run_server()
                          │                   │                     |
                          │                   v                     |
                          │         WebSocketAudioServer.handle_client()
                          │                   │                     |
                          └───────────────────┼─────────────────────┘
                                              |
                                              v
                          realtime_audio_component() ←──────┐
                          (renders HTML/JS iframe)          |
                                  |                         |
                ┌─────────────────┴──────────────┐         |
                |                                |         |
        [USER CLICKS "CONNECT"]        [USER CLICKS "START RECORDING"]
                |                                |
                v                                v
        JavaScript: connectWebSocket()   JavaScript: startRecording()
                |                                |
                v                                |
        Opens ws://localhost:8765                |
                |                                |
                v                                v
        WebSocket.onopen                 navigator.mediaDevices.getUserMedia()
                |                                |
                |                                v
                |                        AudioWorkletProcessor.process()
                |                                |
                |                                v
                |                        [Captures audio samples]
                |                                |
                |                                v
                |                        sendAudioChunk(base64Audio)
                |                                |
                v                                v
        ┌───────────────────────────────────────────────────────┐
        │    WebSocketAudioServer.handle_client()               │
        │    (receives WebSocket message)                       │
        └───────────────────────────────────────────────────────┘
                                |
                                v
                    WebSocketAudioServer.process_message()
                                |
                ┌───────────────┴───────────────┐
                |                               |
        [type: "init"]                 [type: "audio_chunk"]
                |                               |
                v                               v
        Initialize processor        RealTimeVoiceProcessor.process_audio_chunk()
                |                               |
                |                               v
                |                   VoiceActivityDetector.process_audio_chunks()
                |                               |
                |               ┌───────────────┴───────────────┐
                |               |                               |
                |           [Speech Detected]           [No Speech / Silence]
                |               |                               |
                |               v                               v
                |       Buffer audio                    [After >1s silence]
                |       Send stats back                         |
                |               |                               v
                |               |               SpeechToTextProcessor.transcribe_audio()
                |               |                       (Whisper Model)
                |               |                               |
                |               |                               v
                |               |                   {text, language, duration}
                |               |                               |
                |               |                               v
                |               |                   Send via WebSocket
                |               |                               |
                |               └───────────────┬───────────────┘
                |                               |
                v                               v
        ┌───────────────────────────────────────────────────────┐
        │    JavaScript: WebSocket.onmessage                    │
        │    (receives transcription or stats)                  │
        └───────────────────────────────────────────────────────┘
                                |
                ┌───────────────┴───────────────┐
                |                               |
        [type: "transcription"]        [type: "audio_processed"]
                |                               |
                v                               v
        onTranscription(data)          onAudioProcessed(data)
                |                               |
                v                               v
        Streamlit.setComponentValue()  Streamlit.setComponentValue()
        {event: "transcription"}       {event: "audio_processed"}
                |                               |
                └───────────────┬───────────────┘
                                |
                                v
                ┌───────────────────────────────────────┐
                │  handle_component_event(event_data)   │
                │  (THIS FILE - app.py)                 │
                └───────────────────────────────────────┘
                                |
                ┌───────────────┴───────────────┐
                |                               |
        [event: "transcription"]       [event: "audio_processed"]
                |                               |
                v                               v
        Update session_state           Update session_state
        .transcriptions[]              .audio_stats{}
                |                               |
                v                               v
        st.success(text)               (stats updated silently)
                |                               |
                v                               v
        render_transcription_history()  render_statistics()
        (displays in "History" tab)     (displays in "Statistics" tab)

=================================================================================

=== ARCHITECTURE OVERVIEW ===

This application uses a WebSocket-based architecture for real-time voice processing:

FLOW: Browser → WebSocket → Python → Whisper → WebSocket → Browser → Streamlit

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. BROWSER (JavaScript - realtime_audio.js)                                │
│    - Captures microphone audio using Web Audio API                          │
│    - Converts audio to PCM format (16-bit, 16kHz)                          │
│    - Sends audio chunks via WebSocket every ~100ms                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. WEBSOCKET SERVER (Python - websocket_server.py)                         │
│    - Receives audio chunks from browser                                     │
│    - Decodes base64 audio data                                             │
│    - Feeds chunks to VAD (Voice Activity Detection)                        │
│    - Buffers speech segments                                               │
│    - Triggers Whisper transcription when silence detected                  │
│    - Sends transcription back to browser                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. BROWSER (JavaScript - receives transcription)                           │
│    - Receives transcription from WebSocket                                  │
│    - Sends event to Streamlit via Streamlit.setComponentValue()           │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. STREAMLIT (Python - app.py - THIS FILE)                                │
│    - Receives event from component                                          │
│    - Updates session state with transcription                              │
│    - Displays in UI (tabs: Recording, Statistics, History)                 │
└─────────────────────────────────────────────────────────────────────────────┘

KEY COMPONENTS:
- realtime_audio_component.py: Streamlit custom component (HTML/JS bridge)
- realtime_audio.js: Browser-side audio capture and WebSocket client
- websocket_server.py: Python WebSocket server with VAD and Whisper
- app.py (this file): Streamlit UI and event handling
"""

import streamlit as st
import time
from typing import Dict, Any, Optional
from loguru import logger

# Import our custom modules
from utils.config import config
from utils.websocket_server import start_websocket_server_background
from components.realtime_audio_component import realtime_audio_component


def init_session_state():
    """
    Initialize Streamlit session state variables.
    
    Session state persists across Streamlit reruns and stores:
    - websocket_started: Flag indicating if WebSocket server is running
    - transcriptions: List of all transcribed text with metadata
    - audio_stats: Real-time statistics (chunks processed, speech detected, etc.)
    - connection_status: Boolean indicating if browser is connected to WebSocket
    """
    if 'websocket_started' not in st.session_state:
        st.session_state.websocket_started = False
    if 'transcriptions' not in st.session_state:
        st.session_state.transcriptions = []
    if 'audio_stats' not in st.session_state:
        st.session_state.audio_stats = {
            'total_chunks': 0,
            'speech_chunks': 0,
            'buffer_duration': 0.0,
            'max_amplitude': 0.0
        }
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = False


def start_websocket_server():
    """
    Start the Python WebSocket server in a background thread.
    
    STEP 1 of the audio processing pipeline:
    - Creates a WebSocket server listening on ws://localhost:8765
    - Server runs in a separate thread to not block Streamlit
    - Waits for browser connections to send/receive audio data
    
    Returns:
        bool: True if server started successfully or already running
    """
    if not st.session_state.websocket_started:
        try:
            with st.spinner("Starting WebSocket server..."):
                start_websocket_server_background(host="localhost", port=8765)
                time.sleep(2)
            
            st.session_state.websocket_started = True
            st.success("✅ WebSocket server started on ws://localhost:8765")
            return True
        except OSError as e:
            if "address already in use" in str(e).lower():
                # Server is already running, mark as started
                st.session_state.websocket_started = True
                st.info("ℹ️ WebSocket server already running")
                return True
            else:
                st.error(f"❌ Failed to start WebSocket server: {e}")
                logger.error(f"WebSocket server error: {e}")
                return False
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            logger.error(f"Unexpected error: {e}")
            return False
    
    return True


def render_settings():
    """Render sidebar settings and return configuration."""
    st.sidebar.title("🎤 NeuroTalk")
    
    # Server status
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 Status")
    
    if st.session_state.websocket_started:
        st.sidebar.success("✅ WebSocket Server\n\nRunning on port 8765")
    else:
        st.sidebar.error("⚠️ Server Stopped")
    
    if st.session_state.connection_status:
        st.sidebar.success("🔗 Client Connected")
    else:
        st.sidebar.info("🔌 Disconnected")
    
    # Settings
    st.sidebar.subheader("⚙️ Settings")
    
    vad_aggressiveness = st.sidebar.slider(
        "VAD Aggressiveness",
        min_value=0,
        max_value=3,
        value=1,
        help="0=Most sensitive, 3=Least sensitive"
    )
    
    whisper_model = st.sidebar.selectbox(
        "Whisper Model",
        options=["tiny", "base", "small", "medium"],
        index=1,
        help="Larger models = more accurate but slower"
    )
    
    sample_rate = st.sidebar.selectbox(
        "Sample Rate (Hz)",
        options=[8000, 16000, 32000, 48000],
        index=1
    )
    
    return {
        "vad_aggressiveness": vad_aggressiveness,
        "whisper_model": whisper_model,
        "sample_rate": sample_rate
    }


def handle_component_event(event_data: Optional[Dict[str, Any]]):
    """
    Handle events from the JavaScript component.
    
    STEP 4 of the audio processing pipeline - FINAL STEP:
    This is where the transcription results arrive back in Python/Streamlit!
    
    EVENT FLOW:
    1. Browser JavaScript receives transcription from WebSocket server
    2. JavaScript calls: Streamlit.setComponentValue({event: "transcription", data: {...}})
    3. Streamlit detects the component value changed
    4. This function is called with the event data
    5. We update session state and UI
    
    EVENT TYPES:
    - "transcription": Final text from Whisper (triggered after speech + silence)
    - "audio_processed": Real-time stats (sent ~every 100ms during recording)
    - "connection_change": WebSocket connection status
    - "recording_started": User clicked "Start Recording"
    - "recording_stopped": User clicked "Stop Recording"
    - "error": Any error from browser/WebSocket
    
    Args:
        event_data: Dictionary with 'event' (type) and 'data' (payload) keys
    """
    if not event_data:
        return
    
    # Safely extract event type and data from the component
    event_type = event_data.get("event", "") if isinstance(event_data, dict) else ""
    data = event_data.get("data", {}) if isinstance(event_data, dict) else {}
    
    logger.debug(f"Event received: {event_type}, data keys: {data.keys() if isinstance(data, dict) else 'not a dict'}")
    
    # === TRANSCRIPTION EVENT ===
    # This is the final result after:
    # 1. Browser captured audio
    # 2. WebSocket server received chunks
    # 3. VAD detected speech
    # 4. Whisper transcribed the audio
    # 5. Result sent back via WebSocket
    if event_type == "transcription":
        text = data.get("text", "") if isinstance(data, dict) else ""
        if text and text.strip():
            transcription = {
                "text": text,
                "language": data.get("language", "en") if isinstance(data, dict) else "en",
                "duration": data.get("duration", 0) if isinstance(data, dict) else 0,
                "timestamp": time.strftime("%H:%M:%S")
            }
            st.session_state.transcriptions.append(transcription)
            logger.info(f"Transcription added: {len(st.session_state.transcriptions)} total")
            st.success(f"🗣️ **[{transcription['language'].upper()}]** {transcription['text']}")
    
    # === AUDIO PROCESSED EVENT ===
    # Sent continuously (~every 100ms) while recording
    # Contains real-time statistics from the WebSocket server:
    # - total_chunks: Number of audio chunks received
    # - speech_chunks: Number of chunks containing speech (VAD detected)
    # - buffer_duration: Current audio buffer size in seconds
    # - max_amplitude: Peak audio level (for visualizations)
    elif event_type == "audio_processed":
        if isinstance(data, dict):
            # Update session state with latest statistics
            if 'total_chunks' in data:
                st.session_state.audio_stats['total_chunks'] = data.get('total_chunks', 0)
            if 'speech_chunks' in data:
                st.session_state.audio_stats['speech_chunks'] = data.get('speech_chunks', 0)
            if 'buffer_duration' in data:
                st.session_state.audio_stats['buffer_duration'] = data.get('buffer_duration', 0.0)
            if 'max_amplitude' in data:
                st.session_state.audio_stats['max_amplitude'] = data.get('max_amplitude', 0.0)
            
            logger.debug(f"Stats updated: {st.session_state.audio_stats}")
    
    elif event_type == "connection_change":
        connected = data.get("connected", False) if isinstance(data, dict) else False
        st.session_state.connection_status = connected
        
        if connected:
            st.info("🔗 Connected to WebSocket server")
        else:
            st.warning("🔌 Disconnected from WebSocket server")
    
    elif event_type == "error":
        error_msg = data.get("message", "Unknown error") if isinstance(data, dict) else "Unknown error"
        st.error(f"❌ Error: {error_msg}")
    
    elif event_type == "recording_started":
        st.info("🔴 Recording started - speak now!")
    
    elif event_type == "recording_stopped":
        st.info("⏹️ Recording stopped")
    
    elif event_type == "reset":
        st.session_state.transcriptions = []
        st.session_state.audio_stats = {
            'total_chunks': 0,
            'speech_chunks': 0,
            'buffer_duration': 0.0,
            'max_amplitude': 0.0
        }
        st.success("🔄 System reset")


def render_statistics():
    """Render live statistics."""
    st.subheader("📊 Live Statistics")
    
    stats = st.session_state.audio_stats
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Chunks", stats.get("total_chunks", 0))
    
    with col2:
        st.metric("Speech Chunks", stats.get("speech_chunks", 0))
    
    with col3:
        total = stats.get("total_chunks", 0)
        speech = stats.get("speech_chunks", 0)
        ratio = (speech / total * 100) if total > 0 else 0
        st.metric("Speech Ratio", f"{ratio:.1f}%")
    
    with col4:
        st.metric("Buffer", f"{stats.get('buffer_duration', 0):.1f}s")
    
    # Additional stats
    st.markdown("---")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.metric("Max Amplitude", f"{stats.get('max_amplitude', 0):.3f}")
    
    with col6:
        total_transcriptions = len(st.session_state.transcriptions)
        st.metric("Transcriptions", total_transcriptions)
    
    if stats.get("total_chunks", 0) == 0:
        st.info("💡 **Tip:** Start recording to see real-time audio statistics.")


def render_transcription_history():
    """Render transcription history."""
    st.subheader("📝 Transcription History")
    
    total = len(st.session_state.transcriptions)
    
    # Show count at top
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric("Total Transcriptions", total)
    with col2:
        if total > 0:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.transcriptions = []
                st.success("✅ History cleared!")
                st.rerun()
    
    st.markdown("---")
    
    if st.session_state.transcriptions:
        # Show all transcriptions (most recent first)
        st.write("**Recent Transcriptions:**")
        
        for i, trans in enumerate(reversed(st.session_state.transcriptions)):
            with st.expander(
                f"🗣️ #{total - i} • {trans['timestamp']} • [{trans['language'].upper()}] • {trans.get('duration', 0):.1f}s",
                expanded=(i < 2)  # Auto-expand first 2
            ):
                st.markdown(f"**Text:** {trans['text']}")
                st.caption(f"Recorded at: {trans['timestamp']}")
        
        # Show summary at bottom
        if total > 5:
            st.info(f"📊 Showing all {total} transcriptions. Scroll up to see older entries.")
    else:
        st.info("📝 **No transcriptions yet.**\n\nStart recording in the **Recording** tab and speak to see your transcriptions appear here.")


def main():
    """Main function to run the real-time NeuroTalk application."""
    
    # Page config
    st.set_page_config(
        page_title="NeuroTalk - Real-time Voice Processing",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Clean modern styling - simplified and cleaner
    st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 12px 24px;
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white !important;
        border: 1px solid #6366f1;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stMetricValue"] {
        color: #6366f1 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Title and description
    st.title("🧠 NeuroTalk - Real-time Voice Processing")
    st.markdown("**Real-time Voice Activity Detection & Speech-to-Text**")
    st.markdown("---")
    
    # Render sidebar and get settings
    settings = render_settings()
    
    # Start WebSocket server
    if not start_websocket_server():
        st.error("Cannot start real-time processing without WebSocket server")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎙️ Recording", "📊 Statistics", "📝 History"])
    
    with tab1:
        st.subheader("🎙️ Real-time Voice Processing")
        
        st.info("""
        💡 **How to use:**
        1. Click **"Connect"** button below to establish WebSocket connection
        2. Once connected, click **"🔴 Start Recording"** to begin capturing audio
        3. **Allow microphone access** when your browser asks for permission
        4. **Speak clearly** - transcription will appear automatically
        5. Click **"⏹️ Stop Recording"** when finished
        """)
        
        # === STEP 2 & 3: RENDER THE AUDIO COMPONENT ===
        # This creates an iframe with JavaScript that:
        # 1. Captures microphone audio in the browser (Web Audio API)
        # 2. Connects to WebSocket server at ws://localhost:8765
        # 3. Sends audio chunks to Python server
        # 4. Receives transcriptions back from server
        # 5. Sends events to Streamlit via setComponentValue()
        #
        # The component returns data whenever JavaScript calls setComponentValue()
        # This happens for: transcription, audio_processed, connection_change, etc.
        try:
            component_data = realtime_audio_component(
                websocket_url="ws://localhost:8765",  # Where Python WebSocket server is listening
                vad_aggressiveness=settings["vad_aggressiveness"],  # Passed to server
                sample_rate=settings["sample_rate"],  # Audio quality (16kHz recommended)
                whisper_model=settings["whisper_model"]  # Which Whisper model to use
            )
            
            # === STEP 4: HANDLE EVENTS FROM COMPONENT ===
            # When component_data changes (JavaScript sent new data), process it
            # This is where transcriptions arrive back in Python!
            handle_component_event(component_data)
        
        except Exception as e:
            st.error(f"Component error: {e}")
            logger.error(f"Component error: {e}", exc_info=True)
    
    with tab2:
        render_statistics()
    
    with tab3:
        render_transcription_history()


if __name__ == "__main__":
    """
    =============================================================================
    COMPLETE DATA FLOW SUMMARY - From Browser to Transcription
    =============================================================================
    
    USER ACTION: User clicks "Start Recording" and speaks
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ STEP 1: BROWSER CAPTURES AUDIO (JavaScript - realtime_audio.js)        │
    │─────────────────────────────────────────────────────────────────────────│
    │ 1. Web Audio API accesses microphone                                    │
    │ 2. AudioWorkletProcessor captures raw audio samples                     │
    │ 3. Converts to PCM 16-bit format at 16kHz sample rate                  │
    │ 4. Chunks audio into ~100ms segments                                    │
    │ 5. Base64 encodes the audio data                                        │
    │ 6. Sends via WebSocket:                                                 │
    │    ws.send(JSON.stringify({                                             │
    │      type: "audio_chunk",                                               │
    │      audio_data: base64EncodedAudio                                     │
    │    }))                                                                   │
    └─────────────────────────────────────────────────────────────────────────┘
                                      ↓ WebSocket
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ STEP 2: PYTHON SERVER PROCESSES (websocket_server.py)                  │
    │─────────────────────────────────────────────────────────────────────────│
    │ 1. WebSocket server receives message on port 8765                       │
    │ 2. Decodes base64 audio to bytes                                        │
    │ 3. Converts bytes to NumPy array (float32)                             │
    │ 4. Feeds to VAD (Voice Activity Detection):                            │
    │    - WebRTC VAD analyzes 30ms frames                                    │
    │    - Returns True/False for each frame (speech detected?)              │
    │ 5. If speech detected:                                                  │
    │    - Adds audio to buffer                                              │
    │    - Updates statistics (total_chunks, speech_chunks, etc.)            │
    │    - Sends stats back to browser via WebSocket                         │
    │ 6. When silence detected after speech (>1 second):                     │
    │    - Triggers Whisper transcription on buffered audio                  │
    │    - Whisper returns: {text, language, duration}                       │
    │    - Sends transcription back to browser via WebSocket                 │
    │    - Clears buffer for next speech segment                             │
    └─────────────────────────────────────────────────────────────────────────┘
                                      ↓ WebSocket
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ STEP 3: BROWSER RECEIVES RESULT (JavaScript)                           │
    │─────────────────────────────────────────────────────────────────────────│
    │ 1. WebSocket onmessage handler receives:                                │
    │    {type: "transcription", text: "hello world", language: "en"}        │
    │ 2. Updates UI (shows transcription in component)                        │
    │ 3. Sends to Streamlit via:                                              │
    │    Streamlit.setComponentValue({                                        │
    │      event: "transcription",                                            │
    │      data: {text, language, duration},                                  │
    │      timestamp: Date.now()                                              │
    │    })                                                                    │
    └─────────────────────────────────────────────────────────────────────────┘
                                      ↓ Streamlit Bridge
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ STEP 4: STREAMLIT UI UPDATES (app.py - THIS FILE)                      │
    │─────────────────────────────────────────────────────────────────────────│
    │ 1. Streamlit detects component value changed                            │
    │ 2. Calls: handle_component_event(event_data)                           │
    │ 3. Extracts transcription text                                          │
    │ 4. Updates session_state.transcriptions list                           │
    │ 5. Displays in UI:                                                      │
    │    - Shows success message with text                                    │
    │    - Updates "History" tab                                              │
    │    - Updates counters                                                   │
    └─────────────────────────────────────────────────────────────────────────┘
    
    RESULT: User sees their spoken words as text in the Streamlit UI!
    
    TIMING:
    - Audio chunks sent every ~100ms
    - VAD processes each chunk in <5ms
    - Transcription triggered after 1 second of silence
    - Whisper transcription takes 0.5-2 seconds (depending on model)
    - Total latency: ~2-4 seconds from speech end to displayed text
    
    =============================================================================
    """
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}", exc_info=True)

