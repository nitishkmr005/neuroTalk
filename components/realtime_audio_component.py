"""
Streamlit component for real-time audio processing with WebSocket.

This component provides a bridge between Streamlit and the WebSocket-based
real-time audio processing system.
"""

import streamlit as st
import streamlit.components.v1 as components
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


def realtime_audio_component(
    websocket_url: str = "ws://localhost:8765",
    vad_aggressiveness: int = 1,
    sample_rate: int = 16000,
    whisper_model: str = "base",
    key: Optional[str] = None  # Kept for compatibility but not used
) -> Optional[Dict[str, Any]]:
    """
    Create a real-time audio processing component.
    
    Args:
        websocket_url (str): WebSocket server URL
        vad_aggressiveness (int): VAD aggressiveness level (0-3)
        sample_rate (int): Audio sample rate
        whisper_model (str): Whisper model size
        key (Optional[str]): Unique key for the component
        
    Returns:
        Optional[Dict[str, Any]]: Component data or None
    """
    
    # Get the directory of this file
    component_dir = Path(__file__).parent.parent
    static_dir = component_dir / "static"
    
    # Read the JavaScript file
    js_file = static_dir / "realtime_audio.js"
    
    if not js_file.exists():
        st.error(f"JavaScript file not found: {js_file}")
        return None
    
    with open(js_file, 'r') as f:
        js_code = f.read()
    
    # Configuration for the component
    config = {
        "websocketUrl": websocket_url,
        "vadAggressiveness": vad_aggressiveness,
        "sampleRate": sample_rate,
        "whisperModel": whisper_model
    }
    
    # Convert config to JSON string
    config_json = json.dumps(config)
    
    # Create the HTML template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Real-time Audio Component</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #fafafa;
            }}
            
            .audio-component {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                max-width: 600px;
            }}
            
            .status-indicator {{
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }}
            
            .status-connected {{ background-color: #4CAF50; }}
            .status-disconnected {{ background-color: #f44336; }}
            .status-recording {{ background-color: #FF9800; animation: pulse 1s infinite; }}
            
            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
                100% {{ opacity: 1; }}
            }}
            
            .controls {{
                margin: 20px 0;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }}
            
            .btn {{
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: background-color 0.2s;
            }}
            
            .btn-primary {{
                background-color: #1976D2;
                color: white;
            }}
            
            .btn-primary:hover {{
                background-color: #1565C0;
            }}
            
            .btn-secondary {{
                background-color: #757575;
                color: white;
            }}
            
            .btn-secondary:hover {{
                background-color: #616161;
            }}
            
            .btn:disabled {{
                background-color: #E0E0E0;
                color: #9E9E9E;
                cursor: not-allowed;
            }}
            
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            
            .stat-item {{
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            }}
            
            .stat-value {{
                font-size: 18px;
                font-weight: bold;
                color: #1976D2;
            }}
            
            .stat-label {{
                font-size: 12px;
                color: #666;
                margin-top: 4px;
            }}
            
            .transcription {{
                background-color: #E3F2FD;
                border-left: 4px solid #1976D2;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
                min-height: 60px;
            }}
            
            .transcription-text {{
                font-size: 16px;
                line-height: 1.5;
                color: #333;
            }}
            
            .transcription-empty {{
                color: #666;
                font-style: italic;
            }}
            
            .error {{
                background-color: #FFEBEE;
                border-left: 4px solid #f44336;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
                color: #D32F2F;
            }}
            
            .audio-level {{
                margin: 15px 0;
            }}
            
            .level-bar {{
                width: 100%;
                height: 20px;
                background-color: #E0E0E0;
                border-radius: 10px;
                overflow: hidden;
            }}
            
            .level-fill {{
                height: 100%;
                background: linear-gradient(90deg, #4CAF50, #FF9800, #f44336);
                width: 0%;
                transition: width 0.1s ease;
            }}
        </style>
    </head>
    <body>
        <div class="audio-component">
            <h3>🎤 Real-time Voice Processing</h3>
            
            <div id="status">
                <span class="status-indicator status-disconnected" id="statusIndicator"></span>
                <span id="statusText">Ready - Click Connect to start</span>
            </div>
            
            <div class="controls">
                <button class="btn btn-primary" id="connectBtn" onclick="connect()">Connect</button>
                <button class="btn btn-primary" id="startBtn" onclick="startRecording()" disabled>🔴 Start Recording</button>
                <button class="btn btn-secondary" id="stopBtn" onclick="stopRecording()" disabled>⏹️ Stop Recording</button>
                <button class="btn btn-secondary" id="resetBtn" onclick="reset()" disabled>🔄 Reset</button>
            </div>
            
            <div class="audio-level">
                <div>Audio Level:</div>
                <div class="level-bar">
                    <div class="level-fill" id="audioLevel"></div>
                </div>
            </div>
            
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value" id="totalChunks">0</div>
                    <div class="stat-label">Total Chunks</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="speechChunks">0</div>
                    <div class="stat-label">Speech Chunks</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="speechRatio">0%</div>
                    <div class="stat-label">Speech Ratio</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="bufferDuration">0.0s</div>
                    <div class="stat-label">Buffer Duration</div>
                </div>
            </div>
            
            <div class="transcription">
                <div class="transcription-text" id="transcriptionText">
                    <span class="transcription-empty">Transcription will appear here...</span>
                </div>
            </div>
            
            <div id="errorMessage" class="error" style="display: none;"></div>
        </div>
        
        <script>
            {js_code}
            
            // Component state
            let audioComponent = null;
            let isRecording = false;
            let isConnected = false;
            
            // Configuration
            const config = {config_json};
            
            // DOM elements
            const statusIndicator = document.getElementById('statusIndicator');
            const statusText = document.getElementById('statusText');
            const connectBtn = document.getElementById('connectBtn');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const resetBtn = document.getElementById('resetBtn');
            const audioLevel = document.getElementById('audioLevel');
            const transcriptionText = document.getElementById('transcriptionText');
            const errorMessage = document.getElementById('errorMessage');
            
            // Statistics elements
            const totalChunks = document.getElementById('totalChunks');
            const speechChunks = document.getElementById('speechChunks');
            const speechRatio = document.getElementById('speechRatio');
            const bufferDuration = document.getElementById('bufferDuration');
            
            // Initialize component
            async function connect() {{
                try {{
                    console.log('[InstaVoice] Connect button clicked');
                    statusText.textContent = 'Connecting...';
                    connectBtn.disabled = true;
                    
                    console.log('[InstaVoice] Creating StreamlitAudioComponent...');
                    audioComponent = new StreamlitAudioComponent();
                    
                    console.log('[InstaVoice] Initializing audio component with config:', config);
                    // Initialize first to create the processor
                    const success = await audioComponent.initialize(config);
                    
                    console.log('[InstaVoice] Initialization result:', success);
                    if (!success) {{
                        throw new Error('Failed to initialize audio component');
                    }}
                    
                    // Now set up event handlers after processor is created
                    if (audioComponent.processor) {{
                        audioComponent.processor.onConnectionChange = (connected) => {{
                            console.log('[InstaVoice] onConnectionChange callback:', connected);
                            isConnected = connected;
                            updateConnectionStatus(connected);
                            sendToStreamlit('connection_change', {{ connected }});
                        }};
                        
                        audioComponent.processor.onTranscription = (data) => {{
                            updateTranscription(data);
                            sendToStreamlit('transcription', data);
                        }};
                        
                        audioComponent.processor.onAudioProcessed = (data) => {{
                            updateStats(data);
                            updateAudioLevel(data.max_amplitude || 0);
                            sendToStreamlit('audio_processed', data);
                        }};
                        
                        audioComponent.processor.onError = (error) => {{
                            showError(error);
                            sendToStreamlit('error', {{ message: error }});
                        }};
                    }}
                    
                    // Manually update UI to connected state (since connection already happened)
                    console.log('[InstaVoice] Manually updating UI to connected state');
                    isConnected = true;
                    updateConnectionStatus(true);
                    sendToStreamlit('connection_change', {{ connected: true }});
                    
                }} catch (error) {{
                    console.error('Connection failed:', error);
                    showError(`Connection failed: ${{error.message}}`);
                    connectBtn.disabled = false;
                    statusText.textContent = 'Connection failed';
                }}
            }}
            
            async function startRecording() {{
                try {{
                    const success = await audioComponent.startRecording();
                    if (success) {{
                        isRecording = true;
                        updateRecordingStatus(true);
                        sendToStreamlit('recording_started', {{}});
                    }}
                }} catch (error) {{
                    showError(`Recording failed: ${{error.message}}`);
                }}
            }}
            
            function stopRecording() {{
                if (audioComponent) {{
                    audioComponent.stopRecording();
                    isRecording = false;
                    updateRecordingStatus(false);
                    sendToStreamlit('recording_stopped', {{}});
                }}
            }}
            
            function reset() {{
                if (audioComponent) {{
                    audioComponent.reset();
                    clearTranscription();
                    clearStats();
                    sendToStreamlit('reset', {{}});
                }}
            }}
            
            function updateConnectionStatus(connected) {{
                console.log('[InstaVoice] Updating connection status:', connected);
                if (connected) {{
                    console.log('[InstaVoice] Setting UI to connected state');
                    statusIndicator.className = 'status-indicator status-connected';
                    statusText.textContent = 'Connected';
                    startBtn.disabled = false;
                    resetBtn.disabled = false;
                    connectBtn.style.display = 'none';
                    console.log('[InstaVoice] Start button enabled:', !startBtn.disabled);
                }} else {{
                    console.log('[InstaVoice] Setting UI to disconnected state');
                    statusIndicator.className = 'status-indicator status-disconnected';
                    statusText.textContent = 'Disconnected';
                    startBtn.disabled = true;
                    stopBtn.disabled = true;
                    resetBtn.disabled = true;
                    connectBtn.style.display = 'inline-block';
                    connectBtn.disabled = false;
                }}
            }}
            
            function updateRecordingStatus(recording) {{
                if (recording) {{
                    statusIndicator.className = 'status-indicator status-recording';
                    statusText.textContent = 'Recording...';
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                }} else {{
                    statusIndicator.className = 'status-indicator status-connected';
                    statusText.textContent = 'Connected';
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                }}
            }}
            
            function updateTranscription(data) {{
                if (data.text && data.text.trim()) {{
                    transcriptionText.innerHTML = `
                        <strong>[${{data.language || 'unknown'}}]</strong> ${{data.text}}
                        <br><small>Duration: ${{(data.duration || 0).toFixed(2)}}s</small>
                    `;
                }}
            }}
            
            function clearTranscription() {{
                transcriptionText.innerHTML = '<span class="transcription-empty">Transcription will appear here...</span>';
            }}
            
            function updateStats(data) {{
                if (data.total_chunks !== undefined) totalChunks.textContent = data.total_chunks;
                if (data.speech_chunks !== undefined) speechChunks.textContent = data.speech_chunks;
                if (data.buffer_duration !== undefined) bufferDuration.textContent = data.buffer_duration.toFixed(1) + 's';
                
                // Calculate speech ratio
                const total = parseInt(totalChunks.textContent) || 0;
                const speech = parseInt(speechChunks.textContent) || 0;
                const ratio = total > 0 ? (speech / total * 100).toFixed(1) : 0;
                speechRatio.textContent = ratio + '%';
            }}
            
            function clearStats() {{
                totalChunks.textContent = '0';
                speechChunks.textContent = '0';
                speechRatio.textContent = '0%';
                bufferDuration.textContent = '0.0s';
                updateAudioLevel(0);
            }}
            
            function updateAudioLevel(amplitude) {{
                const percentage = Math.min(100, amplitude * 100);
                audioLevel.style.width = percentage + '%';
            }}
            
            function showError(message) {{
                errorMessage.textContent = message;
                errorMessage.style.display = 'block';
                setTimeout(() => {{
                    errorMessage.style.display = 'none';
                }}, 5000);
            }}
            
            function sendToStreamlit(eventType, data) {{
                if (window.Streamlit) {{
                    window.Streamlit.setComponentValue({{
                        event: eventType,
                        data: data,
                        timestamp: Date.now()
                    }});
                }}
            }}
            
            // Initialize Streamlit communication
            if (window.Streamlit) {{
                window.Streamlit.setComponentReady();
                window.Streamlit.setFrameHeight(700);  // Increased from 500 to 700
            }}
            
            // Note: No auto-connect - user must click Connect button
            // This ensures proper user interaction for microphone permissions
        </script>
    </body>
    </html>
    """
    
    # Render the component with increased height
    component_value = components.html(
        html_content,
        height=700  # Increased from 500 to 700
    )
    
    return component_value
