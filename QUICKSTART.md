# NeuroTalk Real-time - Quick Start Guide

## 🚀 Quick Start

### Using Make (Recommended)
```bash
make run
```

### Or Direct Command
```bash
uv run streamlit run app.py --server.port 8501 --server.address localhost
```

### Access the Application
Open your browser: **http://localhost:8501**

---

## 🎤 How to Use

1. **Grant Microphone Permission** - Your browser will ask for microphone access
2. **Click "Connect"** - Establishes WebSocket connection to the audio server
3. **Click "Start Recording"** - Begin real-time voice capture
4. **Speak Naturally** - Transcription appears automatically as you speak
5. **Click "Stop Recording"** - Stop when finished
6. **Use "Reset"** - Clear buffer and start fresh

---

## ⚙️ Configuration

### Sidebar Settings

- **VAD Aggressiveness (0-3)**
  - `0` = Most sensitive (detects more speech)
  - `1` = Balanced (recommended)
  - `2` = Moderate
  - `3` = Least sensitive (only clear speech)

- **Whisper Model**
  - `tiny` = Fastest, less accurate
  - `base` = Good balance (recommended)
  - `small` = More accurate, slower
  - `medium` = Very accurate, much slower

- **Sample Rate**
  - `16000 Hz` = Recommended for speech
  - `48000 Hz` = High quality

---

## 🔧 Troubleshooting

### Port Already in Use Error

If you see `OSError: [Errno 48] error while attempting to bind on address`:

```bash
# Kill processes on specific ports
lsof -ti:8501 | xargs kill -9
lsof -ti:8765 | xargs kill -9

# Or use make run (auto-cleanup)
make run
```

### Microphone Not Working

1. **Check macOS System Preferences**
   - System Settings → Privacy & Security → Microphone
   - Ensure your browser (Chrome/Firefox/Safari) is allowed

2. **Check Browser Permissions**
   - Chrome: Settings → Privacy and security → Site Settings → Microphone
   - Look for `http://localhost:8502` and ensure it's allowed

3. **Check Site-Specific Permissions**
   - Click the 🔒 or 🔓 icon in the address bar
   - Ensure Microphone is set to "Allow"

### WebSocket Connection Fails

1. **Ensure the WebSocket server started successfully**
   - Check the terminal output for error messages
   - Look for: "WebSocket server started on localhost:8765"

2. **Clear browser cache and reload**
   - Hard refresh: `Cmd + Shift + R` (Mac) or `Ctrl + Shift + R` (Windows/Linux)

3. **Check if port 8765 is available**
   ```bash
   lsof -i :8765
   ```

### No Speech Detected

1. **Lower VAD Aggressiveness** to 0 or 1
2. **Check your microphone volume** in system settings
3. **Try speaking louder and clearer**
4. **Ensure browser has microphone access**

---

## 📊 Application Tabs

### 🎙️ Recording Tab
- Real-time voice capture interface
- Audio component with Connect/Start/Stop/Reset buttons
- Live transcription display
- Connection status indicators

### 📊 Statistics Tab
- Total chunks processed
- Speech chunks detected
- Speech ratio percentage
- Buffer duration
- Audio amplitude levels

### 📝 History Tab
- View all transcriptions from current session
- Shows timestamp, language, duration for each
- Clear history button

---

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit web interface (port 8502)
- **WebSocket Server**: Real-time audio processing (port 8765)
- **VAD Engine**: WebRTC Voice Activity Detection
- **STT Engine**: OpenAI Whisper

### Audio Processing Pipeline
1. Browser captures microphone audio (Web Audio API)
2. Audio chunks sent via WebSocket (base64 encoded)
3. Server performs VAD on each chunk
4. Speech segments buffered and transcribed
5. Results sent back to browser in real-time

### Dependencies
- `streamlit` - Web UI framework
- `websockets` - WebSocket server
- `webrtcvad` - Voice activity detection
- `whisper` (OpenAI) - Speech-to-text
- `numpy`, `pydub` - Audio processing

---

## 🆘 Still Having Issues?

1. **Check the terminal output** for error messages
2. **Ensure all dependencies are installed**: `make install`
3. **Try the original non-realtime version**: `make run`
4. **Review logs** in the terminal for detailed error information

---

## 📝 Files

- `realtime_voice_app.py` - Main Streamlit application (fixed version)
- `components/realtime_audio_component.py` - WebSocket audio component
- `utils/websocket_server.py` - WebSocket server implementation
- `utils/vad.py` - Voice Activity Detection
- `utils/speech_to_text.py` - Whisper integration
- `utils/audio_utils.py` - Audio processing utilities

---

## 🎯 Features

✅ Real-time audio streaming via WebSocket
✅ Instant voice activity detection
✅ Live speech-to-text transcription
✅ Multiple language support (auto-detect)
✅ Configurable VAD sensitivity
✅ Multiple Whisper model options
✅ Transcription history tracking
✅ Live audio statistics
✅ Clean error handling
✅ Automatic port conflict resolution

---

## 📚 Comparison: Original vs Real-time

| Feature | Original (`app.py`) | Real-time (`realtime_voice_app.py`) |
|---------|---------------------|-------------------------------------|
| Audio Input | Fixed-duration recording | Continuous streaming |
| Latency | High (wait for recording) | Low (instant processing) |
| Port | 8501 | 8502 |
| Communication | File upload | WebSocket |
| VAD | Post-processing | Real-time |
| Transcription | Batch | Streaming |
| Use Case | File analysis | Live conversation |

---

**Enjoy real-time voice processing! 🎤✨**

