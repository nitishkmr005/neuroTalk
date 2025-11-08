# 🧠 NeuroTalk

**Real-time Voice Activity Detection & Speech-to-Text**

A modern WebSocket-based voice processing application with real-time speech recognition.

---

## ✨ Features

- 🎙️ **Real-time Audio Streaming** - Live microphone capture via WebSocket
- 🤖 **Voice Activity Detection** - Instant speech detection with WebRTC VAD
- 📝 **Speech-to-Text** - Accurate transcription with OpenAI Whisper
- 📊 **Live Statistics** - Real-time audio levels and processing metrics
- 🎨 **Modern UI** - Clean Streamlit interface with visual feedback
- ⚙️ **Configurable** - Adjustable VAD sensitivity and Whisper models

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
make install
```

### Running

```bash
# Start the application
make run

# Open browser at: http://localhost:8501
```

### Usage

1. Click **"Connect"** to establish WebSocket connection
2. Click **"🔴 Start Recording"** to begin audio capture
3. **Allow microphone access** when prompted
4. **Speak clearly** - transcription appears automatically
5. Click **"⏹️ Stop Recording"** when finished

---

## 📖 Configuration

**Environment Variables** (optional `.env` file):
```bash
WHISPER_MODEL_SIZE=base    # tiny|base|small|medium|large
VAD_AGGRESSIVENESS=2       # 0-3 (higher = stricter)
SAMPLE_RATE=16000          # Audio sample rate (Hz)
```

**Sidebar Settings:**
- **VAD Aggressiveness** (0-3): Speech detection sensitivity
- **Whisper Model**: Choose accuracy vs speed
- **Sample Rate**: Audio quality (16kHz recommended)

---

## 🧪 Testing

```bash
# Run automated tests
make test

# Interactive microphone test (requires speaking)
make test-mic
```

---

## 📂 Project Structure

```
neuroTalk/
├── app.py                  # Main application (start here)
├── QUICKSTART.md          # Detailed usage guide
├── README.md              # This file
├── Makefile               # Automation commands
├── pyproject.toml         # Dependencies
├── components/            # Streamlit components
├── static/                # JavaScript client
├── utils/                 # Core modules (VAD, STT, audio, WebSocket)
└── tests/                 # Test suite
```

---

## 🔧 Common Commands

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies |
| `make run` | Start the application |
| `make test` | Run test suite |
| `make test-mic` | Test your microphone |
| `make clean` | Clean cache files |

---

## 🐛 Troubleshooting

### Microphone Not Working
1. Check System Settings → Privacy & Security → Microphone
2. Enable for your browser (Chrome, Firefox, Safari)
3. Click 🔒 in address bar → Set Microphone to "Allow"

### No Speech Detected
1. Lower VAD aggressiveness to 0 or 1
2. Speak louder and closer to microphone
3. Check audio level bar is moving

### Port Already in Use
```bash
lsof -ti:8501 | xargs kill -9
lsof -ti:8765 | xargs kill -9
```

For more help, see [QUICKSTART.md](QUICKSTART.md)

---

## 🛠️ Technology Stack

- **Streamlit** - Web UI framework
- **WebSockets** - Real-time communication
- **OpenAI Whisper** - Speech recognition
- **WebRTC VAD** - Voice detection
- **PyAudio** - Audio I/O

---

## 📝 License

MIT License - See LICENSE file for details

---

**Built for seamless voice interaction** 🎤✨

For detailed documentation, see [QUICKSTART.md](QUICKSTART.md)
