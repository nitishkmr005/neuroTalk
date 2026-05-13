# NeuroTalk — Demo Script

> Read this aloud while recording your screen.
> Estimated recording time: ~90 seconds.
> Pause naturally at each `[pause]` marker.

---

## OPENING  *(show the landing screen)*

This is NeuroTalk — a real-time voice AI agent that runs entirely on your laptop.

No cloud. No API keys. No data leaving your machine. [pause]

---

## START A SESSION  *(click the orb)*

I'll click the orb to connect.

Instantly, the browser opens a WebRTC session to a local FastAPI backend, the AI greets me, and notice — the text appears in sync with the voice, sentence by sentence, as the audio plays. [pause]

---

## SPEAK NATURALLY  *(speak to the mic — ask something simple)*

I'll ask it a question.

Watch the chat — my words appear live while I'm still speaking. That's Whisper running on my CPU, transcribing in real time. [pause]

A streaming voice activity detector decides exactly when I've finished speaking and fires the LLM immediately — no button press, no delay. [pause]

---

## INTERRUPT IT  *(speak over the AI while it's responding)*

Now I'll talk over it mid-response.

It stopped instantly. The browser detected my voice, cancelled the audio queue, and the backend cancelled the LLM generation — both sides act independently so there's no waiting. [pause]

That's what makes it feel like a real conversation. [pause]

---

## CLOSING  *(show the full chat feed)*

In under two minutes you saw:

Live transcription. Sentence-streamed TTS. Instant turn detection. And clean interrupt handling. [pause]

The full stack — Whisper, Silero VAD, a local LLM via Ollama, and Kokoro TTS on Apple Silicon — all running locally, all open source.

The hard part wasn't the models. It was the orchestration — knowing when to speak, when to stop, and how to keep voice and text in sync.

---
