# NeuroTalk — Demo Narration Script

> Read this aloud while walking through the UI.
> Estimated recording time: ~3 minutes.

---

## INTRO  *(show the landing screen)*

Hey everyone — this is NeuroTalk, a fully local, real-time voice AI agent I built from scratch.

Everything you're about to see runs entirely on my laptop. No cloud, no API keys, nothing leaving the machine.

---

## CLICKING THE ORB  *(click the orb)*

I'll click this orb to start a session.

The moment I click, the browser creates an audio context, requests microphone access, and opens a WebSocket connection to a local FastAPI backend.

You can see the welcome message appear in the chat — the AI greets me — and the text is appearing in sync with the voice. Not all at once. Character by character, as the audio plays.

---

## FIRST QUESTION  *(speak naturally to the mic)*

Let me ask it something simple — I'll say: *"Hi, how are you?"*

Watch the chat feed. As I'm speaking, my words appear live — that's the Whisper speech-to-text model running on my CPU, re-transcribing my audio every few hundred milliseconds in real time.

The AI picks up the transcript, generates a response through a local LLM running in Ollama, and converts it to voice using the Kokoro TTS model.

Notice the text in the AI bubble — it's revealing itself in step with the spoken words. The voice and the text are perfectly in sync.

---

## LATENCY DEMO  *(ask a short follow-up)*

Now I'll ask: *"What time is it in Tokyo right now?"*

Notice how fast the AI starts responding. That's because the system doesn't wait for me to fully stop talking. It detects a pause in my speech — about 900 milliseconds of silence — and fires the LLM immediately. By the time I've finished my sentence, the response is already streaming back.

---

## WEB SEARCH  *(ask a news or current events question)*

Let me ask something that requires live information — *"What are the latest AI news today?"*

Behind the scenes, a keyword classifier detects words like "latest" and "today" and fires a DuckDuckGo web search in parallel before the LLM even starts. By the time the language model is ready to respond, the search results are already injected into its context. It cites the sources naturally in speech — no URLs read out loud.

---

## BARGE-IN / INTERRUPT  *(interrupt the AI mid-response)*

Here's one of my favourite features — barge-in. I'm going to interrupt the AI while it's still speaking.

*(Wait for AI to start speaking, then speak over it)*

I just spoke over the AI and it stopped immediately — no awkward overlap, no waiting for it to finish. The browser detected the spike in my voice volume, sent an interrupt signal to the backend, and cancelled the running response. It's already listening for my next question.

That's what makes this feel like an actual conversation rather than a push-to-talk demo.

---

## CONVERSATION MEMORY  *(ask a follow-up referencing previous context)*

Now let me ask a follow-up that relies on what we talked about earlier.

*(Reference something from earlier in the conversation)*

It remembered the context from our earlier exchange. The system keeps the last several turns of conversation in memory, so the LLM always has context for follow-up questions — just like a real conversation.

---

## CLOSING  *(show the chat feed with full conversation)*

So to recap what you just saw:

- Live speech-to-text with partial transcripts as you speak
- LLM inference firing before you even stop talking
- Text-to-speech with voice and text perfectly in sync
- Barge-in interruption that cancels mid-response instantly
- Web search grounded answers with natural citations
- Full conversation memory across turns

The entire stack is open source, runs offline, and costs nothing per query.

STT is faster-whisper. The LLM is qwen3 through Ollama. TTS is Kokoro 82M on Apple Silicon. The backend is FastAPI, the frontend is Next.js.

The hardest part of building this wasn't the models — it was the orchestration. Knowing exactly when to fire the LLM, when to start TTS, when to cancel, and how to keep voice and text in sync. That's what makes a voice agent feel human.

Thanks for watching — link to the repo is in the description.

---

---

## LinkedIn Post

---

I built a fully local real-time voice AI agent — no cloud, no API keys, nothing leaving your machine.

It's called **NeuroTalk**.

Here's what happens when you speak to it:

→ Your voice streams live to a local Whisper model which transcribes word by word as you talk — not after you stop  
→ A local LLM starts composing a reply before you finish your sentence  
→ TTS synthesis starts on the first sentence while the LLM is still generating the rest  
→ Text appears in the UI character by character, in perfect sync with the spoken voice  
→ Speak over the AI at any point — it stops immediately and listens again  
→ Ask about current events — it fires a web search in parallel and cites sources naturally in speech  

The stack is fully open source:

- **STT** → faster-whisper (Whisper small, int8)
- **LLM** → qwen3:4b via Ollama
- **TTS** → Kokoro 82M via MLX on Apple Silicon
- **Backend** → FastAPI + WebSocket streaming
- **Frontend** → Next.js

The models were the easy part. The hard part was the orchestration — debounce timing, partial transcript handling, parallel TTS synthesis, barge-in interruption, and keeping voice and text perfectly in sync without any flicker or race conditions.

If you're building voice agents, the real work lives in the seams between the models.

Demo video 👆 · Full source on GitHub in the comments.

#VoiceAI #LocalAI #OpenSource #LLM #SpeechAI #FastAPI #NextJS #BuildInPublic #AIEngineering #Ollama #Whisper
