"use client";

import { Fragment, startTransition, useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { WebRTCTransport } from "./webrtc-transport";

type Mode = "listening" | "thinking" | "responding" | "speaking";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  isStreaming: boolean;
  isError: boolean;
};

type Metrics = {
  request_read_ms: number;
  file_write_ms: number;
  model_load_ms: number;
  transcribe_ms: number;
  total_ms: number;
  buffered_audio_ms: number;
  client_roundtrip_ms: number | null;
};

type DebugInfo = {
  request_id: string;
  filename: string;
  audio_bytes: number;
  detected_language: string | null;
  segments: number;
  model_size: string;
  device: string;
  compute_type: string;
  sample_rate: number | null;
  chunks_received: number | null;
};

type StreamMessage = {
  type: "ready" | "partial" | "final" | "error" | "llm_start" | "llm_partial" | "llm_final" | "llm_error" | "tts_start" | "tts_audio" | "tts_done" | "tts_interrupted";
  request_id?: string;
  text?: string;
  user_text?: string;
  message?: string;
  timings_ms?: Metrics;
  debug?: DebugInfo;
  llm_ms?: number;
  data?: string;
  tts_ms?: number;
  sentence_text?: string;
};

const modeConfig: Record<
  Mode,
  {
    eyebrow: string;
    headline: string;
    summary: string;
    accent: string;
  }
> = {
  listening: {
    eyebrow: "Listening",
    headline: "I'm listening. Speak naturally.",
    summary:
      "Ask me anything — I'll respond as soon as I understand. You can interrupt my reply at any point by speaking.",
    accent: "active-listening",
  },
  thinking: {
    eyebrow: "Processing your speech",
    headline: "Catching your words in real time.",
    summary:
      "Your speech is being transcribed as you talk. The AI will begin composing a reply once it understands your intent.",
    accent: "deep-reasoning",
  },
  responding: {
    eyebrow: "Composing a reply",
    headline: "Thinking of what to say.",
    summary:
      "The AI is crafting a response to what you said. It will speak the reply aloud in just a moment.",
    accent: "voice-delivery",
  },
  speaking: {
    eyebrow: "AI speaking",
    headline: "Hear the reply.",
    summary:
      "The AI is speaking its reply. Interrupt at any time by speaking — it will stop and listen immediately.",
    accent: "voice-delivery",
  },
};

const waveformHeights = [28, 46, 32, 64, 24, 58, 38, 72, 44, 30, 66, 35, 54, 26, 60, 40];
const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
const websocketUrl = `${backendUrl.replace(/^http/, "ws")}/ws/transcribe`;
const initialWaveLevels = waveformHeights.map(() => 0.18);
const BARGE_IN_THRESHOLD = 0.15;
const BARGE_IN_FRAMES = 1;

function float32ToInt16(input: Float32Array): Int16Array {
  const output = new Int16Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, input[index]));
    output[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output;
}

function getRmsAmplitude(input: Float32Array): number {
  let sum = 0;
  for (let index = 0; index < input.length; index += 1) {
    sum += input[index] * input[index];
  }

  return Math.sqrt(sum / input.length);
}

function formatSeconds(valueMs: number | null | undefined, options?: { cachedWhenZero?: boolean }): string {
  if (valueMs === null || valueMs === undefined) {
    return "--";
  }

  if (options?.cachedWhenZero && valueMs <= 0) {
    return "cached";
  }

  return `${(valueMs / 1000).toFixed(valueMs >= 1000 ? 2 : 3)} s`;
}

type TransportType = "websocket" | "webrtc";

export function VoiceAgentConsole() {
  const [isDark, setIsDark] = useState(true);
  const [transportType, setTransportType] = useState<TransportType>("webrtc");
  const [mode, setMode] = useState<Mode>("listening");
  const [isRecording, setIsRecording] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isFinalizing, setIsFinalizing] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const [amplitude, setAmplitude] = useState(0.08);
  const [waveLevels, setWaveLevels] = useState(initialWaveLevels);
  const [copied, setCopied] = useState(false);
  const [llmLatencyMs, setLlmLatencyMs] = useState<number | null>(null);
  const [ttsLatencyMs, setTtsLatencyMs] = useState<number | null>(null);
  const ttsSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const ttsAudioReceivedRef = useRef(false);
  const interruptSentRef = useRef(false);
  const bargeinFrameCountRef = useRef(0);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const activeUserIdRef = useRef<string | null>(null);
  const activeAssistantIdRef = useRef<string | null>(null);
  const pendingAssistantTextRef = useRef<string>("");
  const revealRafRef = useRef<number | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  // Audio queue: sentences arrive one at a time; we play them sequentially.
  type TtsChunk = { buffer: AudioBuffer; text: string };
  const ttsQueueRef = useRef<TtsChunk[]>([]);
  const isTtsPlayingRef = useRef(false);
  const ttsAllChunksReceivedRef = useRef(false);
  // Text revealed so far in the current assistant turn (accumulates across chunks).
  const revealedTextRef = useRef("");
  // Counts tts_audio chunks whose decodeAudioData hasn't fired yet — prevents
  // premature finalisation when tts_done arrives before all decodes complete.
  const pendingDecodesRef = useRef(0);

  const websocketRef = useRef<WebSocket | null>(null);
  const webrtcRef = useRef<WebRTCTransport | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const sessionStartedAtRef = useRef<number | null>(null);
  const streamReadyRef = useRef(false);
  const isFinalizingRef = useRef(false);
  const receivedFinalRef = useRef(false);
  const normalCloseRef = useRef(false);
  const isRecordingRef = useRef(false);
  const errorRef = useRef<string | null>(null);
  const amplitudeRef = useRef(0.08);
  const waveLevelsRef = useRef(initialWaveLevels);

  useEffect(() => {
    const saved = localStorage.getItem("nt-theme");
    const dark = saved !== "light";
    setIsDark(dark);
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  }, []);

  const toggleTheme = () => {
    const next = !isDark;
    setIsDark(next);
    const value = next ? "dark" : "light";
    localStorage.setItem("nt-theme", value);
    document.documentElement.setAttribute("data-theme", value);
  };

  useEffect(() => {
    errorRef.current = error;
  }, [error]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const stopAudioGraph = () => {
    processorNodeRef.current?.disconnect();
    sourceNodeRef.current?.disconnect();
    gainNodeRef.current?.disconnect();
    processorNodeRef.current = null;
    sourceNodeRef.current = null;
    gainNodeRef.current = null;
    ttsSourceRef.current?.stop();
    ttsSourceRef.current = null;
    if (revealRafRef.current !== null) {
      cancelAnimationFrame(revealRafRef.current);
      revealRafRef.current = null;
    }
    amplitudeRef.current = 0.08;
    setAmplitude(0.08);
    waveLevelsRef.current = initialWaveLevels;
    setWaveLevels(initialWaveLevels);

    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    mediaStreamRef.current = null;

    if (audioContextRef.current) {
      void audioContextRef.current.close();
      audioContextRef.current = null;
    }

    webrtcRef.current?.close();
    webrtcRef.current = null;

    // Clear audio queue
    ttsQueueRef.current = [];
    isTtsPlayingRef.current = false;
    ttsAllChunksReceivedRef.current = false;
    revealedTextRef.current = "";
    pendingDecodesRef.current = 0;
  };

  // Hoisted helper so playNextTtsChunk and message handlers can both call it.
  const updateMsg = useCallback((id: string, patch: Partial<ChatMessage>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  }, []);

  const clearTtsQueue = () => {
    ttsSourceRef.current?.stop();
    ttsSourceRef.current = null;
    ttsQueueRef.current = [];
    isTtsPlayingRef.current = false;
    ttsAllChunksReceivedRef.current = false;
    revealedTextRef.current = "";
    pendingDecodesRef.current = 0;
    if (revealRafRef.current !== null) {
      cancelAnimationFrame(revealRafRef.current);
      revealRafRef.current = null;
    }
  };

  const playNextTtsChunk = useCallback(() => {
    if (ttsQueueRef.current.length === 0) {
      isTtsPlayingRef.current = false;
      // Only finalise once all decodeAudioData callbacks have fired — prevents
      // premature wrap-up when tts_done arrives before the last decode completes.
      if (ttsAllChunksReceivedRef.current && pendingDecodesRef.current === 0) {
        const aid = activeAssistantIdRef.current;
        const fullText = pendingAssistantTextRef.current;
        if (aid && fullText) updateMsg(aid, { text: fullText, isStreaming: false });
        activeAssistantIdRef.current = null;
        pendingAssistantTextRef.current = "";
        revealedTextRef.current = "";
        // WebRTC is a persistent session — add a fresh Listening bubble for the next turn.
        if (webrtcRef.current && !activeUserIdRef.current) {
          receivedFinalRef.current = false;
          const freshUid = crypto.randomUUID();
          activeUserIdRef.current = freshUid;
          setMessages((prev) => [
            ...prev,
            { id: freshUid, role: "user", text: "Listening…", isStreaming: true, isError: false },
          ]);
        }
        startTransition(() => { setMode("listening"); });
      }
      return;
    }

    const chunk = ttsQueueRef.current.shift()!;
    const audioCtx = audioContextRef.current ?? new AudioContext();
    if (!audioContextRef.current) audioContextRef.current = audioCtx;

    const source = audioCtx.createBufferSource();
    source.buffer = chunk.buffer;
    source.connect(audioCtx.destination);
    ttsSourceRef.current = source;
    interruptSentRef.current = false;
    bargeinFrameCountRef.current = 0;

    const aid = activeAssistantIdRef.current;
    const chunkText = chunk.text;
    const baseText = revealedTextRef.current;
    const durationMs = Math.max(200, chunk.buffer.duration * 1000);
    const startTime = performance.now();

    // Reveal just this sentence's text during its playback window.
    const tick = () => {
      if (!aid || ttsSourceRef.current !== source) { revealRafRef.current = null; return; }
      const elapsed = performance.now() - startTime;
      const progress = Math.min(1, elapsed / durationMs);
      const partial = chunkText.slice(0, Math.floor(progress * chunkText.length));
      const visible = baseText ? `${baseText} ${partial}` : partial;
      updateMsg(aid, { text: visible.trim(), isStreaming: true });
      if (progress < 1) {
        revealRafRef.current = requestAnimationFrame(tick);
      } else {
        revealRafRef.current = null;
      }
    };
    if (aid && chunkText) revealRafRef.current = requestAnimationFrame(tick);

    source.onended = () => {
      ttsSourceRef.current = null;
      if (revealRafRef.current !== null) { cancelAnimationFrame(revealRafRef.current); revealRafRef.current = null; }
      // Accumulate this sentence into the revealed text baseline.
      const newBase = baseText ? `${baseText} ${chunkText}` : chunkText;
      revealedTextRef.current = newBase.trim();
      if (aid) updateMsg(aid, { text: revealedTextRef.current, isStreaming: true });
      playNextTtsChunk();
    };

    isTtsPlayingRef.current = true;
    void audioCtx.resume().then(() => { source.start(); });
  }, [updateMsg]);

  useEffect(() => {
    return () => {
      stopAudioGraph();
      websocketRef.current?.close();
      websocketRef.current = null;
    };
  }, []);

  const applyStreamPayload = (payload: StreamMessage) => {
    if (payload.text !== undefined) {
      setTranscript(payload.text || "No speech detected yet.");
    }

    if (payload.timings_ms) {
      const sessionRoundtripMs =
        sessionStartedAtRef.current === null ? null : Number((performance.now() - sessionStartedAtRef.current).toFixed(2));
      setMetrics({
        ...payload.timings_ms,
        client_roundtrip_ms: sessionRoundtripMs,
      });
    }

    if (payload.debug) {
      setDebugInfo(payload.debug);
    }
  };

  const startStreaming = async () => {
    try {
      // Stop any audio still playing from a previous session before starting a new one.
      stopAudioGraph();
      setError(null);
      setMetrics(null);
      setDebugInfo(null);
      setTranscript("");
      setLlmLatencyMs(null);
      setTtsLatencyMs(null);
      activeUserIdRef.current = null;
      activeAssistantIdRef.current = null;
      setIsConnecting(true);
      setIsFinalizing(false);
      isFinalizingRef.current = false;
      receivedFinalRef.current = false;
      normalCloseRef.current = false;
      interruptSentRef.current = false;
      bargeinFrameCountRef.current = 0;

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;

      // ── WebRTC path ────────────────────────────────────────────────────────
      if (transportType === "webrtc") {
        const transport = new WebRTCTransport(backendUrl);
        webrtcRef.current = transport;

        // Shared message handler for WebRTC (same protocol as WebSocket).
        // Key differences: no socket.close() on tts_done — the session is
        // persistent; instead we reset to "listening" for the next turn.
        transport.onMessage = (payload) => {
          const updateMsg = (id: string, patch: Partial<ChatMessage>) => {
            setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
          };

          if (payload.type === "ready") {
            streamReadyRef.current = true;
            const uid = crypto.randomUUID();
            activeUserIdRef.current = uid;
            activeAssistantIdRef.current = null;
            setMessages((prev) => [
              ...prev,
              { id: uid, role: "user", text: "Listening…", isStreaming: true, isError: false },
            ]);
            if (payload.request_id) {
              setDebugInfo((current) => ({
                request_id: String(payload.request_id ?? current?.request_id ?? "--"),
                filename: current?.filename ?? "stream.wav",
                audio_bytes: current?.audio_bytes ?? 0,
                detected_language: current?.detected_language ?? null,
                segments: current?.segments ?? 0,
                model_size: current?.model_size ?? "--",
                device: current?.device ?? "--",
                compute_type: current?.compute_type ?? "--",
                sample_rate: current?.sample_rate ?? null,
                chunks_received: current?.chunks_received ?? null,
              }));
            }
            return;
          }

          if (payload.type === "partial") {
            const text = typeof payload.text === "string" ? payload.text : undefined;
            if (text !== undefined) setTranscript(text || "No speech detected yet.");
            const timings = payload.timings_ms as Metrics | undefined;
            if (timings) {
              setMetrics({
                ...timings,
                client_roundtrip_ms:
                  sessionStartedAtRef.current === null
                    ? null
                    : Number((performance.now() - sessionStartedAtRef.current).toFixed(2)),
              });
            }
            if (payload.debug) setDebugInfo(payload.debug as DebugInfo);
            let uid = activeUserIdRef.current;
            if (!uid && text?.trim()) {
              uid = crypto.randomUUID();
              activeUserIdRef.current = uid;
              setMessages((prev) => [
                ...prev,
                { id: uid!, role: "user", text: text ?? "", isStreaming: true, isError: false },
              ]);
            } else if (uid && text?.trim()) {
              updateMsg(uid, { text });
            }
            startTransition(() => { setMode("thinking"); });
            return;
          }

          if (payload.type === "final") {
            receivedFinalRef.current = true;
            const text = typeof payload.text === "string" ? payload.text : undefined;
            if (text !== undefined) setTranscript(text || "No speech detected yet.");
            setIsFinalizing(false);
            isFinalizingRef.current = false;
            const uid = activeUserIdRef.current;
            const finalText = text?.trim() ?? "";
            if (uid) updateMsg(uid, { text: finalText || "…", isStreaming: false });
            startTransition(() => { setMode("responding"); });
            return;
          }

          if (payload.type === "llm_start") {
            const uid = activeUserIdRef.current;
            const userText = typeof payload.user_text === "string" ? payload.user_text.trim() : "";
            if (uid) {
              if (userText) {
                updateMsg(uid, { text: userText, isStreaming: false });
              } else {
                setMessages((prev) => prev.filter((m) => m.id !== uid));
              }
            }
            activeUserIdRef.current = null;
            // Finalize any lingering assistant bubble from a previous interrupted turn
            // instead of resetting it — that way the partial response stays visible.
            const prevAid = activeAssistantIdRef.current;
            if (prevAid) {
              const prevText = revealedTextRef.current || pendingAssistantTextRef.current;
              if (prevText) updateMsg(prevAid, { text: prevText, isStreaming: false });
            }
            activeAssistantIdRef.current = null;
            pendingAssistantTextRef.current = "";
            ttsAudioReceivedRef.current = false;
            if (revealRafRef.current !== null) {
              cancelAnimationFrame(revealRafRef.current);
              revealRafRef.current = null;
            }
            // Always open a fresh assistant bubble for this turn.
            const aid = crypto.randomUUID();
            activeAssistantIdRef.current = aid;
            setMessages((prev) => [
              ...prev,
              { id: aid, role: "assistant", text: "", isStreaming: true, isError: false },
            ]);
            return;
          }

          if (payload.type === "llm_partial") {
            pendingAssistantTextRef.current = typeof payload.text === "string" ? payload.text : "";
            startTransition(() => { setMode("responding"); });
            return;
          }

          if (payload.type === "llm_final") {
            pendingAssistantTextRef.current = typeof payload.text === "string" ? payload.text : "";
            if (payload.llm_ms != null) setLlmLatencyMs(Number(payload.llm_ms));
            startTransition(() => { setMode("listening"); });
            return;
          }

          if (payload.type === "llm_error") {
            const aid = activeAssistantIdRef.current;
            if (aid)
              updateMsg(aid, {
                text: "AI unavailable — make sure Ollama is running.",
                isStreaming: false,
                isError: true,
              });
            return;
          }

          if (payload.type === "tts_start") {
            // Stop any currently playing audio before starting the new sequence.
            // Prevents parallel playback when a second response starts before the
            // first finishes (e.g. after a short-pause false LLM trigger).
            clearTtsQueue();
            pendingDecodesRef.current = 0;
            ttsAudioReceivedRef.current = false;
            startTransition(() => { setMode("speaking"); });
            return;
          }

          if (payload.type === "tts_audio") {
            if (payload.tts_ms != null) setTtsLatencyMs(Number(payload.tts_ms));
            const sentenceText = typeof payload.sentence_text === "string" ? payload.sentence_text : (pendingAssistantTextRef.current ?? "");
            const data = typeof payload.data === "string" ? payload.data : "";
            const binaryString = atob(data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
            const audioCtx = audioContextRef.current ?? new AudioContext();
            if (!audioContextRef.current) audioContextRef.current = audioCtx;
            ttsAudioReceivedRef.current = true;
            pendingDecodesRef.current++;
            void audioCtx.decodeAudioData(bytes.buffer.slice(0), (buffer) => {
              pendingDecodesRef.current--;
              ttsQueueRef.current.push({ buffer, text: sentenceText });
              if (!isTtsPlayingRef.current) playNextTtsChunk();
            });
            return;
          }

          if (payload.type === "tts_done") {
            ttsAllChunksReceivedRef.current = true;
            if (!ttsAudioReceivedRef.current) {
              // TTS produced nothing (error path) — show LLM text directly.
              const aid = activeAssistantIdRef.current;
              if (aid && pendingAssistantTextRef.current) {
                updateMsg(aid, { text: pendingAssistantTextRef.current, isStreaming: false });
              }
              activeAssistantIdRef.current = null;
              pendingAssistantTextRef.current = "";
              revealedTextRef.current = "";
              receivedFinalRef.current = false;
              const freshUid = crypto.randomUUID();
              activeUserIdRef.current = freshUid;
              setMessages((prev) => [
                ...prev,
                { id: freshUid, role: "user", text: "Listening…", isStreaming: true, isError: false },
              ]);
              startTransition(() => { setMode("listening"); });
            } else if (!isTtsPlayingRef.current && ttsQueueRef.current.length === 0 && pendingDecodesRef.current === 0) {
              // All audio already played and decoded before tts_done arrived.
              const aid = activeAssistantIdRef.current;
              if (aid) updateMsg(aid, { text: pendingAssistantTextRef.current, isStreaming: false });
              activeAssistantIdRef.current = null;
              pendingAssistantTextRef.current = "";
              revealedTextRef.current = "";
              receivedFinalRef.current = false;
              const freshUid = crypto.randomUUID();
              activeUserIdRef.current = freshUid;
              setMessages((prev) => [
                ...prev,
                { id: freshUid, role: "user", text: "Listening…", isStreaming: true, isError: false },
              ]);
              startTransition(() => { setMode("listening"); });
            }
            // Otherwise playNextTtsChunk's finalise path handles it once the queue drains.
            return;
          }

          if (payload.type === "tts_interrupted") {
            clearTtsQueue();
            startTransition(() => { setMode("listening"); });
            return;
          }

          if (payload.type === "error") {
            const message = typeof payload.message === "string"
              ? payload.message
              : "Streaming transcription failed.";
            errorRef.current = message;
            setError(message);
            setIsFinalizing(false);
            isFinalizingRef.current = false;
            startTransition(() => { setMode("listening"); });
          }
        };

        transport.onClose = () => {
          streamReadyRef.current = false;
          webrtcRef.current = null;
          setIsConnecting(false);
          setIsRecording(false);
          isRecordingRef.current = false;
          if (!errorRef.current) stopAudioGraph();
        };

        try {
          await transport.connect(stream);
        } catch (connectErr) {
          const msg =
            connectErr instanceof Error ? connectErr.message : "WebRTC connection failed.";
          errorRef.current = msg;
          setError(msg);
          setIsConnecting(false);
          setIsRecording(false);
          setIsFinalizing(false);
          stopAudioGraph();
          return;
        }

        // ScriptProcessor: amplitude visualisation + client-side barge-in detection.
        // Audio is NOT sent as binary — it travels via the RTP track added in connect().
        const rtcAudioCtx = new AudioContext();
        audioContextRef.current = rtcAudioCtx;
        const rtcSource = rtcAudioCtx.createMediaStreamSource(stream);
        const rtcProcessor = rtcAudioCtx.createScriptProcessor(2048, 1, 1);
        const rtcGain = rtcAudioCtx.createGain();
        rtcGain.gain.value = 0;
        rtcSource.connect(rtcProcessor);
        rtcProcessor.connect(rtcGain);
        rtcGain.connect(rtcAudioCtx.destination);
        sourceNodeRef.current = rtcSource;
        processorNodeRef.current = rtcProcessor;
        gainNodeRef.current = rtcGain;

        rtcProcessor.onaudioprocess = (event) => {
          const channelData = event.inputBuffer.getChannelData(0);
          const rms = getRmsAmplitude(channelData);
          const nextAmplitude = Math.min(1, Math.max(0.04, rms * 11.5));
          const smoothed = amplitudeRef.current * 0.58 + nextAmplitude * 0.42;
          amplitudeRef.current = smoothed;
          setAmplitude(smoothed);
          const nextBars = [...waveLevelsRef.current.slice(1), smoothed];
          waveLevelsRef.current = nextBars;
          setWaveLevels(nextBars);

          // Client-side barge-in (complements server-side VAD in session.py)
          if ((ttsSourceRef.current || ttsQueueRef.current.length > 0) && !interruptSentRef.current) {
            if (rms > BARGE_IN_THRESHOLD) {
              bargeinFrameCountRef.current += 1;
              if (bargeinFrameCountRef.current >= BARGE_IN_FRAMES) {
                interruptSentRef.current = true;
                clearTtsQueue();
                transport.send({ type: "interrupt" });
                startTransition(() => { setMode("listening"); });
              }
            } else {
              bargeinFrameCountRef.current = 0;
            }
          }
          // No binary send here — audio travels via the RTP track.
        };

        transport.send({ type: "start", sample_rate: rtcAudioCtx.sampleRate });
        setIsConnecting(false);
        setIsRecording(true);
        isRecordingRef.current = true;
        startTransition(() => { setMode("listening"); });
        return;
      }
      // ── End WebRTC path ────────────────────────────────────────────────────

      const socket = new WebSocket(websocketUrl);
      socket.binaryType = "arraybuffer";
      websocketRef.current = socket;
      sessionStartedAtRef.current = performance.now();
      streamReadyRef.current = false;

      socket.onopen = async () => {
        const audioContext = new AudioContext();
        audioContextRef.current = audioContext;

        const sourceNode = audioContext.createMediaStreamSource(stream);
        const processorNode = audioContext.createScriptProcessor(2048, 1, 1);
        const gainNode = audioContext.createGain();
        gainNode.gain.value = 0;

        sourceNode.connect(processorNode);
        processorNode.connect(gainNode);
        gainNode.connect(audioContext.destination);

        sourceNodeRef.current = sourceNode;
        processorNodeRef.current = processorNode;
        gainNodeRef.current = gainNode;

        processorNode.onaudioprocess = (event) => {
          if (!streamReadyRef.current || socket.readyState !== WebSocket.OPEN) {
            return;
          }

          const channelData = event.inputBuffer.getChannelData(0);
          const rms = getRmsAmplitude(channelData);
          const nextAmplitude = Math.min(1, Math.max(0.04, rms * 11.5));
          const smoothedAmplitude = amplitudeRef.current * 0.58 + nextAmplitude * 0.42;
          amplitudeRef.current = smoothedAmplitude;
          setAmplitude(smoothedAmplitude);

          const nextBars = [...waveLevelsRef.current.slice(1), smoothedAmplitude];
          waveLevelsRef.current = nextBars;
          setWaveLevels(nextBars);

          if ((ttsSourceRef.current || ttsQueueRef.current.length > 0) && !interruptSentRef.current) {
            if (rms > BARGE_IN_THRESHOLD) {
              bargeinFrameCountRef.current += 1;
              if (bargeinFrameCountRef.current >= BARGE_IN_FRAMES) {
                interruptSentRef.current = true;
                clearTtsQueue();
                if (socket.readyState === WebSocket.OPEN) {
                  socket.send(JSON.stringify({ type: "interrupt" }));
                }
                startTransition(() => { setMode("listening"); });
              }
            } else {
              bargeinFrameCountRef.current = 0;
            }
          }

          const pcm16 = float32ToInt16(channelData);
          socket.send(pcm16.buffer);
        };

        socket.send(
          JSON.stringify({
            type: "start",
            sample_rate: audioContext.sampleRate,
          }),
        );

        setIsConnecting(false);
        setIsRecording(true);
        isRecordingRef.current = true;
        startTransition(() => {
          setMode("listening");
        });
      };

      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data) as StreamMessage;

        if (payload.type === "ready") {
          streamReadyRef.current = true;
          // Create the user message bubble for this recording session
          const uid = crypto.randomUUID();
          activeUserIdRef.current = uid;
          activeAssistantIdRef.current = null;
          setMessages((prev) => [...prev, { id: uid, role: "user", text: "Listening…", isStreaming: true, isError: false }]);
          if (payload.request_id) {
            setDebugInfo((current) => ({
              request_id: payload.request_id ?? current?.request_id ?? "--",
              filename: current?.filename ?? "stream.wav",
              audio_bytes: current?.audio_bytes ?? 0,
              detected_language: current?.detected_language ?? null,
              segments: current?.segments ?? 0,
              model_size: current?.model_size ?? "--",
              device: current?.device ?? "--",
              compute_type: current?.compute_type ?? "--",
              sample_rate: current?.sample_rate ?? null,
              chunks_received: current?.chunks_received ?? null,
            }));
          }
          return;
        }

        if (payload.type === "partial") {
          applyStreamPayload(payload);
          let uid = activeUserIdRef.current;
          if (!uid && payload.text?.trim()) {
            uid = crypto.randomUUID();
            activeUserIdRef.current = uid;
            setMessages((prev) => [...prev, { id: uid!, role: "user", text: payload.text ?? "", isStreaming: true, isError: false }]);
          } else if (uid && payload.text?.trim()) {
            updateMsg(uid, { text: payload.text });
          }
          startTransition(() => { setMode("thinking"); });
          return;
        }

        if (payload.type === "final") {
          receivedFinalRef.current = true;
          applyStreamPayload(payload);
          setIsFinalizing(false);
          isFinalizingRef.current = false;
          const uid = activeUserIdRef.current;
          const finalText = payload.text?.trim() ?? "";
          if (uid) updateMsg(uid, { text: finalText || "…", isStreaming: false });
          startTransition(() => { setMode("responding"); });
          if (!finalText) {
            normalCloseRef.current = true;
            socket.close();
          }
          return;
        }

        if (payload.type === "llm_start") {
          const uid = activeUserIdRef.current;
          const userText = payload.user_text?.trim();
          if (uid) {
            if (userText) {
              updateMsg(uid, { text: userText, isStreaming: false });
            } else {
              setMessages((prev) => prev.filter((m) => m.id !== uid));
            }
          }
          activeUserIdRef.current = null;
          // Finalize any lingering assistant bubble rather than resetting it.
          const prevAid = activeAssistantIdRef.current;
          if (prevAid) {
            const prevText = revealedTextRef.current || pendingAssistantTextRef.current;
            if (prevText) updateMsg(prevAid, { text: prevText, isStreaming: false });
          }
          activeAssistantIdRef.current = null;
          pendingAssistantTextRef.current = "";
          ttsAudioReceivedRef.current = false;
          if (revealRafRef.current !== null) {
            cancelAnimationFrame(revealRafRef.current);
            revealRafRef.current = null;
          }
          const aid = crypto.randomUUID();
          activeAssistantIdRef.current = aid;
          setMessages((prev) => [...prev, { id: aid, role: "assistant", text: "", isStreaming: true, isError: false }]);
          return;
        }

        if (payload.type === "llm_partial") {
          // Buffer the text but DON'T render yet — we want the visible text to advance
          // in lockstep with TTS audio playback, not race ahead of it.
          pendingAssistantTextRef.current = payload.text ?? "";
          startTransition(() => { setMode("responding"); });
          return;
        }

        if (payload.type === "llm_final") {
          pendingAssistantTextRef.current = payload.text ?? "";
          if (payload.llm_ms != null) setLlmLatencyMs(payload.llm_ms);
          startTransition(() => { setMode(isRecordingRef.current ? "listening" : "responding"); });
          return;
        }

        if (payload.type === "llm_error") {
          const aid = activeAssistantIdRef.current;
          if (aid) updateMsg(aid, { text: "AI unavailable — make sure Ollama is running.", isStreaming: false, isError: true });
          if (!isRecordingRef.current) {
            normalCloseRef.current = true;
            socket.close();
          }
          return;
        }

        if (payload.type === "tts_start") {
          clearTtsQueue();
          pendingDecodesRef.current = 0;
          ttsAudioReceivedRef.current = false;
          startTransition(() => { setMode("speaking"); });
          return;
        }

        if (payload.type === "tts_audio") {
          if (payload.tts_ms != null) setTtsLatencyMs(payload.tts_ms);
          const sentenceText = payload.sentence_text ?? pendingAssistantTextRef.current;
          const binaryString = atob(payload.data ?? "");
          const bytes = new Uint8Array(binaryString.length);
          for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
          const audioCtx = audioContextRef.current ?? new AudioContext();
          if (!audioContextRef.current) audioContextRef.current = audioCtx;
          ttsAudioReceivedRef.current = true;
          pendingDecodesRef.current++;
          void audioCtx.decodeAudioData(bytes.buffer.slice(0), (buffer) => {
            pendingDecodesRef.current--;
            ttsQueueRef.current.push({ buffer, text: sentenceText });
            if (!isTtsPlayingRef.current) playNextTtsChunk();
          });
          return;
        }

        if (payload.type === "tts_done") {
          ttsAllChunksReceivedRef.current = true;
          if (!ttsAudioReceivedRef.current) {
            const aid = activeAssistantIdRef.current;
            if (aid && pendingAssistantTextRef.current) {
              updateMsg(aid, { text: pendingAssistantTextRef.current, isStreaming: false });
            }
            activeAssistantIdRef.current = null;
            pendingAssistantTextRef.current = "";
            revealedTextRef.current = "";
            startTransition(() => { setMode(isRecordingRef.current ? "listening" : "responding"); });
            if (!isRecordingRef.current && receivedFinalRef.current) {
              normalCloseRef.current = true;
              socket.close();
            }
          } else if (!isTtsPlayingRef.current && ttsQueueRef.current.length === 0 && pendingDecodesRef.current === 0) {
            const aid = activeAssistantIdRef.current;
            if (aid) updateMsg(aid, { text: pendingAssistantTextRef.current, isStreaming: false });
            activeAssistantIdRef.current = null;
            pendingAssistantTextRef.current = "";
            revealedTextRef.current = "";
            startTransition(() => { setMode(isRecordingRef.current ? "listening" : "responding"); });
            if (!isRecordingRef.current && receivedFinalRef.current) {
              normalCloseRef.current = true;
              socket.close();
            }
          }
          return;
        }

        if (payload.type === "tts_interrupted") {
          clearTtsQueue();
          startTransition(() => { setMode("listening"); });
          return;
        }

        if (payload.type === "error") {
          const message = payload.message ?? "Streaming transcription failed.";
          errorRef.current = message;
          setError(message);
          setIsFinalizing(false);
          isFinalizingRef.current = false;
          startTransition(() => {
            setMode("listening");
          });
        }
      };

      socket.onerror = () => {
        const message = `Could not connect to backend stream at ${websocketUrl}. Run make dev and retry.`;
        errorRef.current = message;
        setError(message);
        setTranscript("Streaming connection failed before transcription could start.");
        setIsConnecting(false);
        setIsRecording(false);
        setIsFinalizing(false);
        stopAudioGraph();
      };

      socket.onclose = (event) => {
        streamReadyRef.current = false;
        websocketRef.current = null;
        setIsConnecting(false);
        setIsRecording(false);

        const wasExpectedClose =
          normalCloseRef.current || receivedFinalRef.current || isFinalizingRef.current || event.code === 1000;

        if (!wasExpectedClose && !errorRef.current) {
          const message = `Streaming connection closed unexpectedly at ${websocketUrl}.`;
          errorRef.current = message;
          setError(message);
        }

        stopAudioGraph();
      };
    } catch (caughtError) {
      const message = caughtError instanceof Error ? caughtError.message : "Microphone access failed.";
      setError(message);
      errorRef.current = message;
      setTranscript("Unable to start live transcription.");
      setIsConnecting(false);
      setIsRecording(false);
      setIsFinalizing(false);
      stopAudioGraph();
    }
  };

  const copyTranscript = useCallback(() => {
    const text = messages
      .map((m) => `${m.role === "user" ? "You" : "AI"}: ${m.text}`)
      .join("\n\n");
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [messages]);

  const stopStreaming = () => {
    setIsRecording(false);
    isRecordingRef.current = false;
    normalCloseRef.current = true;

    // WebRTC: closing the transport tears down the peer connection; the server
    // handles cleanup via connectionstatechange.  No "stop" message is sent
    // because the session is persistent and there is no single "current utterance"
    // to finalise — the server's silence debounce handles turn detection.
    const transport = webrtcRef.current;
    if (transport) {
      stopAudioGraph(); // closes transport + mic + audio graph
      setIsFinalizing(false);
      isFinalizingRef.current = false;
      startTransition(() => { setMode("listening"); });
      return;
    }

    // WebSocket: send "stop" and let the existing WS pipeline finalise.
    setIsFinalizing(true);
    isFinalizingRef.current = true;
    stopAudioGraph();
    const socket = websocketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "stop" }));
      startTransition(() => {
        setMode("thinking");
      });
      return;
    }

    setIsFinalizing(false);
    isFinalizingRef.current = false;
  };

  const activeMode = modeConfig[mode];
  const controlLabel = isRecording ? "Stop Streaming" : isConnecting ? "Connecting..." : "Start Live Transcription";
  const controlDisabled = isConnecting || isFinalizing;
  const orbScale = (1 + amplitude * 0.42).toFixed(3);
  const orbGlow = (0.45 + amplitude * 1.3).toFixed(3);
  const orbTilt = `${(amplitude * 18).toFixed(2)}deg`;
  const orbCoreScale = (1 + amplitude * 0.34).toFixed(3);
  const orbDriftX = `${(amplitude * 12).toFixed(2)}px`;
  const orbDriftY = `${(amplitude * -10).toFixed(2)}px`;
  const signalDescriptor =
    amplitude > 0.58 ? "High energy" : amplitude > 0.28 ? "Active voice" : isRecording ? "Low input" : "Idle";

  const latencyCards = [
    {
      title: "STT",
      label: "Transcription",
      value: formatSeconds(metrics?.transcribe_ms),
      detail: "faster-whisper transcription time per buffer pass.",
    },
    {
      title: "LLM",
      label: "Response Generation",
      value: formatSeconds(llmLatencyMs),
      detail: "End-to-end Ollama response time for the last completed reply.",
    },
    {
      title: "TTS",
      label: "Voice Synthesis",
      value: formatSeconds(ttsLatencyMs),
      detail: "Chatterbox Turbo synthesis time for the last AI reply.",
    },
    {
      title: "E2E",
      label: "Turn Latency",
      value: formatSeconds(
        metrics?.total_ms != null && llmLatencyMs != null && ttsLatencyMs != null
          ? metrics.total_ms + llmLatencyMs + ttsLatencyMs
          : metrics?.total_ms != null && llmLatencyMs != null
            ? metrics.total_ms + llmLatencyMs
            : metrics?.total_ms ?? null
      ),
      detail: "Combined STT + LLM + TTS pipeline time for the last session.",
    },
  ];

  return (
    <main className="console-shell">
      <section className="console-frame">
        <header className="topbar surface">
          <div>
            <p className="kicker">Voice Intelligence Platform</p>
            <h1>NeuroTalk</h1>
            <p className="topbar-tagline">Live transcription · AI reasoning · Expressive voice synthesis</p>
          </div>
          <div className="topbar-meta">
            <span className="status-pill is-live">Live call support</span>
            <div className="transport-toggle" aria-label="Select audio transport">
              <button
                type="button"
                className={`transport-btn${transportType === "webrtc" ? " is-active" : ""}`}
                disabled={isRecording || isConnecting}
                onClick={() => setTransportType("webrtc")}
                title="WebRTC — Opus/RTP over UDP with built-in echo cancellation and server-side VAD. Recommended."
              >
                WebRTC
              </button>
              <button
                type="button"
                className={`transport-btn${transportType === "websocket" ? " is-active" : ""}`}
                disabled={isRecording || isConnecting}
                onClick={() => setTransportType("websocket")}
                title="WebSocket — raw PCM over TCP. Simpler but no echo cancellation. Use if WebRTC fails."
              >
                WebSocket
              </button>
            </div>
            <button
              type="button"
              className="theme-toggle"
              onClick={toggleTheme}
              aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
            >
              {isDark ? (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="5"/>
                  <line x1="12" y1="1" x2="12" y2="3"/>
                  <line x1="12" y1="21" x2="12" y2="23"/>
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                  <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                  <line x1="1" y1="12" x2="3" y2="12"/>
                  <line x1="21" y1="12" x2="23" y2="12"/>
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                  <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                </svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                </svg>
              )}
            </button>
          </div>
        </header>

        <section className="hero-grid">
          <article className="hero-panel surface">
            <div className="hero-copy">
              <span className={`mode-chip ${activeMode.accent}`}>
                {(mode === "listening" || mode === "speaking") && (
                  <span className="mode-chip-dot" aria-hidden="true" />
                )}
                {activeMode.eyebrow}
              </span>
              <h2>{activeMode.headline}</h2>
              <p className="hero-summary">{activeMode.summary}</p>
            </div>

            <div className="hero-visual">
              <button
                type="button"
                className={[
                  "orbital-core",
                  isRecording ? "orbital-core--recording" : "",
                  isConnecting ? "orbital-core--connecting" : "",
                ].filter(Boolean).join(" ")}
                disabled={controlDisabled}
                onClick={isRecording ? stopStreaming : () => void startStreaming()}
                aria-label={controlLabel}
                style={
                  {
                    "--orb-scale": orbScale,
                    "--orb-glow": orbGlow,
                    "--orb-tilt": orbTilt,
                    "--orb-core-scale": orbCoreScale,
                    "--orb-drift-x": orbDriftX,
                    "--orb-drift-y": orbDriftY,
                  } as CSSProperties
                }
              >
                <div className="orb-ring orb-ring-1" />
                <div className="orb-ring orb-ring-2" />
                <div className="orb-center" />
                <div className="orb-scanline" />
              </button>

              <div className="wave-grid" aria-hidden="true">
                {waveformHeights.map((height, index) => (
                  <span
                    className="wave-bar"
                    key={`${height}-${index}`}
                    style={
                      {
                        "--bar-height": `${18 + waveLevels[index] * 70}px`,
                        "--bar-delay": `${index * 0.03}s`,
                        "--bar-opacity": (0.3 + waveLevels[index] * 0.7).toFixed(3),
                        "--bar-scale": (0.82 + waveLevels[index] * 0.48).toFixed(3),
                      } as CSSProperties
                    }
                  />
                ))}
              </div>

              <p className={`orb-tap-hint${isRecording ? " is-active" : isFinalizing ? " is-muted" : ""}`}>
                {error
                  ? <span className="is-error">{error}</span>
                  : isFinalizing
                    ? "Processing…"
                    : isRecording
                      ? "Tap to stop"
                      : "Tap to speak"}
              </p>
            </div>

            <div className="mode-switcher">
              {(["listening", "thinking", "responding"] as Mode[]).map((item, index) => (
                <Fragment key={item}>
                  {index > 0 && (
                    <span className={`mode-step-line${
                      ["listening", "thinking", "responding"].indexOf(mode) >= index ? " is-active" : ""
                    }`} />
                  )}
                  <button
                    type="button"
                    className={item === mode ? "mode-button is-selected" : "mode-button is-static"}
                    disabled
                  >
                    {item}
                  </button>
                </Fragment>
              ))}
            </div>
          </article>

          <aside className="telemetry-stack">
            <article className="surface telemetry-panel">
              <div className="section-heading">
                <p className="kicker">Live Signal Monitor</p>
                <span className="mini-dot" />
              </div>
              <div className="metric-grid">
                {latencyCards.map((card) => (
                  <div key={card.title}>
                    <span>{card.label}</span>
                    <strong>{card.value}</strong>
                  </div>
                ))}
              </div>
              <div className="telemetry-stage">
                <div className="telemetry-meter">
                  <div className="telemetry-meter-header">
                    <span>Input energy</span>
                    <strong>{signalDescriptor}</strong>
                  </div>
                  <div className="telemetry-meter-track">
                    <span
                      className="telemetry-meter-fill"
                      style={{ "--meter-fill": `${Math.max(8, amplitude * 100)}%` } as CSSProperties}
                    />
                  </div>
                </div>
              </div>
            </article>

            <article className="surface transcript-panel">
              <div className="section-heading">
                <p className="kicker">Live Conversation Feed</p>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span className="status-pill is-ghost">{error ? "Attention needed" : isRecording ? "Live" : messages.length ? "Done" : "Ready"}</span>
                  <button
                    type="button"
                    className={`copy-button${copied ? " is-copied" : ""}`}
                    onClick={copyTranscript}
                    disabled={messages.length === 0}
                    aria-label="Copy conversation"
                  >
                    {copied ? (
                      <>
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                        Copied
                      </>
                    ) : (
                      <>
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                        Copy
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="chat-thread">
                {messages.length === 0 ? (
                  <div className="chat-empty">
                    <p>Start a recording to begin the conversation.</p>
                  </div>
                ) : (
                  messages.map((msg) => (
                    <div key={msg.id} className={`chat-message chat-message--${msg.role}`}>
                      <span className={`chat-avatar chat-avatar--${msg.role}`}>
                        {msg.role === "user" ? "You" : "AI"}
                      </span>
                      <div
                        className={[
                          "chat-bubble",
                          `chat-bubble--${msg.role}`,
                          msg.isError ? "chat-bubble--error" : "",
                          msg.isStreaming && msg.role === "assistant" ? "is-streaming" : "",
                        ].filter(Boolean).join(" ")}
                      >
                        {msg.isError ? (
                          <p className="chat-text chat-text--error">{msg.text}</p>
                        ) : msg.text ? (
                          <p className="chat-text">{msg.text}</p>
                        ) : msg.isStreaming ? (
                          msg.role === "assistant" ? (
                            <div className="chat-typing-indicator" aria-label="AI is thinking">
                              <span /><span /><span />
                            </div>
                          ) : (
                            <p className="chat-text chat-text--placeholder">Listening…</p>
                          )
                        ) : (
                          <p className="chat-text chat-text--placeholder">…</p>
                        )}
                        {/* live-capture dot on user bubble only */}
                        {msg.isStreaming && msg.role === "user" && msg.text && (
                          <span className="chat-typing-dot" aria-hidden="true" />
                        )}
                      </div>
                    </div>
                  ))
                )}
                <div ref={chatEndRef} />
              </div>

              <div className="transcript-footer">
                <span className="transcript-meta">{error ?? `Reference ID: ${debugInfo?.request_id ?? "--"}`}</span>
                <span className="transcript-meta">Language: {debugInfo?.detected_language ?? "--"}</span>
              </div>
            </article>
          </aside>
        </section>

      </section>

      <footer className="console-footer">
        <div className="console-footer-inner">
          <span className="console-footer-brand">NeuroTalk</span>
          <span className="console-footer-sep" aria-hidden="true">·</span>
          <span className="console-footer-tagline">Local voice AI · STT · LLM · TTS</span>
          <span className="console-footer-sep" aria-hidden="true">·</span>
          <a
            className="console-footer-link"
            href="https://github.com/nitishkmr005/neuroTalk"
            target="_blank"
            rel="noopener noreferrer"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" style={{ display: "inline", verticalAlign: "middle", marginRight: 5 }}>
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
            </svg>
            GitHub
          </a>
        </div>
      </footer>
    </main>
  );
}
