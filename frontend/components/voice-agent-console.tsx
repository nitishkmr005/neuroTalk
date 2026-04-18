"use client";

import { startTransition, useCallback, useEffect, useRef, useState, type CSSProperties } from "react";

type Mode = "listening" | "thinking" | "responding";

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
  type: "ready" | "partial" | "final" | "error" | "llm_start" | "llm_partial" | "llm_final" | "llm_error";
  request_id?: string;
  text?: string;
  message?: string;
  timings_ms?: Metrics;
  debug?: DebugInfo;
  llm_ms?: number;
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
    eyebrow: "Live conversation in progress",
    headline: "Listening to the conversation as it happens.",
    summary:
      "The assistant is capturing the call in real time so associates can follow the conversation without losing context.",
    accent: "active-listening",
  },
  thinking: {
    eyebrow: "Transcript updating",
    headline: "Refreshing the live conversation view.",
    summary:
      "The transcript is being updated continuously to help associates track customer intent, details, and phrasing.",
    accent: "deep-reasoning",
  },
  responding: {
    eyebrow: "Conversation captured",
    headline: "Review the latest conversation transcript.",
    summary:
      "When the session stops, the assistant finalizes the latest transcript so associates can review the call clearly.",
    accent: "voice-delivery",
  },
};

const orchestrationSteps = [
  { label: "Microphone capture", detail: "PCM audio streamed in real-time over WebSocket to the backend.", status: "online" },
  { label: "Speech-to-Text (STT)", detail: "faster-whisper transcribes audio incrementally with VAD filtering.", status: "online" },
  { label: "LLM reasoning", detail: "Ollama (llama3.2) responds to the transcript as speech is detected.", status: "online" },
  { label: "Text-to-Speech (TTS)", detail: "Voice synthesis — coming soon to complete the full voice loop.", status: "pending" },
];

const waveformHeights = [28, 46, 32, 64, 24, 58, 38, 72, 44, 30, 66, 35, 54, 26, 60, 40];
const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
const websocketUrl = `${backendUrl.replace(/^http/, "ws")}/ws/transcribe`;
const initialWaveLevels = waveformHeights.map(() => 0.18);

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

export function VoiceAgentConsole() {
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
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const activeUserIdRef = useRef<string | null>(null);
  const activeAssistantIdRef = useRef<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  const websocketRef = useRef<WebSocket | null>(null);
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
  };

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
      setError(null);
      setMetrics(null);
      setDebugInfo(null);
      setTranscript("");
      setLlmLatencyMs(null);
      activeUserIdRef.current = null;
      activeAssistantIdRef.current = null;
      setIsConnecting(true);
      setIsFinalizing(false);
      isFinalizingRef.current = false;
      receivedFinalRef.current = false;
      normalCloseRef.current = false;

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;

      const socket = new WebSocket(websocketUrl);
      socket.binaryType = "arraybuffer";
      websocketRef.current = socket;
      sessionStartedAtRef.current = performance.now();
      streamReadyRef.current = false;

      socket.onopen = async () => {
        const audioContext = new AudioContext();
        audioContextRef.current = audioContext;

        const sourceNode = audioContext.createMediaStreamSource(stream);
        const processorNode = audioContext.createScriptProcessor(4096, 1, 1);
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

        // ── helpers ──────────────────────────────────────────────────────────
        const updateMsg = (id: string, patch: Partial<ChatMessage>) => {
          setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
        };

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
          const uid = activeUserIdRef.current;
          if (uid && payload.text?.trim()) updateMsg(uid, { text: payload.text });
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
          const existingAid = activeAssistantIdRef.current;
          if (existingAid) {
            // Same session (debounced → final handoff): reset the existing bubble rather than creating a second one
            updateMsg(existingAid, { text: "", isStreaming: true, isError: false });
          } else {
            const aid = crypto.randomUUID();
            activeAssistantIdRef.current = aid;
            setMessages((prev) => [...prev, { id: aid, role: "assistant", text: "", isStreaming: true, isError: false }]);
          }
          return;
        }

        if (payload.type === "llm_partial") {
          const aid = activeAssistantIdRef.current;
          if (aid) updateMsg(aid, { text: payload.text ?? "", isStreaming: true });
          return;
        }

        if (payload.type === "llm_final") {
          const aid = activeAssistantIdRef.current;
          if (aid) updateMsg(aid, { text: payload.text ?? "", isStreaming: false });
          if (payload.llm_ms != null) setLlmLatencyMs(payload.llm_ms);
          if (!isRecordingRef.current) {
            normalCloseRef.current = true;
            socket.close();
          }
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
    const socket = websocketRef.current;
    setIsRecording(false);
    isRecordingRef.current = false;
    setIsFinalizing(true);
    isFinalizingRef.current = true;
    normalCloseRef.current = true;
    stopAudioGraph();

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
      label: "Speech-to-Text",
      value: formatSeconds(metrics?.transcribe_ms),
      detail: "faster-whisper transcription time per buffer pass.",
    },
    {
      title: "LLM",
      label: "AI Response",
      value: formatSeconds(llmLatencyMs),
      detail: "End-to-end Ollama response time for the last completed reply.",
    },
    {
      title: "TTS",
      label: "Text-to-Speech",
      value: "--",
      detail: "Voice synthesis latency — coming soon.",
    },
    {
      title: "E2E",
      label: "Overall",
      value: formatSeconds(metrics?.total_ms != null && llmLatencyMs != null ? metrics.total_ms + llmLatencyMs : (metrics?.total_ms ?? null)),
      detail: "Combined STT + LLM pipeline time for the last session.",
    },
  ];

  return (
    <main className="console-shell">
      <section className="console-frame">
        <header className="topbar surface">
          <div>
            <p className="kicker">NeuroTalk / Associate Assist</p>
            <h1>Live Conversation Assist Console.</h1>
          </div>
          <div className="topbar-meta">
            <span className="status-pill is-live">Live call support</span>
          </div>
        </header>

        <section className="hero-grid">
          <article className="hero-panel surface">
            <div className="hero-copy">
              <span className={`mode-chip ${activeMode.accent}`}>{activeMode.eyebrow}</span>
              <h2>{activeMode.headline}</h2>
              <p>{activeMode.summary}</p>
            </div>

            <div className="hero-visual">
              <div
                className="orbital-core"
                aria-hidden="true"
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
              </div>

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
            </div>

            <div className="controls-row">
              <button
                type="button"
                className={isRecording ? "control-button is-recording" : "control-button"}
                disabled={controlDisabled}
                onClick={isRecording ? stopStreaming : () => void startStreaming()}
              >
                {controlLabel}
              </button>
              <div className="control-hints">
                <span>
                  {isRecording
                    ? "The conversation is being captured live"
                    : isFinalizing
                      ? "Finalizing the latest conversation transcript"
                      : "Ready to start a live conversation capture"}
                </span>
                <span className={error ? "is-error" : ""}>{error ?? "Transcript updates will appear during the conversation"}</span>
              </div>
            </div>

            <div className="mode-switcher">
              {(["listening", "thinking", "responding"] as Mode[]).map((item) => (
                <button
                  type="button"
                  key={item}
                  className={item === mode ? "mode-button is-selected" : "mode-button is-static"}
                  disabled
                >
                  {item}
                </button>
              ))}
            </div>
          </article>

          <aside className="telemetry-stack">
            <article className="surface telemetry-panel">
              <div className="section-heading">
                <p className="kicker">Audio Health</p>
                <span className="mini-dot" />
              </div>
              <div className="metric-grid">
                <div>
                  <span>Processing time</span>
                  <strong>{formatSeconds(metrics?.total_ms)}</strong>
                </div>
                <div>
                  <span>Model status</span>
                  <strong>{formatSeconds(metrics?.model_load_ms, { cachedWhenZero: true })}</strong>
                </div>
                <div>
                  <span>Transcript refresh</span>
                  <strong>{formatSeconds(metrics?.transcribe_ms)}</strong>
                </div>
                <div>
                  <span>Session duration</span>
                  <strong>{formatSeconds(metrics?.client_roundtrip_ms)}</strong>
                </div>
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
                <p className="kicker">Conversation</p>
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

        <section className="dashboard-grid">
          <article className="surface stack-panel">
            <div className="section-heading">
              <p className="kicker">Pipeline Steps</p>
              <span className="section-note">Active modules in the voice agent pipeline</span>
            </div>
            <div className="stack-list">
              {orchestrationSteps.map((step) => (
                <div className="stack-item" key={step.label}>
                  <div>
                    <h3>{step.label}</h3>
                    <p>{step.detail}</p>
                  </div>
                  <span className={`status-tag status-${step.status}`}>{step.status}</span>
                </div>
              ))}
            </div>
          </article>

          <article className="surface insights-panel">
            <div className="section-heading">
              <p className="kicker">Latency Breakdown</p>
              <span className="section-note">Pipeline timing for the last conversation turn</span>
            </div>
            <div className="card-grid">
              {latencyCards.map((card) => (
                <div className="info-card" key={card.title}>
                  <span className="info-card-label">{card.label}</span>
                  <strong>{card.value}</strong>
                  <p>{card.detail}</p>
                </div>
              ))}
            </div>
          </article>
        </section>
      </section>
    </main>
  );
}
