"use client";

import { startTransition, useEffect, useRef, useState, type CSSProperties } from "react";

type Mode = "listening" | "thinking" | "responding";

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
  type: "ready" | "partial" | "final" | "error";
  request_id?: string;
  text?: string;
  message?: string;
  timings_ms?: Metrics;
  debug?: DebugInfo;
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
    eyebrow: "Microphone stream open",
    headline: "Listening and transcribing while you speak.",
    summary:
      "The browser streams raw PCM audio over WebSocket. Partial transcript updates appear during the live session instead of only after you stop.",
    accent: "active-listening",
  },
  thinking: {
    eyebrow: "Live transcription running",
    headline: "Refreshing the transcript from the growing audio buffer.",
    summary:
      "The backend rewrites the current PCM buffer to WAV, runs faster-whisper, and returns updated text plus latency metrics on a timed interval.",
    accent: "deep-reasoning",
  },
  responding: {
    eyebrow: "Final transcript returned",
    headline: "Inspect the completed text and timing breakdown.",
    summary:
      "When you stop the microphone, the backend performs one final pass and returns the latest transcript, request id, and timing values for debugging.",
    accent: "voice-delivery",
  },
};

const orchestrationSteps = [
  { label: "PCM capture", detail: "Web Audio collects mono microphone samples and converts them to 16-bit PCM.", status: "online" },
  { label: "WebSocket stream", detail: "Chunks are sent continuously to the backend while the microphone is active.", status: "online" },
  { label: "Incremental STT", detail: "faster-whisper re-transcribes the growing buffer on a fixed cadence.", status: "stable" },
  { label: "Debug telemetry", detail: "Every partial result includes latency and request details for fast diagnosis.", status: "online" },
];

const runtimeNotes = [
  { name: "Run both apps", tone: "Use `make dev` to start the backend and frontend together in one command." },
  { name: "Small / CPU", tone: "Good initial model for simple local development and easier iteration on the pipeline." },
  { name: "Buffered updates", tone: "This is near-real-time incremental transcription, not token-by-token streaming." },
];

const waveformHeights = [28, 46, 32, 64, 24, 58, 38, 72, 44, 30, 66, 35, 54, 26, 60, 40];
const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
const websocketUrl = `${backendUrl.replace(/^http/, "ws")}/ws/transcribe`;

function float32ToInt16(input: Float32Array): Int16Array {
  const output = new Int16Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, input[index]));
    output[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output;
}

export function VoiceAgentConsole() {
  const [mode, setMode] = useState<Mode>("listening");
  const [isRecording, setIsRecording] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isFinalizing, setIsFinalizing] = useState(false);
  const [transcript, setTranscript] = useState("Start recording to stream microphone audio and see live transcription updates.");
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);

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
  const errorRef = useRef<string | null>(null);

  useEffect(() => {
    errorRef.current = error;
  }, [error]);

  const stopAudioGraph = () => {
    processorNodeRef.current?.disconnect();
    sourceNodeRef.current?.disconnect();
    gainNodeRef.current?.disconnect();
    processorNodeRef.current = null;
    sourceNodeRef.current = null;
    gainNodeRef.current = null;

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
      setTranscript("Opening the microphone and connecting to the backend stream...");
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
        startTransition(() => {
          setMode("listening");
        });
      };

      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data) as StreamMessage;

        if (payload.type === "ready") {
          streamReadyRef.current = true;
          setTranscript("Live stream connected. Start speaking.");
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
          startTransition(() => {
            setMode("thinking");
          });
          return;
        }

        if (payload.type === "final") {
          receivedFinalRef.current = true;
          normalCloseRef.current = true;
          applyStreamPayload(payload);
          setIsFinalizing(false);
          isFinalizingRef.current = false;
          startTransition(() => {
            setMode("responding");
          });
          socket.close();
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

  const stopStreaming = () => {
    const socket = websocketRef.current;
    setIsRecording(false);
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

  const latencyCards = [
    {
      title: "Total latency",
      value: metrics ? `${metrics.total_ms} ms` : "--",
      detail: "Per-update backend processing time for the current buffer pass.",
    },
    {
      title: "Buffered audio",
      value: metrics ? `${metrics.buffered_audio_ms} ms` : "--",
      detail: "How much microphone audio has been accumulated for the current transcript window.",
    },
    {
      title: "Transcribe pass",
      value: metrics ? `${metrics.transcribe_ms} ms` : "--",
      detail: "Time spent in faster-whisper for the latest incremental transcription run.",
    },
  ];

  return (
    <main className="console-shell">
      <section className="console-frame">
        <header className="topbar surface">
          <div>
            <p className="kicker">NeuroTalk / Sovereign Voice Agent</p>
            <h1>Control surface for an intelligence-first voice interface.</h1>
          </div>
          <div className="topbar-meta">
            <span className="status-pill is-live">Realtime STT stream</span>
            <span className="status-pill">{websocketUrl}</span>
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
              <div className="orbital-core" aria-hidden="true">
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
                        "--bar-height": `${height}px`,
                        "--bar-delay": `${index * 0.08}s`,
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
                    ? "Microphone is active and PCM chunks are streaming"
                    : isFinalizing
                      ? "Waiting for the final transcript pass"
                      : "Ready to open a live WebSocket transcription session"}
                </span>
                <span>{error ?? "Partial transcript updates will appear while you speak"}</span>
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
                <p className="kicker">Session telemetry</p>
                <span className="mini-dot" />
              </div>
              <div className="metric-grid">
                <div>
                  <span>Total latency</span>
                  <strong>{metrics ? `${metrics.total_ms} ms` : "--"}</strong>
                </div>
                <div>
                  <span>Model load</span>
                  <strong>{metrics ? `${metrics.model_load_ms} ms` : "--"}</strong>
                </div>
                <div>
                  <span>Transcribe time</span>
                  <strong>{metrics ? `${metrics.transcribe_ms} ms` : "--"}</strong>
                </div>
                <div>
                  <span>Client session</span>
                  <strong>{metrics?.client_roundtrip_ms ? `${metrics.client_roundtrip_ms} ms` : "--"}</strong>
                </div>
              </div>
            </article>

            <article className="surface transcript-panel">
              <div className="section-heading">
                <p className="kicker">Transcribed text</p>
                <span className="status-pill is-ghost">{error ? "Error state" : isRecording ? "Live partials" : "Latest result"}</span>
              </div>
              <p className="transcript-line">{transcript}</p>
              <div className="transcript-footer">
                <span>{error ?? `Request ID: ${debugInfo?.request_id ?? "--"}`}</span>
                <span>Language: {debugInfo?.detected_language ?? "--"}</span>
              </div>
            </article>
          </aside>
        </section>

        <section className="dashboard-grid">
          <article className="surface stack-panel">
            <div className="section-heading">
              <p className="kicker">Streaming stack</p>
              <span className="section-note">Minimal live path with current backend</span>
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
              <p className="kicker">Latency profile</p>
              <span className="section-note">Useful for tuning stream cadence and model choice</span>
            </div>
            <div className="card-grid">
              {latencyCards.map((card) => (
                <div className="info-card" key={card.title}>
                  <span>{card.title}</span>
                  <strong>{card.value}</strong>
                  <p>{card.detail}</p>
                </div>
              ))}
            </div>
          </article>

          <article className="surface presets-panel">
            <div className="section-heading">
              <p className="kicker">Runtime notes</p>
              <span className="section-note">Simple operational guidance for the current implementation</span>
            </div>
            <div className="preset-list">
              {runtimeNotes.map((item) => (
                <div className="preset-card" key={item.name}>
                  <h3>{item.name}</h3>
                  <p>{item.tone}</p>
                </div>
              ))}
            </div>
            <div className="debug-strip">
              <span>Buffered: {metrics ? `${metrics.buffered_audio_ms} ms` : "--"}</span>
              <span>Chunks: {debugInfo?.chunks_received ?? "--"}</span>
              <span>Bytes: {debugInfo?.audio_bytes ?? "--"}</span>
              <span>Read: {metrics ? `${metrics.request_read_ms} ms` : "--"}</span>
              <span>Write: {metrics ? `${metrics.file_write_ms} ms` : "--"}</span>
              <span>Model: {debugInfo ? `${debugInfo.model_size} / ${debugInfo.device} / ${debugInfo.compute_type}` : "--"}</span>
              <span>Sample rate: {debugInfo?.sample_rate ?? "--"}</span>
            </div>
          </article>
        </section>
      </section>
    </main>
  );
}
