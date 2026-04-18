"use client";

import { startTransition, useDeferredValue, useEffect, useState, type CSSProperties } from "react";

type Mode = "listening" | "thinking" | "responding";

const modeOrder: Mode[] = ["listening", "thinking", "responding"];

const modeConfig: Record<
  Mode,
  {
    eyebrow: string;
    headline: string;
    summary: string;
    accent: string;
    latency: string;
    confidence: string;
    voice: string;
  }
> = {
  listening: {
    eyebrow: "Acquisition layer engaged",
    headline: "Hearing context before it becomes text.",
    summary:
      "The agent is isolating intent, emotional contour, and operational constraints from the raw voice stream.",
    accent: "active-listening",
    latency: "112 ms",
    confidence: "98.2%",
    voice: "Wideband capture",
  },
  thinking: {
    eyebrow: "Reasoning lattice in motion",
    headline: "Synthesizing memory, tools, and strategy in real time.",
    summary:
      "The system is ranking actions, checking internal policy, and preparing a response path with tool awareness.",
    accent: "deep-reasoning",
    latency: "184 ms",
    confidence: "96.4%",
    voice: "Cognitive mesh",
  },
  responding: {
    eyebrow: "Speech engine delivering",
    headline: "Answering with precision, timing, and tone control.",
    summary:
      "The response layer is shaping cadence, grounding the answer, and streaming speech with adaptive turn-taking.",
    accent: "voice-delivery",
    latency: "138 ms",
    confidence: "99.1%",
    voice: "Expressive synthesis",
  },
};

const transcriptBank: Record<Mode, string[]> = {
  listening: [
    "User voiceprint verified. Acoustic noise floor normalized.",
    "Intent candidates detected: scheduling, technical planning, phased rollout.",
    "Emphasis pattern suggests the user wants ambition without backend coupling yet.",
  ],
  thinking: [
    "Cross-referencing interface goals against future orchestration hooks.",
    "Prioritizing a visual shell that can later expose live VAD, tool calls, and transcript state.",
    "Selecting a response posture: confident, concise, operationally transparent.",
  ],
  responding: [
    "Streaming interface guidance: present a premium prototype with strong motion and clear modular zones.",
    "Answer structure tuned for a technical builder: immediate visibility, live metrics, extensible cards.",
    "Turn closing with room for backend expansion: websocket audio, function traces, memory ledger.",
  ],
};

const orchestrationSteps = [
  { label: "Intent parser", detail: "Semantic pressure map built from live utterance.", status: "online" },
  { label: "Memory fabric", detail: "Session history compressed into fast retrieval cues.", status: "stable" },
  { label: "Tool router", detail: "Awaiting backend adapters for actions and retrieval.", status: "pending" },
  { label: "Voice renderer", detail: "Prosody engine prepared for adaptive delivery.", status: "online" },
];

const systemCards = [
  {
    title: "Conversation gravity",
    value: "14.8x",
    detail: "Signal-to-noise focus during multi-step dialogue.",
  },
  {
    title: "Reasoning depth",
    value: "Layer 07",
    detail: "Fast path, reflective path, and response guardrails are aligned.",
  },
  {
    title: "Context retention",
    value: "92 min",
    detail: "Short-term memory budget reserved for live sessions.",
  },
];

const presets = [
  { name: "Strategist", tone: "Measured, high-context, boardroom calm." },
  { name: "Operator", tone: "Fast, directive, low-latency execution." },
  { name: "Concierge", tone: "Warm, anticipatory, premium assistance." },
];

const waveformHeights = [28, 46, 32, 64, 24, 58, 38, 72, 44, 30, 66, 35, 54, 26, 60, 40];

export function VoiceAgentConsole() {
  const [mode, setMode] = useState<Mode>("thinking");
  const [transcriptIndex, setTranscriptIndex] = useState(0);

  useEffect(() => {
    const transcriptTimer = window.setInterval(() => {
      startTransition(() => {
        setTranscriptIndex((current) => current + 1);
      });
    }, 2600);

    const modeTimer = window.setInterval(() => {
      startTransition(() => {
        setMode((current) => {
          const currentIndex = modeOrder.indexOf(current);
          return modeOrder[(currentIndex + 1) % modeOrder.length];
        });
      });
    }, 7200);

    return () => {
      window.clearInterval(transcriptTimer);
      window.clearInterval(modeTimer);
    };
  }, []);

  const transcript = transcriptBank[mode][transcriptIndex % transcriptBank[mode].length];
  const deferredTranscript = useDeferredValue(transcript);
  const activeMode = modeConfig[mode];

  return (
    <main className="console-shell">
      <section className="console-frame">
        <header className="topbar surface">
          <div>
            <p className="kicker">NeuroTalk / Sovereign Voice Agent</p>
            <h1>Control surface for an intelligence-first voice interface.</h1>
          </div>
          <div className="topbar-meta">
            <span className="status-pill is-live">Frontend prototype</span>
            <span className="status-pill">Backend hooks later</span>
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

            <div className="mode-switcher">
              {modeOrder.map((item) => (
                <button
                  type="button"
                  key={item}
                  className={item === mode ? "mode-button is-selected" : "mode-button"}
                  onClick={() => {
                    startTransition(() => {
                      setMode(item);
                      setTranscriptIndex(0);
                    });
                  }}
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
                  <span>Response latency</span>
                  <strong>{activeMode.latency}</strong>
                </div>
                <div>
                  <span>Intent certainty</span>
                  <strong>{activeMode.confidence}</strong>
                </div>
                <div>
                  <span>Voice profile</span>
                  <strong>{activeMode.voice}</strong>
                </div>
                <div>
                  <span>Turn sync</span>
                  <strong>Bi-directional</strong>
                </div>
              </div>
            </article>

            <article className="surface transcript-panel">
              <div className="section-heading">
                <p className="kicker">Live internal transcript</p>
                <span className="status-pill is-ghost">Mock stream</span>
              </div>
              <p className="transcript-line">{deferredTranscript}</p>
              <div className="transcript-footer">
                <span>Speaker energy balanced</span>
                <span>Turn prediction locked</span>
              </div>
            </article>
          </aside>
        </section>

        <section className="dashboard-grid">
          <article className="surface stack-panel">
            <div className="section-heading">
              <p className="kicker">Cognitive stack</p>
              <span className="section-note">Designed to plug into the backend later</span>
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
              <p className="kicker">Intelligence profile</p>
              <span className="section-note">Operator facing overview</span>
            </div>
            <div className="card-grid">
              {systemCards.map((card) => (
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
              <p className="kicker">Voice posture presets</p>
              <span className="section-note">Prepared personas for future synthesis</span>
            </div>
            <div className="preset-list">
              {presets.map((preset) => (
                <div className="preset-card" key={preset.name}>
                  <h3>{preset.name}</h3>
                  <p>{preset.tone}</p>
                </div>
              ))}
            </div>
          </article>
        </section>
      </section>
    </main>
  );
}
