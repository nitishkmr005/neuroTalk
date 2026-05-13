"use client";

/**
 * Meeting Recorder — self-contained feature module.
 *
 * To remove this feature entirely:
 * 1. Delete  frontend/components/meeting-recorder.tsx  (this file)
 * 2. Remove  MeetingRecorder import + usage from voice-agent-console.tsx
 * 3. Remove  "Meeting Recorder" section from frontend/app/globals.css
 * 4. Delete  backend/app/meeting/
 * 5. Remove  meeting_router lines from backend/app/main.py
 */

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

interface Segment {
  text: string;
  at: number; // elapsed seconds when segment was captured
}

interface Props {
  backendUrl: string;
  active?: boolean; // false = keep mounted but hide (CSS show/hide for persistence)
}

const MAX_SEGMENT_BYTES = 4 * 1024 * 1024;
const SEGMENT_INTERVAL_MS = 8_000;

const LS_SEGMENTS = "nt-meeting-segments";
const LS_SUMMARY  = "nt-meeting-summary";
const LS_DURATION = "nt-meeting-duration";

function getMimeType(): string {
  for (const t of ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"]) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

function formatTime(s: number): string {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

function countWords(segments: Segment[]): number {
  return segments.reduce((n, s) => n + s.text.split(/\s+/).filter(Boolean).length, 0);
}

// ── Lightweight markdown renderer ──────────────────────────────────────────

function renderInline(text: string): ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

function renderMarkdown(text: string): ReactNode {
  if (!text) return null;
  const lines = text.split("\n");
  const nodes: ReactNode[] = [];
  let listItems: ReactNode[] = [];
  let key = 0;

  const flushList = () => {
    if (listItems.length) {
      nodes.push(<ul key={key++} className="meeting-summary-list">{listItems}</ul>);
      listItems = [];
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line === "") { flushList(); continue; }

    if (/^[-*]\s/.test(line)) {
      listItems.push(<li key={key++}>{renderInline(line.replace(/^[-*]\s/, ""))}</li>);
      continue;
    }

    flushList();

    const isSectionHeader = /^\*\*[^*]+\*\*\s*$/.test(line) || /^#{1,3}\s/.test(line);
    if (isSectionHeader) {
      const clean = line.replace(/^\*\*|\*\*\s*$/g, "").replace(/^#{1,3}\s/, "");
      nodes.push(<p key={key++} className="meeting-summary-heading">{clean}</p>);
      continue;
    }

    nodes.push(<p key={key++} className="meeting-summary-para">{renderInline(line)}</p>);
  }

  flushList();
  return <>{nodes}</>;
}

// ── Icons ────────────────────────────────────────────────────────────────────

function DownloadIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
  );
}

function SaveIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
      <polyline points="17 21 17 13 7 13 7 21"/>
      <polyline points="7 3 7 8 15 8"/>
    </svg>
  );
}

// ── Component ────────────────────────────────────────────────────────────────

export function MeetingRecorder({ backendUrl, active = true }: Props) {
  // ── State — initialised from localStorage for cross-session persistence ──

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [segments, setSegments] = useState<Segment[]>(() => {
    if (typeof window === "undefined") return [];
    try { return JSON.parse(localStorage.getItem(LS_SEGMENTS) ?? "[]") as Segment[]; }
    catch { return []; }
  });
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<string>(() => {
    if (typeof window === "undefined") return "";
    return localStorage.getItem(LS_SUMMARY) ?? "";
  });
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [savedPaths, setSavedPaths] = useState<string[]>([]);

  const streamRef       = useRef<MediaStream | null>(null);
  const masterRecRef    = useRef<MediaRecorder | null>(null);
  const segRecRef       = useRef<MediaRecorder | null>(null);
  const masterChunksRef = useRef<Blob[]>([]);
  const timerRef        = useRef<ReturnType<typeof setInterval> | null>(null);
  const segTimerRef     = useRef<ReturnType<typeof setInterval> | null>(null);
  const mimeTypeRef     = useRef("");
  const downloadUrlRef  = useRef<string | null>(null);
  const stoppedRef      = useRef(false);
  const elapsedRef      = useRef(0);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);
  const summaryEndRef   = useRef<HTMLDivElement | null>(null);

  // ── localStorage persistence ─────────────────────────────────────────────

  useEffect(() => {
    try { localStorage.setItem(LS_SEGMENTS, JSON.stringify(segments)); }
    catch { /* quota exceeded */ }
  }, [segments]);

  useEffect(() => {
    try { localStorage.setItem(LS_SUMMARY, summary); }
    catch { /* quota exceeded */ }
  }, [summary]);

  // ── Device enumeration ───────────────────────────────────────────────────

  const loadDevices = useCallback(async () => {
    try {
      let inputs = (await navigator.mediaDevices.enumerateDevices()).filter(
        (d) => d.kind === "audioinput"
      );
      if (inputs.length > 0 && !inputs[0].label) {
        const tmp = await navigator.mediaDevices.getUserMedia({ audio: true });
        tmp.getTracks().forEach((t) => t.stop());
        inputs = (await navigator.mediaDevices.enumerateDevices()).filter(
          (d) => d.kind === "audioinput"
        );
      }
      setDevices(inputs);
      if (inputs.length > 0) setSelectedDeviceId(inputs[0].deviceId);
    } catch { /* Permission denied */ }
  }, []);

  useEffect(() => { void loadDevices(); }, [loadDevices]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [segments]);

  useEffect(() => {
    summaryEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [summary]);

  // ── Save to folder ────────────────────────────────────────────────────────

  const saveToFolder = useCallback(async (transcript?: string, sum?: string) => {
    if (!transcript && !sum) return;
    try {
      const resp = await fetch(`${backendUrl}/meeting/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: transcript ?? null, summary: sum ?? null }),
      });
      if (resp.ok) {
        const data = (await resp.json()) as { saved?: string[] };
        if (data.saved?.length) setSavedPaths((prev) => [...prev, ...data.saved!]);
      }
    } catch { /* backend unavailable */ }
  }, [backendUrl]);

  // ── Transcription ────────────────────────────────────────────────────────

  const transcribeSegment = useCallback(async (chunks: Blob[]) => {
    if (chunks.length === 0) return;
    const mimeType = mimeTypeRef.current || "audio/webm";
    const blob = new Blob(chunks, { type: mimeType });
    if (blob.size < 5_000 || blob.size > MAX_SEGMENT_BYTES) return;
    setIsTranscribing(true);
    try {
      const form = new FormData();
      form.append("audio", blob, "segment.webm");
      const resp = await fetch(`${backendUrl}/transcribe`, { method: "POST", body: form });
      if (!resp.ok) return;
      const data = (await resp.json()) as { text?: string };
      if (data.text?.trim()) {
        setSegments((prev) => [...prev, { text: data.text!.trim(), at: elapsedRef.current }]);
      }
    } catch { /* Network error */ }
    finally { setIsTranscribing(false); }
  }, [backendUrl]);

  // ── Segment recorder ────────────────────────────────────────────────────

  const startSegmentRecorder = useCallback((stream: MediaStream) => {
    const mimeType = mimeTypeRef.current;
    const rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    rec.onstop = () => { void transcribeSegment(chunks); };
    rec.start(200);
    segRecRef.current = rec;
  }, [transcribeSegment]);

  const rotateSegment = useCallback((stream: MediaStream) => {
    if (stoppedRef.current) return;
    segRecRef.current?.stop();
    startSegmentRecorder(stream);
  }, [startSegmentRecorder]);

  // ── Start / stop ─────────────────────────────────────────────────────────

  const startRecording = useCallback(async () => {
    setError(null);
    setSummary("");
    setSavedPaths([]);
    stoppedRef.current = false;
    elapsedRef.current = 0;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true,
      });
      streamRef.current = stream;
      const mimeType = getMimeType();
      mimeTypeRef.current = mimeType;

      masterChunksRef.current = [];
      const masterRec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      masterRec.ondataavailable = (e) => {
        if (e.data.size > 0) masterChunksRef.current.push(e.data);
      };
      masterRec.onstop = () => {
        const blob = new Blob(masterChunksRef.current, { type: mimeType || "audio/webm" });
        if (downloadUrlRef.current) URL.revokeObjectURL(downloadUrlRef.current);
        const url = URL.createObjectURL(blob);
        downloadUrlRef.current = url;
        setDownloadUrl(url);
      };
      masterRec.start(500);
      masterRecRef.current = masterRec;

      startSegmentRecorder(stream);
      timerRef.current = setInterval(() => {
        setElapsed((p) => { elapsedRef.current = p + 1; return p + 1; });
      }, 1_000);
      segTimerRef.current = setInterval(() => rotateSegment(stream), SEGMENT_INTERVAL_MS);

      setIsRecording(true);
      setElapsed(0);
      setSegments([]);
      setDownloadUrl(null);
    } catch (err) {
      setError(err instanceof DOMException ? err.message : "Could not access microphone.");
    }
  }, [selectedDeviceId, startSegmentRecorder, rotateSegment]);

  const stopRecording = useCallback(() => {
    stoppedRef.current = true;
    if (segTimerRef.current) { clearInterval(segTimerRef.current); segTimerRef.current = null; }
    if (timerRef.current)    { clearInterval(timerRef.current);    timerRef.current    = null; }
    segRecRef.current?.stop();
    masterRecRef.current?.stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setIsRecording(false);
    // Save transcript to folder after a brief delay so segments state is flushed
    setTimeout(() => {
      setSegments((current) => {
        if (current.length > 0) {
          const text = current.map((s) => `[${formatTime(s.at)}] ${s.text}`).join("\n\n");
          void saveToFolder(text);
        }
        return current;
      });
    }, 500);
  }, [saveToFolder]);

  useEffect(() => () => {
    stoppedRef.current = true;
    if (segTimerRef.current) clearInterval(segTimerRef.current);
    if (timerRef.current)    clearInterval(timerRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    if (downloadUrlRef.current) URL.revokeObjectURL(downloadUrlRef.current);
  }, []);

  // ── Summarization ────────────────────────────────────────────────────────

  const summarize = useCallback(async () => {
    const transcript = segments.map((s) => s.text).join(" ").trim();
    if (!transcript) return;
    setSummary("");
    setIsSummarizing(true);
    let finalSummary = "";
    try {
      const resp = await fetch(`${backendUrl}/meeting/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: transcript }),
      });
      if (!resp.ok || !resp.body) {
        setSummary("Summarization failed. Is the LLM running?");
        return;
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        finalSummary += chunk;
        setSummary((prev) => prev + chunk);
      }
      // Auto-save summary to folder
      if (finalSummary.trim()) void saveToFolder(undefined, finalSummary);
    } catch {
      setSummary("Summarization failed. Is the LLM running?");
    } finally {
      setIsSummarizing(false);
    }
  }, [backendUrl, segments, saveToFolder]);

  // ── Downloads ────────────────────────────────────────────────────────────

  const triggerDownload = (url: string, filename: string) => {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const ts = () => new Date().toISOString().slice(0, 16).replace("T", "_");

  const downloadAudio      = useCallback(() => { if (downloadUrl) triggerDownload(downloadUrl, `meeting_${ts()}.webm`); }, [downloadUrl]);
  const downloadTranscript = useCallback(() => {
    if (!segments.length) return;
    const text = segments.map((s) => `[${formatTime(s.at)}] ${s.text}`).join("\n\n");
    const blob = new Blob([text], { type: "text/plain" });
    const url  = URL.createObjectURL(blob);
    triggerDownload(url, `transcript_${ts()}.txt`);
    URL.revokeObjectURL(url);
  }, [segments]);
  const downloadSummaryFn  = useCallback(() => {
    if (!summary) return;
    const blob = new Blob([summary], { type: "text/plain" });
    const url  = URL.createObjectURL(blob);
    triggerDownload(url, `summary_${ts()}.txt`);
    URL.revokeObjectURL(url);
  }, [summary]);

  // ── Render ────────────────────────────────────────────────────────────────

  const hasTranscript = segments.length > 0;
  const hasSummary    = Boolean(summary);
  const wordCount     = countWords(segments);
  const lastSavedPath = savedPaths.length > 0 ? savedPaths[savedPaths.length - 1] : null;

  return (
    <section className="meeting-recorder-view" style={{ display: active ? "grid" : "none" }}>

      {/* ── Left: controls panel ── */}
      <article className="meeting-controls-panel surface">

        {/* Status header */}
        <div className="meeting-status-header">
          <div className="meeting-status-badge">
            {isRecording ? (
              <><span className="meeting-rec-dot" aria-hidden="true" /><span>Recording</span></>
            ) : downloadUrl ? (
              <><span className="meeting-done-dot" aria-hidden="true" /><span>Session ended</span></>
            ) : hasTranscript ? (
              <><span className="meeting-done-dot" aria-hidden="true" /><span>Session loaded</span></>
            ) : (
              <span className="meeting-idle-label">Ready to record</span>
            )}
          </div>
          {isRecording && (
            <span className="meeting-elapsed-badge">{formatTime(elapsed)}</span>
          )}
        </div>

        {/* Session stats */}
        {hasTranscript && (
          <div className="meeting-session-stats">
            <span>{segments.length} {segments.length === 1 ? "segment" : "segments"}</span>
            <span className="meeting-stats-sep" aria-hidden="true">·</span>
            <span>~{wordCount} words</span>
            {!isRecording && elapsed > 0 && (
              <>
                <span className="meeting-stats-sep" aria-hidden="true">·</span>
                <span>{formatTime(elapsed)}</span>
              </>
            )}
          </div>
        )}

        {/* Description */}
        <p className="meeting-controls-desc">
          {isRecording
            ? "Transcript updates every ~8 s. Stop when done."
            : hasTranscript
            ? "Previous session loaded. Start a new one or summarize."
            : "Select an audio input and start recording."}
        </p>

        {/* Device selector */}
        <div className="meeting-field">
          <label className="meeting-field-label">Audio Input</label>
          <select
            className="meeting-recorder-select"
            value={selectedDeviceId}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
            disabled={isRecording}
            aria-label="Select audio input device"
          >
            {devices.length === 0 && <option value="">No devices found</option>}
            {devices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `Microphone (${d.deviceId.slice(0, 8)})`}
              </option>
            ))}
          </select>
        </div>

        {error && <p className="meeting-recorder-error">{error}</p>}

        {/* Start / Stop */}
        <button
          type="button"
          className={`meeting-recorder-btn${isRecording ? " is-recording" : ""}`}
          onClick={isRecording ? stopRecording : () => void startRecording()}
        >
          {isRecording
            ? <><span className="meeting-rec-dot" aria-hidden="true" />Stop Recording</>
            : hasTranscript
            ? "Start New Recording"
            : "Start Recording"}
        </button>

        {/* Summarize */}
        {hasTranscript && !isRecording && (
          <button
            type="button"
            className="meeting-recorder-btn"
            onClick={() => void summarize()}
            disabled={isSummarizing}
            style={{ opacity: isSummarizing ? 0.65 : 1 }}
          >
            {isSummarizing ? "Summarizing…" : "Summarize Meeting"}
          </button>
        )}

        {/* Transcribing indicator */}
        {isTranscribing && (
          <span className="status-pill is-live" style={{ width: "fit-content" }}>
            Transcribing…
          </span>
        )}

        {/* Saved confirmation */}
        {lastSavedPath && (
          <div className="meeting-saved-notice">
            <SaveIcon />
            <span>Saved to <code>recordings/</code></span>
          </div>
        )}

        {/* Downloads — pinned to bottom */}
        {(downloadUrl || hasTranscript) && (
          <div className="meeting-downloads-section">
            <span className="meeting-field-label">Downloads</span>
            <div className="meeting-recorder-downloads">
              {downloadUrl && (
                <button type="button" className="meeting-download-btn" onClick={downloadAudio}>
                  <DownloadIcon /> Audio (.webm)
                </button>
              )}
              <button
                type="button"
                className="meeting-download-btn"
                onClick={downloadTranscript}
                disabled={!hasTranscript}
                style={{ opacity: hasTranscript ? 1 : 0.4 }}
              >
                <DownloadIcon /> Transcript (.txt)
              </button>
              <button
                type="button"
                className="meeting-download-btn"
                onClick={downloadSummaryFn}
                disabled={!hasSummary || isSummarizing}
                style={{ opacity: hasSummary && !isSummarizing ? 1 : 0.4 }}
              >
                <DownloadIcon /> Summary (.txt)
              </button>
            </div>
          </div>
        )}
      </article>

      {/* ── Right: transcript + summary ── */}
      <aside className="telemetry-stack">

        {/* Transcript panel */}
        <article className="surface transcript-panel">
          <div className="section-heading">
            <p className="kicker">Live Transcript</p>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              {isRecording && <span className="status-pill is-live">Live</span>}
              {!isRecording && hasTranscript && <span className="status-pill is-ghost">Done</span>}
              {!isRecording && !hasTranscript && <span className="status-pill is-ghost">Ready</span>}
            </div>
          </div>

          <div className="chat-feed">
            {hasTranscript ? (
              <>
                {segments.map((seg, i) => (
                  <div key={i} className="chat-row chat-row--user meeting-segment-row">
                    <div className="chat-bubble chat-bubble--user meeting-segment-bubble">
                      <span className="meeting-segment-timestamp">{formatTime(seg.at)}</span>
                      <p className="chat-text">{seg.text}</p>
                    </div>
                  </div>
                ))}
                {isTranscribing && (
                  <div className="chat-row chat-row--user">
                    <div className="chat-bubble chat-bubble--user meeting-segment-bubble">
                      <div className="chat-typing-indicator" aria-label="Transcribing">
                        <span /><span /><span />
                      </div>
                    </div>
                  </div>
                )}
                <div ref={transcriptEndRef} />
              </>
            ) : (
              <div className="chat-empty">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
                <p>Transcript will appear here</p>
                <span>Updates every ~8 seconds while recording</span>
              </div>
            )}
          </div>
        </article>

        {/* Summary panel */}
        {(hasSummary || isSummarizing) && (
          <article className="surface transcript-panel">
            <div className="section-heading">
              <p className="kicker">Meeting Summary</p>
              {isSummarizing && <span className="status-pill is-live">Generating…</span>}
            </div>
            <div className="chat-feed meeting-summary-body">
              {renderMarkdown(summary)}
              {isSummarizing && <span className="meeting-rec-cursor" aria-hidden="true" />}
              <div ref={summaryEndRef} />
            </div>
          </article>
        )}

      </aside>
    </section>
  );
}
