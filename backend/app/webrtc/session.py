"""WebRTC session: RTP audio → STT → LLM → TTS pipeline over a data channel."""

# Standard library imports — all come with Python, no installation needed
import asyncio      # Python's built-in async scheduler (the "event loop")
import base64       # Converts binary bytes ↔ text strings (used to send WAV over JSON)
import json         # Parses/serialises JSON messages on the data channel
import re           # Regular expressions for sentence boundary detection
import wave         # Writes PCM bytes into a WAV file that Whisper can read
from contextlib import suppress  # Silences a specific exception type (used with CancelledError)
from time import perf_counter    # High-resolution timer for measuring latency in milliseconds

# Third-party packages
import av           # FFmpeg Python bindings — decodes Opus RTP frames to raw PCM
import numpy as np  # Numerical arrays — used for RMS energy calculation in barge-in detection
# aiortc provides the WebRTC peer connection and track objects
from aiortc import MediaStreamTrack, RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError  # Raised when the remote peer closes the audio track
from loguru import logger  # Structured logging with session_id context

# Internal service imports — each wraps a model or processing step
from app.services.denoise import get_denoise_service  # DeepFilterNet3 noise removal
from app.services.llm import stream_llm_response      # LLM token streaming (all providers)
from app.services.stt import get_stt_service          # Whisper speech-to-text
from app.services.tts import get_tts_service          # Kokoro/Chatterbox TTS synthesis
from app.services.vad import StreamingVAD, get_vad_service  # Silero voice activity detection
from app.utils.emotion import clean_for_tts, strip_emotion_tags  # Remove [emotion] tags from LLM output
from config.settings import get_settings  # Singleton Pydantic settings object

# Matches utterances that mean "please wait/pause" so we don't forward them to the LLM
_PAUSE_PATTERN = re.compile(
    r"^\s*(wait|hold on|hold up|one moment|one sec(?:ond)?|just a (?:moment|second|sec)|"
    r"give me a (?:second|moment|sec)|hang on|please wait|just wait|ok wait|okay wait|"
    r"stop|stop it|stop please|please stop|ok stop|okay stop)\s*[.!?,]?\s*$",
    re.IGNORECASE,
)
# Matches end-of-sentence punctuation followed by whitespace or end-of-string
# Used to split LLM output into sentences for per-sentence TTS
_SENT_BOUNDARY = re.compile(r"[.!?](?:\s|$)")
# Minimum character count before a sentence fragment is sent to TTS
# Prevents synthesising tiny phrases like "I." or "Yes."
_MIN_SENTENCE_CHARS = 15

# RMS energy above this level (on a -1..+1 scale) is considered a barge-in attempt
# Slightly lower than browser-side 0.15 because we see raw Opus-decoded audio without browser AGC
_BARGE_IN_THRESHOLD = 0.15
# How many consecutive above-threshold frames before declaring a barge-in
# Requires 3 frames in a row to avoid false triggers from short noise bursts
_BARGE_IN_FRAMES = 3

# No STUN server on the server side: aiortc's setLocalDescription blocks until
# ICE gathering completes, and STUN lookups to stun.l.google.com can stall
# 5+ seconds on restricted networks. For localhost/LAN use, host candidates
# (local IPs) are all that's needed. Add STUN/TURN here for remote deployments.
_RTC_CONFIG = RTCConfiguration(iceServers=[])  # Empty list = no STUN/TURN, LAN-only


class WebRTCSession:
    """One WebRTC peer-connection per browser tab.

    Audio arrives as RTP (Opus → PCM via aiortc/av).
    All JSON signalling (ready / partial / llm_* / tts_*) travels over an
    ordered RTCDataChannel named "signaling" — same message schema as the
    WebSocket path so the frontend handler is reusable.

    Attributes:
        session_id: Short hex ID shared with the frontend as ``request_id``.
        pc: The underlying ``RTCPeerConnection``.
        dc: The signalling ``RTCDataChannel``, set once the browser opens it.
    """

    def __init__(self, session_id: str) -> None:
        """Initialise session state and register peer-connection event handlers.

        Args:
            session_id: Short hex ID used for logging and sent to the frontend.
        """
        self.session_id = session_id  # Short unique ID (e.g. "a1b2c3d4") shared with the browser for log correlation
        self._settings = get_settings()  # Load config once; reused throughout the session
        self.pc = RTCPeerConnection(_RTC_CONFIG)  # The WebRTC peer connection — handles ICE, DTLS, RTP negotiation
        self.dc = None  # Data channel handle; set later when the browser creates it (event-driven)

        # Audio accumulation — raw PCM bytes grow here until transcription runs
        self._pcm_buffer = bytearray()  # Growable byte buffer — all 16 kHz mono PCM since last clear
        self._chunk_count = 0           # How many resampled frames have been added (for debug logging)
        self._last_emit_at = 0.0        # perf_counter() timestamp of last partial STT emit (throttling)
        self._last_text_sent = ""       # Last transcript string sent as "partial" (dedup guard)
        self._sample_rate = 16_000      # Always 16 kHz after resampling; client-reported value is ignored

        # Concurrency helpers — these are the "traffic lights" that coordinate async tasks
        self._send_lock = asyncio.Lock()        # Mutex so two coroutines don't write to the data channel simultaneously
        self._transcribe_lock = asyncio.Lock()  # Mutex so denoise+STT (CPU-bound) don't run in parallel with themselves
        self._llm_task: asyncio.Task | None = None         # Handle to the currently running _run_llm() task (or None)
        self._tts_task: asyncio.Task | None = None         # Handle to the currently running _tts_sentence_pipeline() task
        self._pending_llm_call: str | None = None          # Text queued for LLM after the current LLM task finishes
        self._latest_llm_input = ""                        # Last text that was actually sent to the LLM (dedup guard)
        self._llm_seq = 0                                  # Sequence number — increments each LLM call so browser discards stale responses
        self._interrupt_event = asyncio.Event()            # asyncio.Event acts like a flag: .set() signals interruption, .clear() resets it
        self._silence_debounce_task: asyncio.Task | None = None    # Timer task that fires LLM after silence; cancelled if user resumes speaking
        self._speech_finalization_task: asyncio.Task | None = None # Task that runs final STT + schedules LLM after turn is confirmed

        # Conversation state — persists across turns for multi-turn dialogue
        self._conversation_history: list[dict[str, str]] = []  # List of {"role": "user"/"assistant", "content": "..."} dicts
        self._voice_id: str = self._settings.tts_kokoro_voice  # Current TTS voice name (can be changed per-session)
        self._tts_speed: float = self._settings.tts_kokoro_speed  # TTS speed multiplier (0.5 = slow, 2.0 = fast)

        # Barge-in state — tracks whether the AI is currently speaking so interrupts work
        self._is_agent_speaking = False  # True between tts_start and tts_done; used to trigger barge-in on new speech
        self._barge_in_count = 0        # Consecutive above-threshold RMS frames; resets to 0 on quiet frames
        self._vad_stream: StreamingVAD | None = (
            get_vad_service().create_stream() if self._settings.stream_vad_enabled else None
            # Creates a stateful Silero VAD stream if VAD is enabled in settings, otherwise None (RMS fallback used)
        )

        # Background tasks — long-running coroutines that run for the lifetime of the session
        self._audio_task: asyncio.Task | None = None  # Handle to _consume_audio(); started when the audio track arrives
        self._closed = False  # Guards against double-cleanup if connection state changes multiple times

        self._register_pc_handlers()  # Wire up event callbacks on self.pc before any SDP exchange happens

    # ── Peer-connection lifecycle ────────────────────────────────────────────

    def _register_pc_handlers(self) -> None:
        """Attach aiortc event handlers for track, datachannel, and state changes.

        Registers closures on ``self.pc`` for:
        - ``track``: starts the audio consumer coroutine when an audio track arrives.
        - ``datachannel``: wires up open/message callbacks on the signalling channel.
        - ``connectionstatechange``: triggers cleanup on failure or disconnection.
        """
        @self.pc.on("track")  # aiortc calls this decorator's function when the browser sends an audio track
        def on_track(track: MediaStreamTrack) -> None:
            if track.kind == "audio":  # Ignore video tracks; only audio is relevant here
                # asyncio.ensure_future() schedules _consume_audio() as a background Task
                # It does NOT wait for it to finish — it just starts it and moves on
                self._audio_task = asyncio.ensure_future(self._consume_audio(track))

        @self.pc.on("datachannel")  # Fired when the browser creates the "signaling" data channel
        def on_datachannel(channel) -> None:
            self.dc = channel  # Save the channel reference so _send_json() can use it

            @channel.on("open")  # Fired when the data channel handshake completes and it's safe to send messages
            def on_open() -> None:
                asyncio.ensure_future(self._on_dc_open())  # Schedules the "ready" message + welcome TTS

            @channel.on("message")  # Fired each time the browser sends a JSON message (start/stop/interrupt/etc.)
            def on_message(message: str) -> None:
                asyncio.ensure_future(self._handle_dc_message(message))  # Route the message asynchronously

        @self.pc.on("connectionstatechange")  # Fired when ICE/DTLS connection state transitions
        async def on_state_change() -> None:
            state = self.pc.connectionState  # One of: "new", "connecting", "connected", "disconnected", "failed", "closed"
            logger.info(
                "session_id={} event=rtc_state_change state={}", self.session_id, state
            )
            if state in ("failed", "closed", "disconnected"):  # Terminal states — clean up resources
                await self._cleanup()

    async def setup(self, offer_sdp: str, offer_type: str) -> RTCSessionDescription:
        """Complete the SDP offer/answer exchange.

        Args:
            offer_sdp: SDP string from the browser offer.
            offer_type: SDP type, typically ``"offer"``.

        Returns:
            The local ``RTCSessionDescription`` answer to send back to the browser.
        """
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer_sdp, type=offer_type)  # Tell aiortc what the browser wants (codecs, IPs, ports)
        )
        answer = await self.pc.createAnswer()       # Generate our matching SDP answer (what WE offer back)
        await self.pc.setLocalDescription(answer)  # Apply the answer locally; triggers ICE candidate gathering
        return self.pc.localDescription            # Return the finalised SDP answer to the HTTP handler so it can send it to the browser

    # ── Data-channel open / message ──────────────────────────────────────────

    async def _on_dc_open(self) -> None:
        """Send the ``ready`` signal and start the welcome message once the data channel opens."""
        logger.info("session_id={} event=dc_open", self.session_id)
        await self._send_json({"type": "ready", "request_id": self.session_id})  # Tell the browser "I'm ready, here's your session ID"
        asyncio.ensure_future(self._run_welcome())  # Start welcome TTS in background — don't block the data channel open handler

    async def _handle_dc_message(self, raw: str) -> None:
        """Dispatch an incoming JSON data-channel message to the appropriate handler.

        Supported event types:
        - ``start``: client has begun recording (informational log only).
        - ``interrupt``: cancel any in-flight LLM/TTS.
        - ``stop``: recording ended; run final STT and schedule LLM if transcript changed.

        Args:
            raw: JSON-encoded message string received from the browser.
        """
        try:
            payload = json.loads(raw)  # Parse JSON text → Python dict
        except (json.JSONDecodeError, TypeError):
            return  # Silently ignore malformed messages

        event_type = payload.get("type")  # e.g. "start", "stop", "interrupt", "tts_voice", "tts_speed"

        if event_type == "tts_voice":
            # Browser is changing the voice mid-session (user picked a different voice from the UI)
            self._voice_id = str(payload.get("voice", self._settings.tts_kokoro_voice))
            logger.info("session_id={} event=tts_voice_changed voice={}", self.session_id, self._voice_id)

        elif event_type == "tts_speed":
            try:
                speed = float(payload.get("speed", self._settings.tts_kokoro_speed))
                self._tts_speed = max(0.5, min(2.0, speed))  # Clamp to [0.5, 2.0] — reject extreme values
            except (TypeError, ValueError):
                pass  # Ignore if the speed value is not a valid number
            logger.info("session_id={} event=tts_speed_changed speed={}", self.session_id, self._tts_speed)

        elif event_type == "start":
            # sample_rate from client is informational only — we always resample to 16 kHz
            self._voice_id = payload.get("voice", self._settings.tts_kokoro_voice)  # Capture initial voice preference
            logger.info("session_id={} event=stream_started voice={}", self.session_id, self._voice_id)

        elif event_type == "interrupt":
            logger.info("session_id={} event=interrupt_received", self.session_id)
            await self._handle_interrupt()  # Cancel all in-flight tasks, clear buffers

        elif event_type == "stop":
            # Cancel silence debounce so it doesn't double-fire after the final transcript.
            if self._silence_debounce_task and not self._silence_debounce_task.done():
                self._silence_debounce_task.cancel()  # Tell the task to stop
                with suppress(asyncio.CancelledError):
                    await self._silence_debounce_task  # Wait for it to actually finish (suppress the CancelledError it raises)
                self._silence_debounce_task = None  # Clear the reference
            if self._speech_finalization_task and not self._speech_finalization_task.done():
                self._speech_finalization_task.cancel()  # Same pattern — cancel and await
                with suppress(asyncio.CancelledError):
                    await self._speech_finalization_task
                self._speech_finalization_task = None

            if self._llm_task and not self._llm_task.done():
                # LLM is already running — skip the final STT to avoid a race condition where
                # the "stop" transcript overwrites the turn the LLM is already answering
                logger.info(
                    "session_id={} event=stop_skipped_final_stt reason=llm_in_flight",
                    self.session_id,
                )
                return

            if self._pcm_buffer:
                loop = asyncio.get_event_loop()  # Get the currently running event loop
                async with self._transcribe_lock:  # Acquire the mutex so STT doesn't run concurrently
                    result = await loop.run_in_executor(None, self._transcribe_buffer)  # Run blocking STT in a thread pool
                await self._send_json({"type": "final", **result})  # Send final transcript to browser
                final_text = str(result.get("text", "")).strip()
                if final_text and final_text != self._latest_llm_input:  # Only call LLM if we have new text
                    self._schedule_llm(final_text)

    # ── RTP audio consumer ────────────────────────────────────────────────────

    async def _consume_audio(self, track: MediaStreamTrack) -> None:
        """Continuously drain RTP frames from the peer track.

        Opus frames are decoded by aiortc and arrive as ``av.AudioFrame`` at 48 kHz.
        Each frame is resampled to 16 kHz mono (Whisper's expected rate) and
        appended to ``self._pcm_buffer``.

        On each frame, server-side VAD (if enabled) runs to detect speech start/end
        events for barge-in detection.  When VAD is disabled, a simple RMS energy
        gate is used as a fallback barge-in trigger.

        After accumulating each frame, ``_maybe_emit_stt`` is called to emit a
        partial STT result when buffer and time thresholds are met.

        Args:
            track: The incoming audio ``MediaStreamTrack`` from the peer connection.
        """
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16_000)
        # av.AudioResampler converts from the browser's 48 kHz stereo Opus output
        # to 16 kHz mono PCM-16 ("s16") which faster-whisper expects
        logger.info("session_id={} event=audio_consumer_started", self.session_id)

        while not self._closed:  # Keep looping until the session is torn down
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                # track.recv() is an async call that suspends this coroutine until
                # the next Opus audio frame arrives over RTP from the browser.
                # wait_for() adds a 5-second timeout so we don't hang forever if
                # the browser freezes or network drops — we just try again (continue).
            except asyncio.TimeoutError:
                continue  # No audio in 5 s — try again; the browser might resume
            except (MediaStreamError, asyncio.CancelledError):
                break  # Track closed by browser or this task was cancelled — exit loop
            except Exception as exc:
                logger.warning(
                    "session_id={} event=audio_recv_error error={}", self.session_id, exc
                )
                break  # Unexpected error — exit loop safely

            for resampled in resampler.resample(frame):
                # resampler.resample() may return 0 or more resampled frames
                # depending on how the input frame size aligns with the output rate
                pcm = resampled.to_ndarray().tobytes()  # Convert av.AudioFrame → raw bytes (int16 PCM)
                self._pcm_buffer.extend(pcm)  # Append to the growing buffer; STT will consume this later
                self._chunk_count += 1         # Track how many frames we've accumulated (debug info)

                if self._vad_stream is not None:
                    for vad_event in self._vad_stream.process_pcm16(pcm):
                        # process_pcm16() runs Silero VAD on this frame and may emit "start" or "end" events
                        if vad_event.event == "start":
                            logger.info(
                                "session_id={} event=vad_speech_start sample_index={} speech_prob={}",
                                self.session_id,
                                vad_event.sample_index,
                                round(vad_event.speech_prob, 4),
                            )
                            if self._silence_debounce_task and not self._silence_debounce_task.done():
                                # User started speaking again — cancel the "wait for silence" timer
                                # so we don't fire the LLM prematurely mid-utterance
                                self._silence_debounce_task.cancel()
                            self._silence_debounce_task = None
                            if self._is_agent_speaking:
                                # User spoke while AI was talking → barge-in!
                                logger.info(
                                    "session_id={} event=server_vad_barge_in sample_index={} speech_prob={}",
                                    self.session_id,
                                    vad_event.sample_index,
                                    round(vad_event.speech_prob, 4),
                                )
                                asyncio.ensure_future(self._handle_interrupt())  # Cancel TTS + LLM asynchronously
                        elif vad_event.event == "end":
                            logger.info(
                                "session_id={} event=vad_speech_end sample_index={} speech_prob={}",
                                self.session_id,
                                vad_event.sample_index,
                                round(vad_event.speech_prob, 4),
                            )
                            if not self._is_agent_speaking:
                                # User stopped speaking and the AI is not currently talking
                                # → begin the turn-end sequence
                                if self._settings.stream_smart_turn_enabled:
                                    # Route through debounce so Smart Turn can gate
                                    # on semantic completeness before finalising.
                                    if self._silence_debounce_task and not self._silence_debounce_task.done():
                                        self._silence_debounce_task.cancel()  # Reset any existing timer
#                                       Debounce here is a mechanism to wait for the user to finish speaking before processing their input. Let me explain with a clear analogy:
                                        # Analogy: The Elevator Door
                                        # Imagine an elevator door that needs to close:
                                        # Without debounce: The door starts closing the instant someone steps away. But if they run back at the last moment, it crashes into them.
                                        # With debounce: The door waits 2 seconds after the last person moves away. If someone else runs toward it during those 2 seconds, the timer resets. Only when 2 full seconds pass without any motion does the door actually close.
                                        
                                        # How Debounce Works in NeuroTalk
                                        # In this code, debounce waits for the user to stop speaking before triggering the LLM:
                                        # User speaks:        "How are you..."
                                        #                           ↓
                                        #                     [Start debounce timer]
                                        #                     (wait 500ms for silence)
                                        #                           ↓
                                        # User keeps speaking: "...doing today?"
                                        #                           ↓
                                        #                     [Cancel & reset timer]
                                        #                     (start fresh 500ms countdown)
                                        #                           ↓
                                        # User stops speaking
                                        #                           ↓
                                        #                     [Wait 500ms completes]
                                        #                           ↓
                                        #                     ✓ Fire LLM call
                                    self._silence_debounce_task = asyncio.create_task(
                                        self._silence_debounce_then_fire(
                                            self._last_text_sent, "vad_end", vad_triggered=True
                                            # vad_triggered=True → use 50 ms grace wait instead of full silence timeout
                                        )
                                    )  # schedule the debounce/finalization logic as a background task without blocking the current audio handling loop
                                else:
                                    # Smart Turn disabled → finalize immediately after VAD end
                                    self._schedule_speech_finalization("vad_end")
                elif self._is_agent_speaking:
                    # Fallback RMS gate when the dedicated VAD is disabled.
                    samples = resampled.to_ndarray().astype(np.float32) / 32_768.0
                    # Convert int16 samples to float32 in the range [-1.0, +1.0]
                    # 32_768 = 2^15, the max value of a signed 16-bit integer
                    rms = float(np.sqrt(np.mean(samples ** 2)))
                    # Root Mean Square energy — a simple loudness measure
                    if rms > _BARGE_IN_THRESHOLD:
                        self._barge_in_count += 1  # Increment consecutive loud-frame counter
                        if self._barge_in_count >= _BARGE_IN_FRAMES:
                            # 3+ consecutive loud frames → treat as a barge-in attempt
                            logger.info(
                                "session_id={} event=server_vad_barge_in rms={}",
                                self.session_id,
                                round(rms, 4),
                            )
                            asyncio.ensure_future(self._handle_interrupt())
                    else:
                        self._barge_in_count = 0  # Quiet frame — reset the counter; requires 3 in a row

                await self._maybe_emit_stt()  # Check if buffer + time thresholds are met to emit a partial transcript

        logger.info("session_id={} event=audio_consumer_stopped", self.session_id)

    async def _maybe_emit_stt(self) -> None:
        """Emit a partial STT result when the buffer and time-interval thresholds are met.

        Skips emission if a speech-finalization task or LLM task is already in
        flight to avoid unnecessary transcription during active processing.
        On new text, sends a ``partial`` message and (re)starts the silence
        debounce timer.
        """
        if not self._pcm_buffer:
            return  # Nothing accumulated yet — nothing to transcribe
        if self._speech_finalization_task and not self._speech_finalization_task.done():
            self._last_emit_at = perf_counter()  # Reset timer so we don't emit immediately after finalization
            return  # Turn finalization is already in progress — skip partial to avoid race
        if self._llm_task and not self._llm_task.done():
            self._last_emit_at = perf_counter()  # Same — LLM is running, no point emitting a new partial
            return

        buffered_ms = len(self._pcm_buffer) / 2 / self._sample_rate * 1000
        # len(buffer) is in bytes; divide by 2 because each PCM-16 sample is 2 bytes
        # divide by sample_rate to get seconds, multiply by 1000 for milliseconds
        now = perf_counter()  # Current timestamp in seconds (high-resolution)
        if not (
            buffered_ms >= self._settings.stream_min_audio_ms  # Enough audio buffered (e.g. 500 ms minimum)
            and (now - self._last_emit_at) * 1000 >= self._settings.stream_emit_interval_ms  # Enough time since last emit (e.g. 700 ms)
        ):
            return  # Throttle: don't emit too frequently or with too little data

        loop = asyncio.get_event_loop()  # Get the running event loop so we can submit work to its thread pool
        async with self._transcribe_lock:  # Take the STT mutex — only one transcription runs at a time
            result = await loop.run_in_executor(None, self._transcribe_buffer)
            # run_in_executor() runs _transcribe_buffer() in a thread pool (not the event loop)
            # so the blocking denoise+Whisper CPU work doesn't freeze the async server
        if self._speech_finalization_task and not self._speech_finalization_task.done():
            self._last_emit_at = perf_counter()
            logger.info(
                "session_id={} event=partial_stt_skipped reason=turn_finalizing",
                self.session_id,
            )
            return  # Finalization started while we were transcribing — discard this partial result
        if self._llm_task and not self._llm_task.done():
            self._last_emit_at = perf_counter()
            logger.info(
                "session_id={} event=partial_stt_skipped reason=llm_in_flight",
                self.session_id,
            )
            return  # LLM started while we were transcribing — discard this partial result

        current_text = str(result.get("text", ""))
        if current_text != self._last_text_sent:
            # Only send if the transcript actually changed since last time
            await self._send_json({"type": "partial", **result})  # Send live transcript to browser
            self._last_text_sent = current_text  # Update dedup guard

            if self._silence_debounce_task and not self._silence_debounce_task.done():
                self._silence_debounce_task.cancel()  # Cancel old debounce timer; we'll start a fresh one
            self._silence_debounce_task = asyncio.create_task(
                self._silence_debounce_then_fire(current_text, "debounced_partial")
                # Start the silence countdown: if no new text arrives in stream_llm_silence_ms, fire LLM
            )
        self._last_emit_at = perf_counter()  # Record when we last emitted (for throttling next call)

    def _transcribe_buffer(self) -> dict:
        """Write ``self._pcm_buffer`` to a temp WAV file and run faster-whisper.

        Serialises the current PCM buffer to a temporary WAV, runs the STT
        service, annotates timing fields, then deletes the temp file.

        Returns:
            Dict with keys ``text`` (str), ``timings_ms`` (dict), ``debug`` (dict).
        """
        started_at = perf_counter()  # Start timing for total transcription latency
        temp_path = self._settings.temp_dir / f"{self.session_id}_rtc.wav"
        # Build a unique temp file path in the configured temp directory
        # Using session_id avoids collisions if multiple sessions run simultaneously
        try:
            pcm = get_denoise_service().enhance(bytes(self._pcm_buffer), self._sample_rate)
            # Run DeepFilterNet3 on the raw PCM bytes to remove background noise
            # enhance() returns cleaned PCM bytes at the same sample rate
            with wave.open(str(temp_path), "wb") as wf:
                # Write the denoised PCM into a .wav file that Whisper's C++ backend can read
                wf.setnchannels(1)               # Mono audio (1 channel)
                wf.setsampwidth(2)               # 2 bytes per sample = PCM-16 (int16)
                wf.setframerate(self._sample_rate)  # 16,000 samples per second
                wf.writeframes(pcm)              # Write the actual audio data

            service = get_stt_service()  # Get the singleton SpeechToTextService (loads Whisper model once)
            result = service.transcribe(
                file_path=temp_path,          # Path to the temp WAV file we just wrote
                request_id=self.session_id,   # Passed through for logging correlation
                filename=f"rtc_{self.session_id}.wav",  # Descriptive filename for logs
                audio_bytes=len(self._pcm_buffer),       # Total audio size for debug info
            )
            result.timings_ms.buffered_audio_ms = round(
                len(self._pcm_buffer) / 2 / self._sample_rate * 1000, 2
            )  # Annotate how much audio (in ms) was in the buffer when we transcribed
            result.timings_ms.total_ms = round((perf_counter() - started_at) * 1000, 2)
            # Annotate total time from function entry to transcription complete (ms)
            result.debug.sample_rate = self._sample_rate  # Record the actual sample rate used
            result.debug.chunks_received = self._chunk_count  # Record how many RTP frames accumulated
            return {
                "text": result.text,                        # The transcribed string (e.g. "How are you today?")
                "timings_ms": result.timings_ms.model_dump(),  # Latency breakdown dict
                "debug": result.debug.model_dump(),            # Debug info dict
            }
        finally:
            temp_path.unlink(missing_ok=True)  # Always delete the temp WAV — even if transcription raised an exception

    def _schedule_speech_finalization(self, trigger: str) -> None:
        """Create a speech-finalization task if one is not already running.

        A no-op if ``_speech_finalization_task`` is still active, preventing
        double-firing on rapid VAD end events.

        Args:
            trigger: Label passed through to ``_finalize_speech_turn`` for logging
                     (e.g. ``"vad_end"`` or ``"debounced_partial"``).
        """
        if self._speech_finalization_task and not self._speech_finalization_task.done():
            return  # Already finalizing this turn — don't start a second one
        self._speech_finalization_task = asyncio.create_task(
            self._finalize_speech_turn(trigger)
            # Schedule _finalize_speech_turn as a background task — it will run STT + LLM
        )

    async def _finalize_speech_turn(self, trigger: str) -> None:
        """Transcribe the completed utterance and schedule an LLM call.

        Guards against finalising while the agent is speaking or while an LLM
        is already in flight.  Cancels any pending silence-debounce before
        transcribing to avoid a race with the debounce firing independently.

        Args:
            trigger: Source label for logging (e.g. ``"vad_end"``).
        """
        try:
            if not self._pcm_buffer:
                return  # Nothing to transcribe — user may have triggered stop on an empty buffer
            if self._is_agent_speaking:
                logger.info(
                    "session_id={} event=speech_finalization_skipped reason=agent_speaking",
                    self.session_id,
                )
                return  # Don't finalize while the AI is mid-sentence; barge-in handler covers that path
            if self._llm_task and not self._llm_task.done():
                logger.info(
                    "session_id={} event=speech_finalization_skipped reason=llm_in_flight",
                    self.session_id,
                )
                return  # LLM already running for this utterance — no need to re-submit

            if self._silence_debounce_task and not self._silence_debounce_task.done():
                self._silence_debounce_task.cancel()  # Stop the periodic partial-emit timer before final STT
                with suppress(asyncio.CancelledError):
                    await self._silence_debounce_task  # Wait for it to finish gracefully
                self._silence_debounce_task = None

            loop = asyncio.get_event_loop()
            async with self._transcribe_lock:  # Acquire STT mutex — blocks if _maybe_emit_stt is mid-transcription
                result = await loop.run_in_executor(None, self._transcribe_buffer)
                # Run blocking Whisper STT in a thread pool (same as _maybe_emit_stt)

            if self._llm_task and not self._llm_task.done():
                logger.info(
                    "session_id={} event=speech_finalization_skipped reason=llm_started_during_transcribe",
                    self.session_id,
                )
                return  # LLM launched by something else while we were transcribing — skip

            final_text = str(result.get("text", "")).strip()  # The complete utterance text
            if not final_text:
                logger.info(
                    "session_id={} event=speech_finalization_empty trigger={}",
                    self.session_id,
                    trigger,
                )
                return  # Whisper returned empty — user may have been silent or only made noise

            self._last_text_sent = final_text  # Update dedup guard to suppress re-sending same text as partial
            await self._send_json({"type": "final", **result})  # Tell browser the final transcript
            if final_text != self._latest_llm_input:  # Don't call LLM twice for the same question
                self._schedule_llm(final_text)
        finally:
            self._speech_finalization_task = None  # Always clear the task reference on exit (success or exception)

    # ── Interrupt ─────────────────────────────────────────────────────────────

    async def _handle_interrupt(self) -> None:
        """Cancel all in-flight tasks and reset audio/barge-in state.

        Sets the interrupt event so streaming coroutines exit cleanly, then
        cancels (in order) TTS and LLM tasks to prevent stale audio being
        sent after the interrupt.
        """
        self._interrupt_event.set()  # Signal to _run_llm and _tts_sentence_pipeline to stop on their next iteration check
        self._pending_llm_call = None  # Drop any queued LLM call — user is speaking again
        self._is_agent_speaking = False  # Mark agent as silent so future VAD events don't re-trigger barge-in
        self._barge_in_count = 0  # Reset consecutive-loud-frame counter
        self._pcm_buffer.clear()  # Discard audio accumulated while the AI was speaking (it was the AI's voice)
        self._last_text_sent = ""  # Reset dedup guard so next user utterance registers as new
        self._chunk_count = 0     # Reset frame counter
        self._last_emit_at = 0.0  # Reset throttle timer so next frame can emit immediately

        if self._vad_stream is not None:
            self._vad_stream.reset()  # Reset Silero VAD internal state — clears speech probability history

        if self._silence_debounce_task and not self._silence_debounce_task.done():
            self._silence_debounce_task.cancel()  # Cancel any pending "fire LLM after silence" timer
            self._silence_debounce_task = None

        if self._speech_finalization_task and not self._speech_finalization_task.done():
            self._speech_finalization_task.cancel()  # Cancel any in-progress final STT
            self._speech_finalization_task = None

        # Cancel TTS pipeline first so it doesn't send more audio after the new
        # _run_llm clears interrupt_event.
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()  # Stop TTS sentence pipeline immediately
            with suppress(asyncio.CancelledError):
                await self._tts_task  # Wait for it to exit gracefully before proceeding
            self._tts_task = None

        if self._llm_task and not self._llm_task.done():
            self._llm_task.cancel()  # Cancel the LLM streaming task
            with suppress(asyncio.CancelledError):
                await self._llm_task  # Wait for it to exit before we return
            self._llm_task = None

    # ── LLM scheduling ───────────────────────────────────────────────────────

    async def _silence_debounce_then_fire(
        self, text: str, trigger: str, vad_triggered: bool = False
    ) -> None:
        """Wait for the silence window, then trigger speech finalization or an LLM call.

        After the silence timeout, optionally polls the Smart Turn model for up
        to ``stream_smart_turn_max_budget_ms`` before handing off.  Cancelled if
        new audio or a VAD event resets the debounce timer before the sleep completes.

        Args:
            text: Transcript snapshot at the time the debounce was armed.
            trigger: Source label forwarded to ``_schedule_speech_finalization``
                     or used for logging (e.g. ``"debounced_partial"``).
            vad_triggered: When True (VAD "end" path), uses a 50 ms grace wait
                           instead of the full silence timeout so Smart Turn runs
                           immediately after VAD has already confirmed silence.
        """
        try:
            wait_ms = 50 if vad_triggered else self._settings.stream_llm_silence_ms
            # If VAD already confirmed silence (vad_triggered=True), only wait 50 ms
            # Otherwise wait the full configured silence timeout (e.g. 500 ms) before assuming the user is done
            await asyncio.sleep(wait_ms / 1000)
            # asyncio.sleep() suspends this coroutine without blocking other tasks
            # If this task is cancelled during the sleep, CancelledError is raised here
        except asyncio.CancelledError:
            return  # New audio arrived and reset the debounce — exit cleanly without firing

        if self._settings.stream_smart_turn_enabled:
            from app.services.smart_turn import get_smart_turn_service  # Lazy import to avoid circular deps at startup
            smart_turn = get_smart_turn_service()
            if smart_turn.is_loaded:
                turn_complete = False
                deadline = perf_counter() + self._settings.stream_smart_turn_max_budget_ms / 1000
                # Calculate the absolute time by which we must stop polling Smart Turn
                while perf_counter() < deadline:
                    is_complete, _ = smart_turn.predict(bytes(self._pcm_buffer))
                    # Ask the ONNX model: "Does this audio sound like a complete utterance?"
                    if is_complete:
                        turn_complete = True
                        break  # Smart Turn says yes → proceed to LLM immediately
                    try:
                        await asyncio.sleep(self._settings.stream_smart_turn_base_wait_ms / 1000)
                        # Wait a short interval before polling Smart Turn again
                    except asyncio.CancelledError:
                        return  # Cancelled during Smart Turn polling — user spoke again

                if not turn_complete:
                    # Smart Turn says utterance is incomplete — give the user
                    # extra time to continue their thought.  VAD "start" will
                    # cancel this task the moment they resume speaking.
                    try:
                        await asyncio.sleep(
                            self._settings.stream_smart_turn_incomplete_wait_ms / 1000
                            # Extra patience window when Smart Turn is unsure
                        )
                    except asyncio.CancelledError:
                        return  # User started speaking again during the extra wait — abort

        if self._vad_stream is not None:
            # VAD-enabled path: finalize the full speech turn (run STT on the complete buffer)
            self._schedule_speech_finalization(trigger)
            return
        # VAD-disabled path (no stream VAD): go directly to LLM with the last partial text
        self._schedule_llm(text)

    def _schedule_llm(self, text: str) -> None:
        """Gate and enqueue an LLM call, cancelling any stale in-flight call.

        Ignores short transcripts, duplicate inputs, and pause commands.
        If an LLM is already running for older text, it is cancelled so the
        newer, more complete utterance takes priority.

        Args:
            text: Transcript candidate to send to the LLM.
        """
        normalized = text.strip()  # Remove leading/trailing whitespace for consistent comparison
        if len(normalized) < self._settings.stream_llm_min_chars:
            return  # Too short (e.g. stray noise → "uh") — not worth sending to the LLM
        if normalized == self._latest_llm_input:
            return  # Same question the LLM is already answering — skip duplicate
        if _PAUSE_PATTERN.match(normalized):
            return  # User said "hold on" or "wait" — don't treat it as a real question

        if self._llm_task and not self._llm_task.done():
            # A previous LLM call is still running (user extended their utterance)
            # Cancel the old one so the new, more complete text takes priority
            logger.info(
                "session_id={} event=llm_cancel_for_newer_text", self.session_id
            )
            self._interrupt_event.set()  # Signal streaming coroutines to stop on next check
            self._pending_llm_call = None  # Clear any previously queued call
            if self._tts_task and not self._tts_task.done():
                self._tts_task.cancel()  # Cancel TTS that was running for the stale LLM response
                self._tts_task = None
            self._llm_task.cancel()  # Cancel the stale LLM call
            self._llm_task = None   # Clear the reference so the task below can set a new one

        self._pending_llm_call = normalized  # Tentatively queue this text
        if self._llm_task is None or self._llm_task.done():
            # No LLM running (either we just cancelled it above, or none was running)
            next_text = self._pending_llm_call  # Grab the queued text
            self._pending_llm_call = None        # Clear the queue — it's being consumed now
            self._llm_seq += 1                   # Increment sequence number for this new call
            self._llm_task = asyncio.create_task(self._run_llm(self._llm_seq, next_text))
            # create_task() schedules _run_llm() as a background task — returns immediately

    # ── LLM + TTS pipeline ───────────────────────────────────────────────────

    async def _tts_sentence_pipeline(
        self, llm_seq: int, queue: asyncio.Queue, enable_barge_in: bool = True
    ) -> None:
        """Consume sentences from *queue* and synthesise + stream each one immediately.

        Runs concurrently with ``_run_llm`` so the agent starts speaking after
        the first sentence boundary — not after the full LLM response completes.
        Sends ``tts_start`` before the first audio chunk and ``tts_done`` (or
        ``tts_interrupted``) when the queue is exhausted or interrupted.

        Args:
            queue: Asyncio queue of plain-text sentences terminated by a ``None``
                   sentinel signalling end-of-stream.
            enable_barge_in: When ``False``, ``_is_agent_speaking`` is not set —
                             used for the welcome message to prevent ambient noise
                             from triggering an interrupt before the user speaks.
        """
        tts_service = get_tts_service()  # Get the singleton TTSService (Kokoro or Chatterbox)
        tts_started = False  # Track whether we've sent the "tts_start" message yet
        if enable_barge_in:
            self._is_agent_speaking = True  # Tell the VAD/RMS path that the agent is now speaking

        while True:
            sentence = await queue.get()  # Suspend here until _run_llm puts a sentence into the queue
            if sentence is None or self._interrupt_event.is_set():
                break  # None = end-of-stream sentinel; interrupt_event = user barged in
            try:
                wav_bytes, sr = await tts_service.synthesize(sentence, voice=self._voice_id, speed=self._tts_speed)
                # synthesize() runs the TTS model (GPU/CPU) and returns:
                #   wav_bytes: raw WAV file bytes
                #   sr: sample rate of the audio (e.g. 24000 Hz for Kokoro)
            except Exception as err:
                logger.warning(
                    "session_id={} event=tts_error error={}", self.session_id, err
                )
                continue  # Skip this sentence if TTS fails — don't crash the whole pipeline
            if self._interrupt_event.is_set():
                break  # Check interrupt again after the blocking synthesize call

            if not tts_started:
                tts_started = True
                tts_t0 = perf_counter()  # Start timing from first TTS output
                await self._send_json({"type": "tts_start", "llm_seq": llm_seq})
                # Tell the browser "audio is about to start arriving"
            tts_ms = round((perf_counter() - tts_t0) * 1000, 2)  # Time from tts_start to this chunk
            wav_b64 = base64.b64encode(wav_bytes).decode()
            # base64 encoding converts binary WAV bytes into a text string safe to embed in JSON
            await self._send_json({
                "type": "tts_audio",
                "llm_seq": llm_seq,        # Sequence number so browser can discard stale chunks
                "data": wav_b64,           # The actual audio (base64-encoded WAV)
                "sample_rate": sr,         # Browser needs this to create the correct AudioBuffer
                "tts_ms": tts_ms,          # Latency metric
                "sentence_text": sentence, # The text this audio corresponds to (for progressive reveal)
            })

        self._is_agent_speaking = False  # Agent finished speaking (or was interrupted)
        if self._interrupt_event.is_set():
            await self._send_json({"type": "tts_interrupted", "llm_seq": llm_seq})
            # Tell browser we stopped early — it should discard any queued audio
        else:
            await self._send_json({"type": "tts_done", "llm_seq": llm_seq})
            # Normal completion — browser knows all audio chunks have arrived

    async def _run_llm(self, llm_seq: int, text: str) -> None:
        """Run one full LLM inference turn and stream TTS audio sentence-by-sentence.

        Clears the interrupt flag, streams tokens from the LLM, extracts sentence
        boundaries to feed a concurrent TTS pipeline, and appends the completed
        turn to conversation history on success.  Queues the next pending LLM
        call (if any) once this one finishes.

        Args:
            text: User transcript to send as the current user message.
        """
        self._interrupt_event.clear()  # Reset the interrupt flag — this is a fresh LLM call
        self._latest_llm_input = text  # Record what we're sending so dedup guards work correctly

        full_response = ""      # Accumulates the complete LLM response token by token
        processed_chars = 0     # How many characters of full_response have already been sent to TTS
        llm_t0 = perf_counter() # Start timer for total LLM latency measurement
        call_error: str | None = None  # Captures error message if LLM fails

        sent_queue: asyncio.Queue[str | None] = asyncio.Queue()
        # asyncio.Queue is a thread-safe FIFO channel between two async coroutines
        # _run_llm puts sentences in; _tts_sentence_pipeline takes them out
        tts_task = asyncio.create_task(self._tts_sentence_pipeline(llm_seq, sent_queue))
        # Start TTS pipeline immediately as a concurrent task — it waits on the queue
        # This means TTS starts the moment the first sentence is complete, not after full LLM response
        self._tts_task = tts_task  # Save reference so _handle_interrupt() can cancel it

        try:
            await self._send_json({"type": "llm_start", "llm_seq": llm_seq, "user_text": text})
            # Tell browser "LLM is starting now with this user text"
            async for token in stream_llm_response(
                text, conversation_history=list(self._conversation_history)
            ):
                # stream_llm_response() is an async generator — it yields one token at a time
                # as the LLM produces them. "async for" awaits each token without blocking others.
                if self._interrupt_event.is_set():
                    break  # User barged in — stop consuming tokens immediately

                full_response += token  # Append this token to the growing response
                await self._send_json(
                    {"type": "llm_partial", "llm_seq": llm_seq, "text": strip_emotion_tags(full_response)}
                )
                # Send the running text to the browser for live display (streaming typewriter effect)
                # strip_emotion_tags() removes [happy], [sad] etc. markers before displaying

                # Extract complete sentences and enqueue for immediate TTS.
                tail = full_response[processed_chars:]
                # tail = the part of the response we haven't yet sent to TTS
                while True:
                    m = _SENT_BOUNDARY.search(tail)  # Look for . ! ? in the tail
                    if not m or m.end() < _MIN_SENTENCE_CHARS:
                        break  # No sentence boundary found, or the sentence is too short
                    sentence = clean_for_tts(tail[: m.end()].strip())
                    # Extract text up to and including the punctuation, clean emotion tags
                    if sentence:
                        await sent_queue.put(sentence)  # Hand this sentence to the TTS pipeline
                    processed_chars += m.end()  # Advance the "already processed" cursor
                    tail = full_response[processed_chars:]  # Update tail to remaining text

            llm_ms = round((perf_counter() - llm_t0) * 1000, 2)  # Total LLM time in ms
            display_text = strip_emotion_tags(full_response)  # Clean version for browser display
            await self._send_json(
                {"type": "llm_final", "llm_seq": llm_seq, "text": display_text, "llm_ms": llm_ms}
            )
            # Tell browser the LLM is done and here's the full response + latency

        except Exception as err:
            call_error = str(err)  # Capture error to prevent conversation history save
            logger.warning(
                "session_id={} event=llm_error error={}", self.session_id, err
            )
            await self._send_json(
                {
                    "type": "llm_error",
                    "llm_seq": llm_seq,
                    "message": "LLM unavailable — check your LLM provider and model.",
                }
            )
            # Inform browser that LLM failed (e.g. model not loaded, API error)

        finally:
            # "finally" always runs — even if the task was cancelled or an exception occurred
            if not self._interrupt_event.is_set() and call_error is None:
                remaining = clean_for_tts(full_response[processed_chars:].strip())
                # Any text after the last sentence boundary that hasn't been sent to TTS yet
                if remaining:
                    sent_queue.put_nowait(remaining)
                    # put_nowait() is the non-blocking version of await queue.put()
                    # Safe here because the queue is unbounded and we're in finally
            sent_queue.put_nowait(None)
            # The None sentinel tells _tts_sentence_pipeline() "no more sentences coming"

            # Save the exchange to history inside finally so it runs even when
            # the task is cancelled mid-stream. Without this, interrupted responses
            # are missing from history and the LLM re-answers the same question
            # when the user says "All right." or similar after a barge-in.
            if full_response and call_error is None:
                self._conversation_history.append({"role": "user", "content": text})
                # Add the user's question to conversation history
                self._conversation_history.append({"role": "assistant", "content": full_response})
                # Add the AI's (possibly partial) response to history
                max_msgs = self._settings.llm_max_history_turns * 2
                # Each "turn" = 1 user message + 1 assistant message = 2 entries
                if len(self._conversation_history) > max_msgs:
                    self._conversation_history[:] = self._conversation_history[-max_msgs:]
                    # Sliding window: keep only the most recent N turns to avoid growing the context unboundedly

        with suppress(asyncio.CancelledError):
            await tts_task  # Wait for TTS pipeline to drain and send tts_done before returning
        self._tts_task = None  # Clear reference — this LLM turn is fully complete

        # Clear audio buffer after the full turn so any speech the user started
        # during the agent's response isn't fed into the next query.
        if not self._interrupt_event.is_set():
            self._pcm_buffer.clear()    # Discard any audio captured while the AI was speaking
            self._last_text_sent = ""   # Reset partial transcript dedup guard
            self._chunk_count = 0       # Reset frame counter
            self._last_emit_at = 0.0    # Reset throttle timer
            if self._vad_stream is not None:
                self._vad_stream.reset()  # Reset Silero VAD state for the next turn

        if self._pending_llm_call:
            # While we were running, _schedule_llm() was called with new text
            # (user spoke again immediately after the AI finished)
            next_text = self._pending_llm_call
            self._pending_llm_call = None
            if next_text != self._latest_llm_input:
                # Only fire if it's genuinely new text (not a repeat)
                self._llm_seq += 1  # New sequence number for the follow-up call
                self._llm_task = asyncio.create_task(self._run_llm(self._llm_seq, next_text))
                return  # Return here — self._llm_task is already set to the new task
        self._llm_task = None  # No pending call — this session is back to "listening"

    # ── Welcome message ──────────────────────────────────────────────────────

    async def _run_welcome(self) -> None:
        """Synthesise and stream the configured welcome message at session open.

        Splits ``settings.welcome_message`` on sentence boundaries so the
        frontend receives individual ``tts_audio`` events with ``sentence_text``,
        matching the format produced by the regular LLM pipeline.  Skipped if
        the welcome message is empty.  Barge-in is disabled so ambient noise
        does not cancel the greeting before the user speaks.
        """
        welcome = self._settings.welcome_message  # e.g. "Hi! I'm NeuroTalk. How can I help?"
        if not welcome:
            return  # Welcome message disabled in settings — skip

        await self._send_json({"type": "llm_start", "user_text": ""})
        # Mimic the normal llm_start event so the browser creates an assistant message bubble
        await self._send_json({"type": "llm_final", "text": welcome, "llm_ms": 0})
        # Immediately send the full welcome text (it's not generated — it's a fixed string)

        sent_queue: asyncio.Queue[str | None] = asyncio.Queue()
        for raw in re.split(r"(?<=[.!?])\s+", welcome.strip()):
            # Split the welcome message into sentences using regex
            # (?<=[.!?]) is a lookbehind: split AFTER . ! ? (not before)
            sentence = clean_for_tts(raw.strip())
            if sentence:
                sent_queue.put_nowait(sentence)  # Add each sentence to the TTS queue
        sent_queue.put_nowait(None)  # End-of-stream sentinel

        await self._tts_sentence_pipeline(0, sent_queue, enable_barge_in=False)
        # llm_seq=0 for welcome; enable_barge_in=False so ambient room noise doesn't
        # cancel the greeting before the user has had a chance to speak

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _send_json(self, payload: dict) -> None:
        """Send a JSON message via the RTCDataChannel under a mutex.

        Silently drops the message if the channel is not open, preventing
        exceptions when the peer disconnects mid-stream.

        Args:
            payload: Dict to serialise and transmit over the data channel.
        """
        async with self._send_lock:
            # Acquire the mutex so only one coroutine writes to the data channel at a time
            # (multiple tasks — LLM, TTS, partial STT — call _send_json concurrently)
            if self.dc and self.dc.readyState == "open":
                # Check that the data channel exists and is in the "open" state before sending
                try:
                    self.dc.send(json.dumps(payload))
                    # json.dumps() converts the Python dict to a JSON string
                    # self.dc.send() transmits it over the WebRTC data channel to the browser
                except Exception as exc:
                    logger.debug(
                        "session_id={} event=dc_send_error error={}", self.session_id, exc
                    )
                    # Suppress send errors — the channel may have closed mid-send (race condition)

    async def _cleanup(self) -> None:
        """Cancel all background tasks and mark the session as closed.

        Idempotent — subsequent calls after the first are no-ops.
        Called automatically when the peer connection transitions to a terminal
        state (``"failed"``, ``"closed"``, or ``"disconnected"``).
        """
        if self._closed:
            return  # Already cleaned up — idempotent guard prevents double-cancel
        self._closed = True  # Mark as closed before cancelling tasks (prevents re-entry)
        tasks = [t for t in (
            self._audio_task,
            self._silence_debounce_task,
            self._speech_finalization_task,
            self._tts_task,
            self._llm_task,
        ) if t and not t.done()]
        # Collect all running background tasks into a list (skip None and already-finished tasks)
        for task in tasks:
            task.cancel()  # Request cancellation — raises CancelledError inside each task
        # Await cancelled tasks so they finish before ICE/transport teardown.
        # Without this, tasks outlive the transport and hit aioice's closed socket.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            # asyncio.gather() awaits all tasks concurrently
            # return_exceptions=True means CancelledError from each task is returned as a value
            # rather than re-raised here, so all tasks get a chance to clean up
        logger.info("session_id={} event=rtc_session_closed", self.session_id)
