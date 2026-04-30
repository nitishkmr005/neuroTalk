"""WebRTC session: RTP audio → STT → LLM → TTS pipeline over a data channel."""

import asyncio
import base64
import json
import re
import wave
from contextlib import suppress
from time import perf_counter

import av
import numpy as np
from aiortc import MediaStreamTrack, RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from loguru import logger

from app.services.denoise import get_denoise_service
from app.services.llm import stream_llm_response
from app.services.stt import get_stt_service
from app.services.tts import get_tts_service
from app.services.vad import StreamingVAD, get_vad_service
from app.utils.emotion import clean_for_tts, strip_emotion_tags
from config.settings import get_settings

_PAUSE_PATTERN = re.compile(
    r"^\s*(wait|hold on|hold up|one moment|one sec(?:ond)?|just a (?:moment|second|sec)|"
    r"give me a (?:second|moment|sec)|hang on|please wait|just wait|ok wait|okay wait|"
    r"stop|stop it|stop please|please stop|ok stop|okay stop)\s*[.!?,]?\s*$",
    re.IGNORECASE,
)
_SENT_BOUNDARY = re.compile(r"[.!?](?:\s|$)")
_MIN_SENTENCE_CHARS = 15

# Server-side VAD: energy threshold (normalised to [-1, 1]) to trigger barge-in.
# Slightly lower than the client-side 0.15 because there is no browser AGC on the
# raw Opus-decoded frames we see here.
_BARGE_IN_THRESHOLD = 0.15
_BARGE_IN_FRAMES = 3

# No STUN server on the server side: aiortc's setLocalDescription blocks until
# ICE gathering completes, and STUN lookups to stun.l.google.com can stall
# 5+ seconds on restricted networks. For localhost/LAN use, host candidates
# (local IPs) are all that's needed. Add STUN/TURN here for remote deployments.
_RTC_CONFIG = RTCConfiguration(iceServers=[])


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
        self.session_id = session_id
        self._settings = get_settings()
        self.pc = RTCPeerConnection(_RTC_CONFIG)
        self.dc = None

        # Audio accumulation
        self._pcm_buffer = bytearray()
        self._chunk_count = 0
        self._last_emit_at = 0.0
        self._last_text_sent = ""
        self._sample_rate = 16_000  # resampled server-side; client value ignored

        # Concurrency helpers
        self._send_lock = asyncio.Lock()
        self._transcribe_lock = asyncio.Lock()  # serialises denoise+STT; both are CPU-bound
        self._llm_task: asyncio.Task | None = None
        self._tts_task: asyncio.Task | None = None
        self._pending_llm_call: str | None = None
        self._latest_llm_input = ""
        self._interrupt_event = asyncio.Event()
        self._silence_debounce_task: asyncio.Task | None = None
        self._speech_finalization_task: asyncio.Task | None = None

        # Conversation state
        self._conversation_history: list[dict[str, str]] = []
        self._voice_id: str = self._settings.tts_kokoro_voice
        self._tts_speed: float = self._settings.tts_kokoro_speed

        # Barge-in state
        self._is_agent_speaking = False
        self._barge_in_count = 0
        self._vad_stream: StreamingVAD | None = (
            get_vad_service().create_stream() if self._settings.stream_vad_enabled else None
        )

        # Background tasks
        self._audio_task: asyncio.Task | None = None
        self._closed = False

        self._register_pc_handlers()

    # ── Peer-connection lifecycle ────────────────────────────────────────────

    def _register_pc_handlers(self) -> None:
        """Attach aiortc event handlers for track, datachannel, and state changes.

        Registers closures on ``self.pc`` for:
        - ``track``: starts the audio consumer coroutine when an audio track arrives.
        - ``datachannel``: wires up open/message callbacks on the signalling channel.
        - ``connectionstatechange``: triggers cleanup on failure or disconnection.
        """
        @self.pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if track.kind == "audio":
                self._audio_task = asyncio.ensure_future(self._consume_audio(track))

        @self.pc.on("datachannel")
        def on_datachannel(channel) -> None:
            self.dc = channel

            @channel.on("open")
            def on_open() -> None:
                asyncio.ensure_future(self._on_dc_open())

            @channel.on("message")
            def on_message(message: str) -> None:
                asyncio.ensure_future(self._handle_dc_message(message))

        @self.pc.on("connectionstatechange")
        async def on_state_change() -> None:
            state = self.pc.connectionState
            logger.info(
                "session_id={} event=rtc_state_change state={}", self.session_id, state
            )
            if state in ("failed", "closed", "disconnected"):
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
            RTCSessionDescription(sdp=offer_sdp, type=offer_type)
        )
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription

    # ── Data-channel open / message ──────────────────────────────────────────

    async def _on_dc_open(self) -> None:
        """Send the ``ready`` signal and start the welcome message once the data channel opens."""
        logger.info("session_id={} event=dc_open", self.session_id)
        await self._send_json({"type": "ready", "request_id": self.session_id})
        asyncio.ensure_future(self._run_welcome())

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
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        event_type = payload.get("type")

        if event_type == "tts_voice":
            self._voice_id = str(payload.get("voice", self._settings.tts_kokoro_voice))
            logger.info("session_id={} event=tts_voice_changed voice={}", self.session_id, self._voice_id)

        elif event_type == "tts_speed":
            try:
                speed = float(payload.get("speed", self._settings.tts_kokoro_speed))
                self._tts_speed = max(0.5, min(2.0, speed))
            except (TypeError, ValueError):
                pass
            logger.info("session_id={} event=tts_speed_changed speed={}", self.session_id, self._tts_speed)

        elif event_type == "start":
            # sample_rate from client is informational only — we always resample to 16 kHz
            self._voice_id = payload.get("voice", self._settings.tts_kokoro_voice)
            logger.info("session_id={} event=stream_started voice={}", self.session_id, self._voice_id)

        elif event_type == "interrupt":
            logger.info("session_id={} event=interrupt_received", self.session_id)
            await self._handle_interrupt()

        elif event_type == "stop":
            # Cancel silence debounce so it doesn't double-fire after the final transcript.
            if self._silence_debounce_task and not self._silence_debounce_task.done():
                self._silence_debounce_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._silence_debounce_task
                self._silence_debounce_task = None
            if self._speech_finalization_task and not self._speech_finalization_task.done():
                self._speech_finalization_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._speech_finalization_task
                self._speech_finalization_task = None

            if self._llm_task and not self._llm_task.done():
                logger.info(
                    "session_id={} event=stop_skipped_final_stt reason=llm_in_flight",
                    self.session_id,
                )
                return

            if self._pcm_buffer:
                loop = asyncio.get_event_loop()
                async with self._transcribe_lock:
                    result = await loop.run_in_executor(None, self._transcribe_buffer)
                await self._send_json({"type": "final", **result})
                final_text = str(result.get("text", "")).strip()
                if final_text and final_text != self._latest_llm_input:
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
        logger.info("session_id={} event=audio_consumer_started", self.session_id)

        while not self._closed:
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except (MediaStreamError, asyncio.CancelledError):
                break
            except Exception as exc:
                logger.warning(
                    "session_id={} event=audio_recv_error error={}", self.session_id, exc
                )
                break

            for resampled in resampler.resample(frame):
                pcm = resampled.to_ndarray().tobytes()
                self._pcm_buffer.extend(pcm)
                self._chunk_count += 1

                if self._vad_stream is not None:
                    for vad_event in self._vad_stream.process_pcm16(pcm):
                        if vad_event.event == "start":
                            logger.info(
                                "session_id={} event=vad_speech_start sample_index={} speech_prob={}",
                                self.session_id,
                                vad_event.sample_index,
                                round(vad_event.speech_prob, 4),
                            )
                            if self._silence_debounce_task and not self._silence_debounce_task.done():
                                self._silence_debounce_task.cancel()
                            self._silence_debounce_task = None
                            if self._is_agent_speaking:
                                logger.info(
                                    "session_id={} event=server_vad_barge_in sample_index={} speech_prob={}",
                                    self.session_id,
                                    vad_event.sample_index,
                                    round(vad_event.speech_prob, 4),
                                )
                                asyncio.ensure_future(self._handle_interrupt())
                        elif vad_event.event == "end":
                            logger.info(
                                "session_id={} event=vad_speech_end sample_index={} speech_prob={}",
                                self.session_id,
                                vad_event.sample_index,
                                round(vad_event.speech_prob, 4),
                            )
                            if not self._is_agent_speaking:
                                if self._settings.stream_smart_turn_enabled:
                                    # Route through debounce so Smart Turn can gate
                                    # on semantic completeness before finalising.
                                    if self._silence_debounce_task and not self._silence_debounce_task.done():
                                        self._silence_debounce_task.cancel()
                                    self._silence_debounce_task = asyncio.create_task(
                                        self._silence_debounce_then_fire(
                                            self._last_text_sent, "vad_end", vad_triggered=True
                                        )
                                    )
                                else:
                                    self._schedule_speech_finalization("vad_end")
                elif self._is_agent_speaking:
                    # Fallback RMS gate when the dedicated VAD is disabled.
                    samples = resampled.to_ndarray().astype(np.float32) / 32_768.0
                    rms = float(np.sqrt(np.mean(samples ** 2)))
                    if rms > _BARGE_IN_THRESHOLD:
                        self._barge_in_count += 1
                        if self._barge_in_count >= _BARGE_IN_FRAMES:
                            logger.info(
                                "session_id={} event=server_vad_barge_in rms={}",
                                self.session_id,
                                round(rms, 4),
                            )
                            asyncio.ensure_future(self._handle_interrupt())
                    else:
                        self._barge_in_count = 0

                await self._maybe_emit_stt()

        logger.info("session_id={} event=audio_consumer_stopped", self.session_id)

    async def _maybe_emit_stt(self) -> None:
        """Emit a partial STT result when the buffer and time-interval thresholds are met.

        Skips emission if a speech-finalization task or LLM task is already in
        flight to avoid unnecessary transcription during active processing.
        On new text, sends a ``partial`` message and (re)starts the silence
        debounce timer.
        """
        if not self._pcm_buffer:
            return
        if self._speech_finalization_task and not self._speech_finalization_task.done():
            self._last_emit_at = perf_counter()
            return
        if self._llm_task and not self._llm_task.done():
            self._last_emit_at = perf_counter()
            return
        buffered_ms = len(self._pcm_buffer) / 2 / self._sample_rate * 1000
        now = perf_counter()
        if not (
            buffered_ms >= self._settings.stream_min_audio_ms
            and (now - self._last_emit_at) * 1000 >= self._settings.stream_emit_interval_ms
        ):
            return

        loop = asyncio.get_event_loop()
        async with self._transcribe_lock:
            result = await loop.run_in_executor(None, self._transcribe_buffer)
        if self._speech_finalization_task and not self._speech_finalization_task.done():
            self._last_emit_at = perf_counter()
            logger.info(
                "session_id={} event=partial_stt_skipped reason=turn_finalizing",
                self.session_id,
            )
            return
        if self._llm_task and not self._llm_task.done():
            self._last_emit_at = perf_counter()
            logger.info(
                "session_id={} event=partial_stt_skipped reason=llm_in_flight",
                self.session_id,
            )
            return
        current_text = str(result.get("text", ""))
        if current_text != self._last_text_sent:
            await self._send_json({"type": "partial", **result})
            self._last_text_sent = current_text

            if self._silence_debounce_task and not self._silence_debounce_task.done():
                self._silence_debounce_task.cancel()
            self._silence_debounce_task = asyncio.create_task(
                self._silence_debounce_then_fire(current_text, "debounced_partial")
            )
        self._last_emit_at = perf_counter()

    def _transcribe_buffer(self) -> dict:
        """Write ``self._pcm_buffer`` to a temp WAV file and run faster-whisper.

        Serialises the current PCM buffer to a temporary WAV, runs the STT
        service, annotates timing fields, then deletes the temp file.

        Returns:
            Dict with keys ``text`` (str), ``timings_ms`` (dict), ``debug`` (dict).
        """
        started_at = perf_counter()
        temp_path = self._settings.temp_dir / f"{self.session_id}_rtc.wav"
        try:
            pcm = get_denoise_service().enhance(bytes(self._pcm_buffer), self._sample_rate)
            with wave.open(str(temp_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                wf.writeframes(pcm)

            service = get_stt_service()
            result = service.transcribe(
                file_path=temp_path,
                request_id=self.session_id,
                filename=f"rtc_{self.session_id}.wav",
                audio_bytes=len(self._pcm_buffer),
            )
            result.timings_ms.buffered_audio_ms = round(
                len(self._pcm_buffer) / 2 / self._sample_rate * 1000, 2
            )
            result.timings_ms.total_ms = round((perf_counter() - started_at) * 1000, 2)
            result.debug.sample_rate = self._sample_rate
            result.debug.chunks_received = self._chunk_count
            return {
                "text": result.text,
                "timings_ms": result.timings_ms.model_dump(),
                "debug": result.debug.model_dump(),
            }
        finally:
            temp_path.unlink(missing_ok=True)

    def _schedule_speech_finalization(self, trigger: str) -> None:
        """Create a speech-finalization task if one is not already running.

        A no-op if ``_speech_finalization_task`` is still active, preventing
        double-firing on rapid VAD end events.

        Args:
            trigger: Label passed through to ``_finalize_speech_turn`` for logging
                     (e.g. ``"vad_end"`` or ``"debounced_partial"``).
        """
        if self._speech_finalization_task and not self._speech_finalization_task.done():
            return
        self._speech_finalization_task = asyncio.create_task(
            self._finalize_speech_turn(trigger)
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
                return
            if self._is_agent_speaking:
                logger.info(
                    "session_id={} event=speech_finalization_skipped reason=agent_speaking",
                    self.session_id,
                )
                return
            if self._llm_task and not self._llm_task.done():
                logger.info(
                    "session_id={} event=speech_finalization_skipped reason=llm_in_flight",
                    self.session_id,
                )
                return

            if self._silence_debounce_task and not self._silence_debounce_task.done():
                self._silence_debounce_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._silence_debounce_task
                self._silence_debounce_task = None

            loop = asyncio.get_event_loop()
            async with self._transcribe_lock:
                result = await loop.run_in_executor(None, self._transcribe_buffer)

            if self._llm_task and not self._llm_task.done():
                logger.info(
                    "session_id={} event=speech_finalization_skipped reason=llm_started_during_transcribe",
                    self.session_id,
                )
                return

            final_text = str(result.get("text", "")).strip()
            if not final_text:
                logger.info(
                    "session_id={} event=speech_finalization_empty trigger={}",
                    self.session_id,
                    trigger,
                )
                return

            self._last_text_sent = final_text
            await self._send_json({"type": "final", **result})
            if final_text != self._latest_llm_input:
                self._schedule_llm(final_text)
        finally:
            self._speech_finalization_task = None

    # ── Interrupt ─────────────────────────────────────────────────────────────

    async def _handle_interrupt(self) -> None:
        """Cancel all in-flight tasks and reset audio/barge-in state.

        Sets the interrupt event so streaming coroutines exit cleanly, then
        cancels (in order) TTS and LLM tasks to prevent stale audio being
        sent after the interrupt.
        """
        self._interrupt_event.set()
        self._pending_llm_call = None
        self._is_agent_speaking = False
        self._barge_in_count = 0
        self._pcm_buffer.clear()
        self._last_text_sent = ""
        self._chunk_count = 0
        self._last_emit_at = 0.0
        if self._vad_stream is not None:
            self._vad_stream.reset()

        if self._silence_debounce_task and not self._silence_debounce_task.done():
            self._silence_debounce_task.cancel()
            self._silence_debounce_task = None

        if self._speech_finalization_task and not self._speech_finalization_task.done():
            self._speech_finalization_task.cancel()
            self._speech_finalization_task = None

        # Cancel TTS pipeline first so it doesn't send more audio after the new
        # _run_llm clears interrupt_event.
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
            self._tts_task = None

        if self._llm_task and not self._llm_task.done():
            self._llm_task.cancel()
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
            await asyncio.sleep(wait_ms / 1000)
        except asyncio.CancelledError:
            return

        if self._settings.stream_smart_turn_enabled:
            from app.services.smart_turn import get_smart_turn_service
            smart_turn = get_smart_turn_service()
            if smart_turn.is_loaded:
                turn_complete = False
                deadline = perf_counter() + self._settings.stream_smart_turn_max_budget_ms / 1000
                while perf_counter() < deadline:
                    is_complete, _ = smart_turn.predict(bytes(self._pcm_buffer))
                    if is_complete:
                        turn_complete = True
                        break
                    try:
                        await asyncio.sleep(self._settings.stream_smart_turn_base_wait_ms / 1000)
                    except asyncio.CancelledError:
                        return

                if not turn_complete:
                    # Smart Turn says utterance is incomplete — give the user
                    # extra time to continue their thought.  VAD "start" will
                    # cancel this task the moment they resume speaking.
                    try:
                        await asyncio.sleep(
                            self._settings.stream_smart_turn_incomplete_wait_ms / 1000
                        )
                    except asyncio.CancelledError:
                        return

        if self._vad_stream is not None:
            self._schedule_speech_finalization(trigger)
            return
        self._schedule_llm(text)

    def _schedule_llm(self, text: str) -> None:
        """Gate and enqueue an LLM call, cancelling any stale in-flight call.

        Ignores short transcripts, duplicate inputs, and pause commands.
        If an LLM is already running for older text, it is cancelled so the
        newer, more complete utterance takes priority.

        Args:
            text: Transcript candidate to send to the LLM.
        """
        normalized = text.strip()
        if len(normalized) < self._settings.stream_llm_min_chars:
            return
        if normalized == self._latest_llm_input:
            return
        if _PAUSE_PATTERN.match(normalized):
            return

        if self._llm_task and not self._llm_task.done():
            logger.info(
                "session_id={} event=llm_cancel_for_newer_text", self.session_id
            )
            self._interrupt_event.set()
            self._pending_llm_call = None
            if self._tts_task and not self._tts_task.done():
                self._tts_task.cancel()
                self._tts_task = None
            self._llm_task.cancel()
            self._llm_task = None

        self._pending_llm_call = normalized
        if self._llm_task is None or self._llm_task.done():
            next_text = self._pending_llm_call
            self._pending_llm_call = None
            self._llm_task = asyncio.create_task(self._run_llm(next_text))

    # ── LLM + TTS pipeline ───────────────────────────────────────────────────

    async def _tts_sentence_pipeline(self, queue: asyncio.Queue, enable_barge_in: bool = True) -> None:
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
        tts_service = get_tts_service()
        tts_started = False
        if enable_barge_in:
            self._is_agent_speaking = True

        while True:
            sentence = await queue.get()
            if sentence is None or self._interrupt_event.is_set():
                break
            try:
                wav_bytes, sr = await tts_service.synthesize(sentence, voice=self._voice_id, speed=self._tts_speed)
            except Exception as err:
                logger.warning(
                    "session_id={} event=tts_error error={}", self.session_id, err
                )
                continue
            if self._interrupt_event.is_set():
                break
            if not tts_started:
                tts_started = True
                tts_t0 = perf_counter()
                await self._send_json({"type": "tts_start"})
            tts_ms = round((perf_counter() - tts_t0) * 1000, 2)
            wav_b64 = base64.b64encode(wav_bytes).decode()
            await self._send_json({
                "type": "tts_audio",
                "data": wav_b64,
                "sample_rate": sr,
                "tts_ms": tts_ms,
                "sentence_text": sentence,
            })

        self._is_agent_speaking = False
        if self._interrupt_event.is_set():
            await self._send_json({"type": "tts_interrupted"})
        else:
            await self._send_json({"type": "tts_done"})

    async def _run_llm(self, text: str) -> None:
        """Run one full LLM inference turn and stream TTS audio sentence-by-sentence.

        Clears the interrupt flag, streams tokens from the LLM, extracts sentence
        boundaries to feed a concurrent TTS pipeline, and appends the completed
        turn to conversation history on success.  Queues the next pending LLM
        call (if any) once this one finishes.

        Args:
            text: User transcript to send as the current user message.
        """
        self._interrupt_event.clear()
        self._latest_llm_input = text

        full_response = ""
        processed_chars = 0
        llm_t0 = perf_counter()
        call_error: str | None = None

        sent_queue: asyncio.Queue[str | None] = asyncio.Queue()
        tts_task = asyncio.create_task(self._tts_sentence_pipeline(sent_queue))
        self._tts_task = tts_task

        try:
            await self._send_json({"type": "llm_start", "user_text": text})
            async for token in stream_llm_response(
                text, conversation_history=list(self._conversation_history)
            ):
                if self._interrupt_event.is_set():
                    break
                full_response += token
                await self._send_json(
                    {"type": "llm_partial", "text": strip_emotion_tags(full_response)}
                )
                # Extract complete sentences and enqueue for immediate TTS.
                tail = full_response[processed_chars:]
                while True:
                    m = _SENT_BOUNDARY.search(tail)
                    if not m or m.end() < _MIN_SENTENCE_CHARS:
                        break
                    sentence = clean_for_tts(tail[: m.end()].strip())
                    if sentence:
                        await sent_queue.put(sentence)
                    processed_chars += m.end()
                    tail = full_response[processed_chars:]

            llm_ms = round((perf_counter() - llm_t0) * 1000, 2)
            display_text = strip_emotion_tags(full_response)
            await self._send_json(
                {"type": "llm_final", "text": display_text, "llm_ms": llm_ms}
            )
        except Exception as err:
            call_error = str(err)
            logger.warning(
                "session_id={} event=llm_error error={}", self.session_id, err
            )
            await self._send_json(
                {"type": "llm_error", "message": "LLM unavailable — check your LLM provider and model."}
            )
        finally:
            if not self._interrupt_event.is_set() and call_error is None:
                remaining = clean_for_tts(full_response[processed_chars:].strip())
                if remaining:
                    sent_queue.put_nowait(remaining)
            sent_queue.put_nowait(None)

            # Save the exchange to history inside finally so it runs even when
            # the task is cancelled mid-stream. Without this, interrupted responses
            # are missing from history and the LLM re-answers the same question
            # when the user says "All right." or similar after a barge-in.
            if full_response and call_error is None:
                self._conversation_history.append({"role": "user", "content": text})
                self._conversation_history.append({"role": "assistant", "content": full_response})
                max_msgs = self._settings.llm_max_history_turns * 2
                if len(self._conversation_history) > max_msgs:
                    self._conversation_history[:] = self._conversation_history[-max_msgs:]

        with suppress(asyncio.CancelledError):
            await tts_task
        self._tts_task = None

        # Clear audio buffer after the full turn so any speech the user started
        # during the agent's response isn't fed into the next query.
        if not self._interrupt_event.is_set():
            self._pcm_buffer.clear()
            self._last_text_sent = ""
            self._chunk_count = 0
            self._last_emit_at = 0.0
            if self._vad_stream is not None:
                self._vad_stream.reset()

        if self._pending_llm_call:
            next_text = self._pending_llm_call
            self._pending_llm_call = None
            if next_text != self._latest_llm_input:
                self._llm_task = asyncio.create_task(self._run_llm(next_text))
                return
        self._llm_task = None

    # ── Welcome message ──────────────────────────────────────────────────────

    async def _run_welcome(self) -> None:
        """Synthesise and stream the configured welcome message at session open.

        Splits ``settings.welcome_message`` on sentence boundaries so the
        frontend receives individual ``tts_audio`` events with ``sentence_text``,
        matching the format produced by the regular LLM pipeline.  Skipped if
        the welcome message is empty.  Barge-in is disabled so ambient noise
        does not cancel the greeting before the user speaks.
        """
        welcome = self._settings.welcome_message
        if not welcome:
            return
        await self._send_json({"type": "llm_start", "user_text": ""})
        await self._send_json({"type": "llm_final", "text": welcome, "llm_ms": 0})

        sent_queue: asyncio.Queue[str | None] = asyncio.Queue()
        for raw in re.split(r"(?<=[.!?])\s+", welcome.strip()):
            sentence = clean_for_tts(raw.strip())
            if sentence:
                sent_queue.put_nowait(sentence)
        sent_queue.put_nowait(None)

        await self._tts_sentence_pipeline(sent_queue, enable_barge_in=False)

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _send_json(self, payload: dict) -> None:
        """Send a JSON message via the RTCDataChannel under a mutex.

        Silently drops the message if the channel is not open, preventing
        exceptions when the peer disconnects mid-stream.

        Args:
            payload: Dict to serialise and transmit over the data channel.
        """
        async with self._send_lock:
            if self.dc and self.dc.readyState == "open":
                try:
                    self.dc.send(json.dumps(payload))
                except Exception as exc:
                    logger.debug(
                        "session_id={} event=dc_send_error error={}", self.session_id, exc
                    )

    async def _cleanup(self) -> None:
        """Cancel all background tasks and mark the session as closed.

        Idempotent — subsequent calls after the first are no-ops.
        Called automatically when the peer connection transitions to a terminal
        state (``"failed"``, ``"closed"``, or ``"disconnected"``).
        """
        if self._closed:
            return
        self._closed = True
        tasks = [t for t in (
            self._audio_task,
            self._silence_debounce_task,
            self._speech_finalization_task,
            self._tts_task,
            self._llm_task,
        ) if t and not t.done()]
        for task in tasks:
            task.cancel()
        # Await cancelled tasks so they finish before ICE/transport teardown.
        # Without this, tasks outlive the transport and hit aioice's closed socket.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("session_id={} event=rtc_session_closed", self.session_id)
