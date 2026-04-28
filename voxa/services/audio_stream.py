"""
Bidirectional audio bridge between Twilio Media Streams and Voxa.

Protocol (Twilio → server):
  - JSON messages over WebSocket
  - event: "connected" | "start" | "media" | "stop"
  - media.payload: base64-encoded µ-law 8 kHz audio

Protocol (server → Twilio):
  - JSON {"event":"media","streamSid":"…","media":{"payload":"<b64>"}}
  - JSON {"event":"clear","streamSid":"…"}  ← flushes Twilio's play buffer (barge-in)
"""
from __future__ import annotations

import asyncio
import audioop
import base64
import json
from collections.abc import AsyncIterator

from fastapi import WebSocket

from core.config import get_settings
from core.logger import get_logger
from services.ai_agent import VoxaAgent

logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mulaw_to_linear16(payload_b64: str) -> bytes:
    raw = base64.b64decode(payload_b64)
    return audioop.ulaw2lin(raw, 2)  # type: ignore[attr-defined]


def _linear16_to_mulaw(pcm: bytes) -> str:
    mulaw = audioop.lin2ulaw(pcm, 2)  # type: ignore[attr-defined]
    return base64.b64encode(mulaw).decode()


def _rms_energy(pcm: bytes) -> float:
    return audioop.rms(pcm, 2)  # type: ignore[attr-defined]


def _build_media_message(stream_sid: str, payload_b64: str) -> str:
    return json.dumps(
        {"event": "media", "streamSid": stream_sid, "media": {"payload": payload_b64}}
    )


def _build_clear_message(stream_sid: str) -> str:
    return json.dumps({"event": "clear", "streamSid": stream_sid})


# ── Barge-in detector ─────────────────────────────────────────────────────────

class BargeInDetector:
    """
    Detects speech activity by monitoring RMS energy of incoming PCM frames.
    Raises `barge_in_event` when the caller starts speaking during TTS playback.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._threshold = settings.barge_in_energy_threshold
        self._silence_frames_needed = settings.barge_in_silence_frames
        self._consecutive_silent = 0
        self.barge_in_event = asyncio.Event()

    def reset(self) -> None:
        self._consecutive_silent = 0
        self.barge_in_event.clear()

    def process_frame(self, pcm: bytes, tts_active: bool) -> None:
        energy = _rms_energy(pcm)
        if tts_active and energy > self._threshold:
            logger.debug("barge_in_detected", energy=energy)
            self.barge_in_event.set()
            self._consecutive_silent = 0
        elif energy <= self._threshold:
            self._consecutive_silent += 1
        else:
            self._consecutive_silent = 0


# ── STT accumulator ───────────────────────────────────────────────────────────

class SpeechAccumulator:
    """
    Buffers incoming PCM until silence is detected, then yields the full utterance.
    This is a simple VAD-based approach; replace with Deepgram / Chirp for production.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._threshold = settings.barge_in_energy_threshold
        self._silence_frames_needed = settings.barge_in_silence_frames
        self._buffer: list[bytes] = []
        self._silent_frames = 0
        self._speaking = False

    def feed(self, pcm: bytes) -> bytes | None:
        energy = _rms_energy(pcm)
        if energy > self._threshold:
            self._speaking = True
            self._silent_frames = 0
            self._buffer.append(pcm)
            return None
        elif self._speaking:
            self._silent_frames += 1
            self._buffer.append(pcm)
            if self._silent_frames >= self._silence_frames_needed:
                utterance = b"".join(self._buffer)
                self._buffer.clear()
                self._silent_frames = 0
                self._speaking = False
                return utterance
        return None


# ── TTS streamer ──────────────────────────────────────────────────────────────

async def stream_tts_to_twilio(
    websocket: WebSocket,
    stream_sid: str,
    text_chunks: AsyncIterator[str],
    barge_in: BargeInDetector,
) -> None:
    """
    Convert each text chunk from Gemini to µ-law audio and push it to Twilio.
    Monitors barge_in_event; if set, sends a "clear" event and stops.

    NOTE: This stub uses a placeholder TTS encoder. Wire in Google Cloud TTS
    (streaming synthesis) or a comparable service for production.
    """
    async for text in text_chunks:
        if barge_in.barge_in_event.is_set():
            logger.info("tts_interrupted", stream_sid=stream_sid)
            await websocket.send_text(_build_clear_message(stream_sid))
            return

        # ── Placeholder: encode text as silence (replace with real TTS) ──────
        # In production: call Google Cloud TTS Streaming API and iterate audio chunks.
        pcm_chunk = _text_to_pcm_placeholder(text)

        payload = _linear16_to_mulaw(pcm_chunk)
        await websocket.send_text(_build_media_message(stream_sid, payload))
        logger.debug("tts_chunk_sent", chars=len(text), stream_sid=stream_sid)


def _text_to_pcm_placeholder(text: str) -> bytes:
    """Placeholder: generates silence of proportional duration. Replace with real TTS."""
    duration_ms = max(len(text) * 60, 200)  # rough 60 ms/char estimate
    num_samples = (8000 * duration_ms) // 1000
    return b"\x00" * (num_samples * 2)  # 16-bit silence


# ── Main WebSocket handler ────────────────────────────────────────────────────

async def handle_media_stream(websocket: WebSocket, call_sid: str) -> None:
    """
    Full-duplex WebSocket handler for a single Twilio Media Stream session.

    Loop:
      1. Receive Twilio frames (µ-law audio).
      2. Decode → PCM, run barge-in detection + VAD accumulation.
      3. On complete utterance: transcribe (STT stub) → LLM → TTS → Twilio.
      4. On barge-in: abort ongoing TTS, restart.
    """
    stream_sid: str = ""
    caller_phone: str = ""
    agent: VoxaAgent | None = None
    barge_in = BargeInDetector()
    accumulator = SpeechAccumulator()
    tts_task: asyncio.Task | None = None
    tts_active = False

    await websocket.accept()
    logger.info("ws_connected", call_sid=call_sid)

    try:
        async for raw_message in websocket.iter_text():
            msg = json.loads(raw_message)
            event = msg.get("event")

            if event == "connected":
                logger.info("twilio_connected", protocol=msg.get("protocol"))

            elif event == "start":
                stream_sid = msg["start"]["streamSid"]
                caller_phone = (
                    msg["start"].get("customParameters", {}).get("caller", "unknown")
                )
                logger.info("stream_started", stream_sid=stream_sid, caller=caller_phone)

                agent = VoxaAgent(call_sid=call_sid, caller_phone=caller_phone)
                await agent.initialize()

                # Send greeting immediately
                barge_in.reset()
                tts_active = True
                tts_task = asyncio.create_task(
                    stream_tts_to_twilio(
                        websocket, stream_sid, agent.get_greeting(), barge_in
                    )
                )
                tts_task.add_done_callback(lambda _: setattr(barge_in, "_tts_active", False))

            elif event == "media" and agent:
                pcm = _mulaw_to_linear16(msg["media"]["payload"])
                barge_in.process_frame(pcm, tts_active)

                # Abort TTS if barge-in triggered
                if barge_in.barge_in_event.is_set() and tts_task and not tts_task.done():
                    tts_task.cancel()
                    await websocket.send_text(_build_clear_message(stream_sid))
                    tts_active = False
                    barge_in.reset()

                utterance_pcm = accumulator.feed(pcm)
                if utterance_pcm is not None:
                    user_text = await _transcribe(utterance_pcm)
                    if user_text:
                        logger.info("user_utterance", call_sid=call_sid, text=user_text)
                        barge_in.reset()
                        tts_active = True
                        tts_task = asyncio.create_task(
                            stream_tts_to_twilio(
                                websocket,
                                stream_sid,
                                agent.stream_response(user_text),
                                barge_in,
                            )
                        )

            elif event == "stop":
                logger.info("stream_stopped", stream_sid=stream_sid)
                break

    except Exception as exc:
        logger.error("ws_error", call_sid=call_sid, error=str(exc))
    finally:
        if tts_task and not tts_task.done():
            tts_task.cancel()
        if agent:
            await agent.close()
        logger.info("ws_closed", call_sid=call_sid)


async def _transcribe(pcm: bytes) -> str:
    """
    STT stub. Replace with Google Cloud Speech-to-Text (streaming recognize)
    or Chirp for production-grade transcription.
    """
    # Placeholder: returns empty string so the system silently waits.
    # Wire in real STT here:
    # async with speech.SpeechAsyncClient() as client:
    #     ...
    return ""
