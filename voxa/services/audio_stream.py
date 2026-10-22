"""
Full-duplex bridge between a Twilio Media Stream and Voxa.

Inbound  (Twilio → us): JSON frames, µ-law 8 kHz mono, 20 ms (160 bytes) each.
Outbound (us → Twilio): {"event":"media"} audio, {"event":"mark"} playback markers,
                        {"event":"clear"} to flush Twilio's playback buffer (barge-in).

Pipeline per call:
    media ─► StreamingTranscriber (Google STT) ─► final transcript ─► VoxaAgent (Gemini, streamed)
          ─► sentence ─► Google TTS (µ-law 8 kHz) ─► media ─► Twilio

Barge-in: while Voxa is speaking, ≥ N consecutive loud inbound frames (or a non-empty interim STT
result) bump the "turn" counter, send `clear` and cancel the response task. Every outbound audio
write re-checks the turn counter under a lock, so no stale audio can follow the `clear`.
"""
from __future__ import annotations

import asyncio
import audioop
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect
from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts

from core.config import get_settings
from core.logger import get_logger
from core.security import verify_stream_token
from services.ai_agent import VoxaAgent

logger = get_logger(__name__)

SAMPLE_RATE = 8000
AUDIO_CHUNK_BYTES = 3200  # 400 ms of µ-law per outbound media message
STT_STREAM_MAX_SECONDS = 280  # Google closes streaming sessions at ~5 min; rotate before that
_STT_QUEUE_FRAMES = 500  # ~10 s of buffered audio


# ── Audio helpers ─────────────────────────────────────────────────────────────

def mulaw_rms(frame: bytes) -> int:
    """RMS energy (16-bit PCM scale) of a µ-law frame."""
    return audioop.rms(audioop.ulaw2lin(frame, 2), 2) if frame else 0


def strip_wav_header(audio: bytes) -> bytes:
    """Google TTS returns MULAW wrapped in a WAV container; Twilio wants the raw samples."""
    if audio[:4] == b"RIFF":
        idx = audio.find(b"data")
        if idx != -1:
            return audio[idx + 8:]
    return audio


def chunk_audio(audio: bytes, size: int = AUDIO_CHUNK_BYTES) -> list[bytes]:
    return [audio[i:i + size] for i in range(0, len(audio), size)]


# ── Speech-to-Text (streaming) ────────────────────────────────────────────────

class StreamingTranscriber:
    """Feeds µ-law frames to Google STT streaming recognition and yields (text, is_final)."""

    def __init__(self, language_code: str) -> None:
        self._language = language_code
        self._audio: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=_STT_QUEUE_FRAMES)
        self._results: asyncio.Queue[tuple[str, bool] | None] = asyncio.Queue()
        self._closed = False
        self._task: asyncio.Task | None = None
        self._client: speech.SpeechAsyncClient | None = None

    def start(self) -> None:
        self._client = speech.SpeechAsyncClient()
        self._task = asyncio.create_task(self._run(), name="stt")

    def feed(self, frame: bytes) -> None:
        try:
            self._audio.put_nowait(frame)
        except asyncio.QueueFull:  # STT is lagging: drop audio rather than grow unbounded
            logger.warning("stt_queue_full_dropping_frame")

    async def close(self) -> None:
        self._closed = True
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def results(self) -> AsyncIterator[tuple[str, bool]]:
        while (item := await self._results.get()) is not None:
            yield item

    async def _run(self) -> None:
        try:
            while not self._closed:
                try:
                    await self._recognize_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error("stt_stream_error", error=str(exc))
                    await asyncio.sleep(0.5)
        finally:
            self._results.put_nowait(None)

    async def _recognize_once(self) -> None:
        assert self._client is not None
        config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
                sample_rate_hertz=SAMPLE_RATE,
                language_code=self._language,
                model="telephony",
                enable_automatic_punctuation=True,
            ),
            interim_results=True,
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + STT_STREAM_MAX_SECONDS

        async def requests() -> AsyncIterator[speech.StreamingRecognizeRequest]:
            yield speech.StreamingRecognizeRequest(streaming_config=config)
            while (remaining := deadline - loop.time()) > 0:
                try:
                    frame = await asyncio.wait_for(self._audio.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    return
                if frame is None:
                    return
                yield speech.StreamingRecognizeRequest(audio_content=frame)

        responses = await self._client.streaming_recognize(requests=requests())
        async for response in responses:
            for result in response.results:
                if result.alternatives:
                    self._results.put_nowait((result.alternatives[0].transcript.strip(), result.is_final))


# ── Text-to-Speech ────────────────────────────────────────────────────────────

class Synthesizer:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = tts.TextToSpeechAsyncClient()
        self._voice = tts.VoiceSelectionParams(
            language_code=settings.tts_language_code, name=settings.tts_voice_name
        )
        self._audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.MULAW, sample_rate_hertz=SAMPLE_RATE
        )

    async def synthesize(self, text: str) -> bytes:
        response = await self._client.synthesize_speech(
            input=tts.SynthesisInput(text=text), voice=self._voice, audio_config=self._audio_config
        )
        return strip_wav_header(response.audio_content)


# ── Call session ──────────────────────────────────────────────────────────────

class CallSession:
    """All the state and behaviour for one phone call."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        agent_factory: Callable[[str, str], VoxaAgent] = VoxaAgent,
        transcriber_factory: Callable[[str], StreamingTranscriber] = StreamingTranscriber,
        synthesizer_factory: Callable[[], Synthesizer] = Synthesizer,
    ) -> None:
        self._ws = websocket
        self._settings = get_settings()
        self._agent_factory = agent_factory
        self._transcriber_factory = transcriber_factory
        self._synthesizer_factory = synthesizer_factory

        self.call_sid = ""
        self.stream_sid = ""
        self.agent: VoxaAgent | None = None
        self._stt: StreamingTranscriber | None = None
        self._tts: Synthesizer | None = None
        self._stt_task: asyncio.Task | None = None
        self._response_task: asyncio.Task | None = None

        self._send_lock = asyncio.Lock()
        self._turn = 0  # bumped on every new response / interruption
        self.speaking = False  # True from first audio sent until Twilio confirms playback ended
        self._loud_frames = 0

    # ── Outbound ──────────────────────────────────────────────────────────────

    async def _send(self, message: dict, turn: int | None = None) -> bool:
        """Send a JSON message; drops it if `turn` is stale (interrupted since it was produced)."""
        async with self._send_lock:
            if turn is not None and turn != self._turn:
                return False
            await self._ws.send_text(json.dumps({**message, "streamSid": self.stream_sid}))
            return True

    async def _speak(self, sentences: AsyncIterator[str], turn: int) -> None:
        assert self._tts is not None
        try:
            async for sentence in sentences:
                audio = await self._tts.synthesize(sentence)
                for piece in chunk_audio(audio):
                    payload = base64.b64encode(piece).decode()
                    if not await self._send({"event": "media", "media": {"payload": payload}}, turn):
                        return
                    self.speaking = True
                logger.debug("tts_sentence_sent", call_sid=self.call_sid, chars=len(sentence))
            await self._send({"event": "mark", "mark": {"name": f"turn-{turn}"}}, turn)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("response_error", call_sid=self.call_sid, error=str(exc))
        finally:
            await sentences.aclose()  # type: ignore[attr-defined]  # stop the Gemini stream

    async def interrupt(self) -> None:
        """Stop the current response and flush whatever Twilio has queued."""
        self._turn += 1
        task, self._response_task = self._response_task, None
        if task and not task.done():
            task.cancel()
        if self.speaking:
            self.speaking = False
            await self._send({"event": "clear"})
            logger.info("barge_in", call_sid=self.call_sid)

    async def respond(self, make_sentences: Callable[[], AsyncIterator[str]]) -> None:
        await self.interrupt()
        self._loud_frames = 0
        self._response_task = asyncio.create_task(self._speak(make_sentences(), self._turn), name="response")

    # ── Inbound events ────────────────────────────────────────────────────────

    async def on_start(self, start: dict) -> bool:
        self.call_sid = start.get("callSid", "")
        self.stream_sid = start["streamSid"]
        params = start.get("customParameters", {})
        if not verify_stream_token(self.call_sid, params.get("token", "")):
            logger.warning("stream_token_rejected", call_sid=self.call_sid)
            return False

        caller = params.get("caller", "unknown")
        logger.info("stream_started", call_sid=self.call_sid, stream_sid=self.stream_sid)

        self._tts = self._synthesizer_factory()
        self._stt = self._transcriber_factory(self._settings.stt_language_code)
        self._stt.start()
        self._stt_task = asyncio.create_task(self._consume_transcripts(), name="transcripts")

        self.agent = self._agent_factory(self.call_sid, caller)
        await self.agent.initialize()  # Firestore profile → Gemini context, before the first word
        await self.respond(self.agent.greeting)
        return True

    async def on_media(self, payload_b64: str) -> None:
        frame = base64.b64decode(payload_b64)
        if self._stt:
            self._stt.feed(frame)

        if not self.speaking:
            self._loud_frames = 0
            return
        if mulaw_rms(frame) > self._settings.barge_in_energy_threshold:
            self._loud_frames += 1
            if self._loud_frames >= self._settings.barge_in_min_frames:
                self._loud_frames = 0
                await self.interrupt()
        else:
            self._loud_frames = 0

    def on_mark(self, name: str) -> None:
        # Twilio echoes a mark once all audio before it was played; ignore marks of stale turns.
        if name == f"turn-{self._turn}":
            self.speaking = False

    async def _consume_transcripts(self) -> None:
        assert self._stt is not None and self.agent is not None
        async for text, is_final in self._stt.results():
            if not text:
                continue
            if not is_final:
                if self.speaking:  # caller started talking: STT confirms what the energy gate saw
                    await self.interrupt()
                continue
            logger.info("user_utterance", call_sid=self.call_sid, text=text)
            await self.respond(lambda t=text: self.agent.reply(t))  # type: ignore[union-attr]

    async def close(self) -> None:
        self._turn += 1
        for task in (self._response_task, self._stt_task):
            if task and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (self._response_task, self._stt_task) if t), return_exceptions=True)
        if self._stt:
            await self._stt.close()
        if self.agent:
            await self.agent.close()
        logger.info("session_closed", call_sid=self.call_sid)


async def handle_media_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint body: accept, dispatch Twilio events, always clean up."""
    await websocket.accept()
    session = CallSession(websocket)
    try:
        while True:
            message = json.loads(await websocket.receive_text())
            event = message.get("event")
            if event == "start":
                if not await session.on_start(message["start"]):
                    await websocket.close(code=1008)
                    return
            elif event == "media":
                await session.on_media(message["media"]["payload"])
            elif event == "mark":
                session.on_mark(message.get("mark", {}).get("name", ""))
            elif event == "stop":
                logger.info("stream_stopped", call_sid=session.call_sid)
                return
    except WebSocketDisconnect:
        logger.info("ws_disconnected", call_sid=session.call_sid)
    except Exception as exc:
        logger.error("ws_error", call_sid=session.call_sid, error=str(exc))
    finally:
        await session.close()
