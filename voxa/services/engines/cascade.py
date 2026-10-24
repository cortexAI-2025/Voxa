"""Cascade engine: Google STT (streaming) → Gemini text (streaming) → Google TTS (µ-law), with barge-in.

Barge-in: while Voxa is audible, N consecutive loud 20 ms frames, or an interim STT result of at least
BARGE_IN_MIN_CHARS characters, bump the "turn" counter, flush Twilio (`clear`) and cancel the response.
Every audio write re-checks the turn counter, so no stale audio can follow the `clear`.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

from google.cloud import speech_v1 as speech
from google.cloud import texttospeech_v1 as tts

from core.config import get_settings
from core.logger import get_logger
from services.ai_agent import VoxaAgent, greeting_text
from services.audio_codec import TWILIO_RATE, chunk_audio, mulaw_rms, strip_wav_header
from services.engines.base import Goodbye
from services.tools import CallContext, ToolRunner
from services.transport import Transport

logger = get_logger(__name__)

STT_STREAM_MAX_SECONDS = 280  # Google closes streaming sessions at ~5 min; rotate before that
_STT_QUEUE_FRAMES = 500  # ~10 s of buffered audio


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
                sample_rate_hertz=TWILIO_RATE,
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


class Synthesizer:
    """Google TTS straight to µ-law 8 kHz, with a cache for fixed phrases (greeting)."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = tts.TextToSpeechAsyncClient()
        self._voice = tts.VoiceSelectionParams(language_code=settings.tts_language_code, name=settings.tts_voice_name)
        self._audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MULAW, sample_rate_hertz=TWILIO_RATE)
        self._cache: dict[str, bytes] = {}

    async def synthesize(self, text: str, *, cache: bool = False) -> bytes:
        if cache and text in self._cache:
            return self._cache[text]
        response = await self._client.synthesize_speech(
            input=tts.SynthesisInput(text=text), voice=self._voice, audio_config=self._audio_config
        )
        audio = strip_wav_header(response.audio_content)
        if cache:
            self._cache[text] = audio
        return audio


async def _one(text: str) -> AsyncIterator[str]:
    yield text


class CascadeEngine:
    def __init__(
        self,
        transport: Transport,
        ctx: CallContext,
        tools: ToolRunner,
        finish: Callable[[], Awaitable[None]],
        *,
        agent_factory: Callable[[CallContext, ToolRunner], VoxaAgent] = VoxaAgent,
        transcriber_factory: Callable[[str], StreamingTranscriber] = StreamingTranscriber,
        synthesizer_factory: Callable[[], Synthesizer] = Synthesizer,
    ) -> None:
        self._settings = get_settings()
        self._transport, self._ctx, self._tools, self._finish = transport, ctx, tools, finish
        self._goodbye = Goodbye(transport, ctx, finish, self._settings.goodbye_grace_seconds)
        self._agent_factory = agent_factory
        self._transcriber_factory = transcriber_factory
        self._synthesizer_factory = synthesizer_factory

        self.agent: VoxaAgent | None = None
        self._stt: StreamingTranscriber | None = None
        self._tts: Synthesizer | None = None
        self._stt_task: asyncio.Task | None = None
        self._response_task: asyncio.Task | None = None

        self._turn = 0  # bumped on every new response / interruption
        self.speaking = False  # first audio sent → Twilio confirms playback ended
        self._loud_frames = 0
        self._utterance_at: float | None = None  # when the final transcript arrived (latency metric)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, announce: str | None = None) -> None:
        self._tts = self._synthesizer_factory()
        # Listen right away: audio spoken while the agent is still initialising is not lost.
        self._stt = self._transcriber_factory(self._settings.stt_language_code)
        self._stt.start()
        self._stt_task = asyncio.create_task(self._consume_transcripts(), name="transcripts")

        self.agent = self._agent_factory(self._ctx, self._tools)
        await self.agent.initialize()
        # `announce` replaces the greeting when taking over a call already in progress.
        text = announce or greeting_text(self._ctx)
        await self._respond(lambda: _one(text), cache=True)

    async def close(self) -> None:
        self._turn += 1
        self._goodbye.cancel()
        tasks = [t for t in (self._response_task, self._stt_task) if t]
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._stt:
            await self._stt.close()

    # ── Outbound ──────────────────────────────────────────────────────────────

    async def _speak(self, sentences: AsyncIterator[str], turn: int, cache: bool) -> None:
        assert self._tts is not None
        try:
            async for sentence in sentences:
                audio = await self._tts.synthesize(sentence, cache=cache)
                for piece in chunk_audio(audio):
                    if turn != self._turn:
                        return
                    await self._transport.send_audio(piece)
                    if not self.speaking:
                        self.speaking = True
                        self._log_first_audio()
            if turn == self._turn:
                await self._transport.mark(f"turn-{turn}")
                await self._goodbye.after_turn()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("response_error", call_sid=self._ctx.call_sid, error=str(exc))
        finally:
            await sentences.aclose()  # type: ignore[attr-defined]  # stop the Gemini stream

    def _log_first_audio(self) -> None:
        if self._utterance_at is not None:
            ms = int((asyncio.get_running_loop().time() - self._utterance_at) * 1000)
            logger.info("turn_latency", call_sid=self._ctx.call_sid, first_audio_ms=ms)
            self._utterance_at = None

    async def _interrupt(self) -> None:
        """Stop the current response and flush whatever Twilio has queued."""
        self._turn += 1
        task, self._response_task = self._response_task, None
        if task and not task.done():
            task.cancel()
        if self.speaking:
            self.speaking = False
            await self._transport.clear()
            logger.info("barge_in", call_sid=self._ctx.call_sid)

    async def _respond(self, make_sentences: Callable[[], AsyncIterator[str]], *, cache: bool = False) -> None:
        await self._interrupt()
        self._loud_frames = 0
        self._response_task = asyncio.create_task(
            self._speak(make_sentences(), self._turn, cache), name="response"
        )

    # ── Inbound ───────────────────────────────────────────────────────────────

    async def on_audio(self, mulaw: bytes) -> None:
        if self._stt:
            self._stt.feed(mulaw)
        if not self.speaking:
            self._loud_frames = 0
            return
        if mulaw_rms(mulaw) > self._settings.barge_in_energy_threshold:
            self._loud_frames += 1
            if self._loud_frames >= self._settings.barge_in_min_frames:
                self._loud_frames = 0
                await self._interrupt()
        else:
            self._loud_frames = 0

    def on_mark(self, name: str) -> None:
        if self._goodbye.on_mark(name):
            asyncio.create_task(self._finish(), name="finish")
        elif name == f"turn-{self._turn}":  # marks of stale turns are ignored
            self.speaking = False

    async def _consume_transcripts(self) -> None:
        assert self._stt is not None
        async for text, is_final in self._stt.results():
            if not text:
                continue
            if not is_final:
                # Interim text confirms the caller is talking; ignore 'mm' / 'ah' backchannel.
                if self.speaking and len(text) >= self._settings.barge_in_min_chars:
                    await self._interrupt()
                continue
            logger.info("user_utterance", call_sid=self._ctx.call_sid, chars=len(text))
            self._utterance_at = asyncio.get_running_loop().time()
            await self._respond(lambda t=text: self.agent.reply(t))  # type: ignore[union-attr]
