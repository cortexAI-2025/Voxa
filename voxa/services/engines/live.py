"""Gemini Live engine: caller audio in → native audio out, one WebSocket to Vertex AI.

Server-side VAD does turn-taking and barge-in: on `interrupted` we flush Twilio's buffer immediately.
Audio path: µ-law 8 kHz → PCM16 16 kHz (stateful resampler) → Gemini; Gemini PCM16 24 kHz → 8 kHz → µ-law.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack

from google.genai import types

from core.config import get_settings
from core.logger import get_logger
from services.ai_agent import build_system_instruction, get_genai_client, greeting_text
from services.audio_codec import ModelToPhone, PhoneToModel, chunk_audio, mulaw_rms
from services.engines.base import Goodbye
from services.tools import CallContext, ToolRunner
from services.transport import Transport

logger = get_logger(__name__)

MAX_RECONNECTS = 2
_VOICE_RMS = 500  # loud-frame threshold, only used for the latency metric


class LiveEngine:
    def __init__(
        self,
        transport: Transport,
        ctx: CallContext,
        tools: ToolRunner,
        finish: Callable[[], Awaitable[None]],
        on_fatal: Callable[[], Awaitable[None]],
    ) -> None:
        self._settings = get_settings()
        self._transport, self._ctx, self._tools, self._finish, self._on_fatal = transport, ctx, tools, finish, on_fatal
        self._goodbye = Goodbye(transport, ctx, finish, self._settings.goodbye_grace_seconds)
        self._up, self._down = PhoneToModel(), ModelToPhone()

        self._stack: AsyncExitStack | None = None
        self._session = None
        self._receiver: asyncio.Task | None = None
        self._resume_handle: str | None = None
        self._closed = False

        self._turn = 0
        self.speaking = False
        self._in_text: list[str] = []
        self._out_text: list[str] = []
        self._last_voice_at: float | None = None

    # ── Connection ────────────────────────────────────────────────────────────

    def _config(self) -> types.LiveConnectConfig:
        s = self._settings
        client = get_genai_client()
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=build_system_instruction(self._ctx),
            speech_config=types.SpeechConfig(
                language_code=s.live_language_code,
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=s.live_voice)),
            ),
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration.from_callable(client=client, callable=fn) for fn in self._tools.callables
                    ]
                )
            ],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(silence_duration_ms=600, prefix_padding_ms=120)
            ),
            session_resumption=types.SessionResumptionConfig(handle=self._resume_handle),
            # Lets calls run past the default audio session limits.
            context_window_compression=types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow()),
        )

    async def _connect(self) -> None:
        self._stack = AsyncExitStack()
        cm = get_genai_client().aio.live.connect(model=self._settings.live_model, config=self._config())
        self._session = await self._stack.enter_async_context(cm)

    async def _disconnect(self) -> None:
        session, self._session = self._session, None
        if self._stack:
            stack, self._stack = self._stack, None
            try:
                await stack.aclose()
            except Exception as exc:
                logger.debug("live_disconnect_error", error=str(exc))
        del session

    async def start(self) -> None:
        await self._connect()
        self._receiver = asyncio.create_task(self._run(), name="live-receiver")
        await self._session.send_realtime_input(  # type: ignore[union-attr]
            text=f"[L'appel commence] Dis exactement ceci au client, rien de plus : « {greeting_text(self._ctx)} »"
        )

    async def close(self) -> None:
        self._closed = True
        self._goodbye.cancel()
        if self._receiver and not self._receiver.done():
            self._receiver.cancel()
            await asyncio.gather(self._receiver, return_exceptions=True)
        await self._disconnect()

    # ── Inbound audio ─────────────────────────────────────────────────────────

    async def on_audio(self, mulaw: bytes) -> None:
        if mulaw_rms(mulaw) > _VOICE_RMS:
            self._last_voice_at = asyncio.get_running_loop().time()
        session = self._session
        if session is None:  # reconnecting: drop audio instead of buffering stale speech
            return
        try:
            await session.send_realtime_input(
                audio=types.Blob(data=self._up(mulaw), mime_type="audio/pcm;rate=16000")
            )
        except Exception as exc:  # the receiver loop handles reconnection
            logger.debug("live_send_failed", error=str(exc))

    def on_mark(self, name: str) -> None:
        if self._goodbye.on_mark(name):
            asyncio.create_task(self._finish(), name="finish")
        elif name == f"turn-{self._turn}":
            self.speaking = False

    # ── Receive loop ──────────────────────────────────────────────────────────

    async def _run(self) -> None:
        reconnects = 0
        while not self._closed:
            try:
                async for message in self._session.receive():  # type: ignore[union-attr]
                    await self._handle(message)
                reconnects = 0  # receive() returns after each completed model turn
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._closed:
                    return
                logger.warning("live_session_error", call_sid=self._ctx.call_sid, error=str(exc), attempt=reconnects + 1)
                if reconnects >= MAX_RECONNECTS:
                    break
                reconnects += 1
                try:
                    await self._disconnect()
                    await self._connect()  # resumes with the last session handle
                    continue
                except Exception as exc2:
                    logger.error("live_reconnect_failed", error=str(exc2))
                    break
        if not self._closed:
            await self._on_fatal()  # hand the call over to the cascade engine

    async def _handle(self, message: types.LiveServerMessage) -> None:
        if message.session_resumption_update and message.session_resumption_update.new_handle:
            self._resume_handle = message.session_resumption_update.new_handle
        if message.go_away:
            logger.warning("live_go_away", call_sid=self._ctx.call_sid, time_left=str(message.go_away.time_left))
        if message.tool_call:
            await self._on_tool_call(message.tool_call)

        sc = message.server_content
        if sc is None:
            return
        if sc.interrupted:
            await self._on_interrupted()
        if sc.model_turn and sc.model_turn.parts:
            for part in sc.model_turn.parts:
                if part.inline_data and part.inline_data.data:
                    await self._on_model_audio(part.inline_data.data)
        if sc.input_transcription and sc.input_transcription.text:
            self._in_text.append(sc.input_transcription.text)
        if sc.output_transcription and sc.output_transcription.text:
            self._out_text.append(sc.output_transcription.text)
        if sc.turn_complete:
            await self._on_turn_complete()

    async def _on_model_audio(self, pcm24k: bytes) -> None:
        for piece in chunk_audio(self._down(pcm24k)):
            await self._transport.send_audio(piece)
        if not self.speaking:
            self.speaking = True
            if self._last_voice_at is not None:
                ms = int((asyncio.get_running_loop().time() - self._last_voice_at) * 1000)
                logger.info("turn_latency", call_sid=self._ctx.call_sid, first_audio_ms=ms)
                self._last_voice_at = None

    async def _on_interrupted(self) -> None:
        self._turn += 1
        self._down.reset()
        if self.speaking:
            self.speaking = False
            await self._transport.clear()
            logger.info("barge_in", call_sid=self._ctx.call_sid)
        self._flush_transcripts()

    async def _on_turn_complete(self) -> None:
        self._flush_transcripts()
        await self._transport.mark(f"turn-{self._turn}")
        await self._goodbye.after_turn()

    def _flush_transcripts(self) -> None:
        for role, parts in (("client", self._in_text), ("voxa", self._out_text)):
            text = "".join(parts).strip()
            if text:
                self._ctx.transcript.append((role, text))
            parts.clear()

    async def _on_tool_call(self, tool_call: types.LiveServerToolCall) -> None:
        responses = []
        for fc in tool_call.function_calls or []:
            result = await self._tools.run(fc.name or "", dict(fc.args or {}))
            responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=result))
        if responses and self._session is not None:
            await self._session.send_tool_response(function_responses=responses)
