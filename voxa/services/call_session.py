"""Engine-agnostic call orchestration: authentication, context, engine choice + fallback, lifecycle."""
from __future__ import annotations

import asyncio
import base64

from core.config import get_settings
from core.logger import get_logger
from core.security import verify_stream_token
from services.ai_agent import build_call_context, summarize_and_save
from services.engines.base import Engine
from services.engines.cascade import CascadeEngine
from services.engines.live import LiveEngine
from services.tools import CallContext, ToolRunner
from services.transport import Transport

logger = get_logger(__name__)

LIVE_START_TIMEOUT = 8.0
FALLBACK_NOTICE = "Excusez-moi, j'ai eu un petit souci technique. Pouvez-vous répéter ?"


class CallSession:
    def __init__(self, transport: Transport) -> None:
        self._settings = get_settings()
        self.transport = transport
        self.ctx: CallContext | None = None
        self._tools: ToolRunner | None = None
        self.engine: Engine | None = None
        self.engine_name = ""
        self._watchdog: asyncio.Task | None = None
        self._switching = False

    async def on_start(self, start: dict) -> bool:
        call_sid = start.get("callSid", "")
        params = start.get("customParameters", {})
        if not verify_stream_token(call_sid, params.get("token", "")):
            logger.warning("stream_token_rejected", call_sid=call_sid)
            return False
        self.transport.call_sid, self.transport.stream_sid = call_sid, start["streamSid"]
        logger.info("stream_started", call_sid=call_sid, stream_sid=self.transport.stream_sid)

        self.ctx = await build_call_context(call_sid, params.get("caller", "unknown"))
        self._tools = ToolRunner(self.ctx)
        self._watchdog = asyncio.create_task(self._enforce_max_duration(), name="watchdog")

        if self._settings.voice_engine == "live":
            try:
                await self._start_live()
                return True
            except Exception as exc:
                logger.error("live_start_failed_falling_back", call_sid=call_sid, error=str(exc))
        await self._start_cascade(announce=None)
        return True

    # ── Engines ───────────────────────────────────────────────────────────────

    async def _start_live(self) -> None:
        engine = LiveEngine(self.transport, self.ctx, self._tools, self.finish, self._switch_to_cascade)
        try:
            await asyncio.wait_for(engine.start(), timeout=LIVE_START_TIMEOUT)
        except BaseException:
            await engine.close()
            raise
        self.engine, self.engine_name = engine, "live"

    async def _start_cascade(self, announce: str | None) -> None:
        engine = CascadeEngine(self.transport, self.ctx, self._tools, self.finish)
        await engine.start(announce=announce)
        self.engine, self.engine_name = engine, "cascade"
        logger.info("engine_active", call_sid=self.ctx.call_sid, engine="cascade")

    async def _switch_to_cascade(self) -> None:
        """Live session died for good mid-call: continue the same call on the cascade pipeline."""
        if self._switching:
            return
        self._switching = True
        logger.error("live_failed_switching_to_cascade", call_sid=self.ctx.call_sid)
        old, self.engine = self.engine, None
        if old:
            asyncio.create_task(old.close())  # we are called from the old engine's own task
        try:
            await self._start_cascade(announce=FALLBACK_NOTICE)
        except Exception as exc:
            logger.error("cascade_fallback_failed", error=str(exc))
            await self.finish()

    # ── Inbound ───────────────────────────────────────────────────────────────

    async def on_media(self, payload_b64: str) -> None:
        if self.engine:
            await self.engine.on_audio(base64.b64decode(payload_b64))

    def on_mark(self, name: str) -> None:
        if self.engine:
            self.engine.on_mark(name)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def _enforce_max_duration(self) -> None:
        await asyncio.sleep(self._settings.max_call_seconds)
        logger.warning("max_call_duration_reached", call_sid=self.ctx.call_sid if self.ctx else "")
        await self.finish()

    async def finish(self) -> None:
        """Hang up the phone leg (idempotent)."""
        await self.transport.hangup()

    async def close(self) -> None:
        if self._watchdog and not self._watchdog.done():
            self._watchdog.cancel()
        if self.engine:
            await self.engine.close()
        if self.ctx:
            await summarize_and_save(self.ctx)
        logger.info("session_closed", call_sid=self.transport.call_sid, engine=self.engine_name)
