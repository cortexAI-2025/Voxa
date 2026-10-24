import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

from core.logger import get_logger
from services.tools import CallContext
from services.transport import Transport

logger = get_logger(__name__)

FINAL_MARK = "final"


class Engine(Protocol):
    async def start(self) -> None: ...
    async def on_audio(self, mulaw: bytes) -> None: ...
    def on_mark(self, name: str) -> None: ...
    async def close(self) -> None: ...


class Goodbye:
    """Hang up only after the goodbye audio was actually played (Twilio mark echo), never mid-sentence.

    A timer guarantees the call still ends if the mark never comes back.
    """

    def __init__(self, transport: Transport, ctx: CallContext, finish: Callable[[], Awaitable[None]], grace: float) -> None:
        self._transport, self._ctx, self._finish, self._grace = transport, ctx, finish, grace
        self._timer: asyncio.Task | None = None

    async def after_turn(self) -> None:
        """Call when the engine has sent the last audio of a turn."""
        if not self._ctx.end_requested or self._timer is not None:
            return
        await self._transport.mark(FINAL_MARK)
        self._timer = asyncio.create_task(self._force(), name="goodbye-timer")

    async def _force(self) -> None:
        await asyncio.sleep(self._grace)
        logger.warning("goodbye_mark_timeout", call_sid=self._ctx.call_sid)
        await self._finish()

    def on_mark(self, name: str) -> bool:
        """True if this mark was the final one (the caller must then hang up)."""
        return name == FINAL_MARK and self._timer is not None

    def cancel(self) -> None:
        if self._timer and not self._timer.done():
            self._timer.cancel()
