"""Transport = protocol framing only. Engines never see Twilio JSON, so they are interchangeable."""
import asyncio
import base64
import json
from typing import Protocol

from fastapi import WebSocket
from twilio.rest import Client as TwilioRestClient

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


class Transport(Protocol):
    call_sid: str
    stream_sid: str

    async def send_audio(self, mulaw: bytes) -> None: ...
    async def clear(self) -> None: ...
    async def mark(self, name: str) -> None: ...
    async def hangup(self) -> None: ...


class TwilioTransport:
    """Twilio Media Streams over a FastAPI WebSocket (µ-law 8 kHz)."""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self._lock = asyncio.Lock()
        self.call_sid = ""
        self.stream_sid = ""
        self._hung_up = False

    async def _send(self, message: dict) -> None:
        async with self._lock:
            await self._ws.send_text(json.dumps({**message, "streamSid": self.stream_sid}))

    async def send_audio(self, mulaw: bytes) -> None:
        await self._send({"event": "media", "media": {"payload": base64.b64encode(mulaw).decode()}})

    async def clear(self) -> None:
        """Flush Twilio's playback buffer (barge-in)."""
        await self._send({"event": "clear"})

    async def mark(self, name: str) -> None:
        """Twilio echoes the mark once all audio sent before it has been played."""
        await self._send({"event": "mark", "mark": {"name": name}})

    async def hangup(self) -> None:
        """End the call via the REST API (the WebSocket closing alone does not hang up a <Connect> call)."""
        if self._hung_up or not self.call_sid:
            return
        self._hung_up = True
        settings = get_settings()

        def _complete() -> None:
            TwilioRestClient(settings.twilio_account_sid, settings.twilio_auth_token).calls(
                self.call_sid
            ).update(status="completed")

        try:
            await asyncio.to_thread(_complete)  # the Twilio SDK is blocking
            logger.info("call_hung_up", call_sid=self.call_sid)
        except Exception as exc:
            logger.error("hangup_failed", call_sid=self.call_sid, error=str(exc))
