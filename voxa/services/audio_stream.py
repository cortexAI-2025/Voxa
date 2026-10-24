"""WebSocket entry point: parse Twilio Media Stream events and drive a CallSession."""
import json

from fastapi import WebSocket, WebSocketDisconnect

from core.logger import get_logger
from services.call_session import CallSession
from services.transport import TwilioTransport

logger = get_logger(__name__)


async def handle_media_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    transport = TwilioTransport(websocket)
    session = CallSession(transport)
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
                logger.info("stream_stopped", call_sid=transport.call_sid)
                return
    except WebSocketDisconnect:
        logger.info("ws_disconnected", call_sid=transport.call_sid)
    except Exception as exc:
        logger.error("ws_error", call_sid=transport.call_sid, error=str(exc))
    finally:
        await session.close()
