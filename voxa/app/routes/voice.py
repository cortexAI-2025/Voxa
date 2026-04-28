from fastapi import APIRouter, Form, Request, WebSocket
from fastapi.responses import PlainTextResponse

from core.config import get_settings
from core.logger import get_logger
from services.audio_stream import handle_media_stream

router = APIRouter(prefix="/voice", tags=["voice"])
logger = get_logger(__name__)


@router.post("/incoming", response_class=PlainTextResponse)
async def incoming_call(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(...),
    To: str = Form(...),
) -> str:
    """
    Twilio webhook: called when a new inbound call arrives.
    Returns TwiML that instructs Twilio to open a Media Stream WebSocket back to us.
    """
    settings = get_settings()
    ws_url = f"{settings.public_base_url}/voice/stream/{CallSid}".replace(
        "https://", "wss://"
    ).replace("http://", "ws://")

    logger.info("incoming_call", call_sid=CallSid, from_=From, to=To)

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}">
      <Parameter name="caller" value="{From}"/>
    </Stream>
  </Connect>
</Response>"""
    return twiml


@router.websocket("/stream/{call_sid}")
async def media_stream(websocket: WebSocket, call_sid: str) -> None:
    """
    Twilio Media Stream WebSocket endpoint.
    Receives µ-law audio chunks from Twilio and handles full-duplex conversation.
    """
    await handle_media_stream(websocket, call_sid)
