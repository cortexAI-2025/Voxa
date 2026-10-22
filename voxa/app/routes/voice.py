from fastapi import APIRouter, HTTPException, Request, Response, WebSocket
from twilio.twiml.voice_response import Connect, VoiceResponse

from core.config import get_settings
from core.logger import get_logger
from core.security import is_valid_twilio_request, make_stream_token
from services.audio_stream import handle_media_stream

router = APIRouter(prefix="/voice", tags=["voice"])
logger = get_logger(__name__)


@router.post("/incoming")
async def incoming_call(request: Request) -> Response:
    """Twilio webhook for an inbound call: answer with TwiML that opens a Media Stream to /voice/stream."""
    settings = get_settings()
    form = {k: str(v) for k, v in (await request.form()).items()}

    # Twilio signs the public URL, which may differ from request.url behind a proxy / Cloud Run.
    public_url = f"{settings.public_base_url}{request.url.path}"
    if not is_valid_twilio_request(public_url, form, request.headers.get("X-Twilio-Signature")):
        logger.warning("invalid_twilio_signature", path=request.url.path)
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    call_sid, caller = form.get("CallSid", ""), form.get("From", "unknown")
    logger.info("incoming_call", call_sid=call_sid, caller=caller)

    response = VoiceResponse()
    connect = Connect()
    stream = connect.stream(url=settings.stream_ws_url)
    stream.parameter(name="caller", value=caller)
    stream.parameter(name="token", value=make_stream_token(call_sid))
    response.append(connect)
    return Response(content=str(response), media_type="application/xml")


@router.websocket("/stream")
async def media_stream(websocket: WebSocket) -> None:
    """Twilio Media Stream WebSocket (authenticated by the token carried in the `start` event)."""
    await handle_media_stream(websocket)
