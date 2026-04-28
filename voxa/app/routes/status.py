from fastapi import APIRouter, Form

from core.logger import get_logger

router = APIRouter(prefix="/status", tags=["status"])
logger = get_logger(__name__)


@router.post("/call")
async def call_status(
    CallSid: str = Form(...),
    CallStatus: str = Form(...),
    Duration: str = Form(default="0"),
) -> dict:
    """
    Twilio status callback: receives call lifecycle events (ringing, in-progress, completed…).
    Wire in Firestore write / analytics here as needed.
    """
    logger.info("call_status_update", call_sid=CallSid, status=CallStatus, duration=Duration)
    return {"received": True}


@router.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "service": "voxa"}
