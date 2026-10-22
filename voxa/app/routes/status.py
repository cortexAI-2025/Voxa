from fastapi import APIRouter, HTTPException, Request

from core.config import get_settings
from core.logger import get_logger
from core.security import is_valid_twilio_request

router = APIRouter(prefix="/status", tags=["status"])
logger = get_logger(__name__)


@router.post("/call")
async def call_status(request: Request) -> dict[str, bool]:
    """Twilio status callback (ringing, in-progress, completed…)."""
    form = {k: str(v) for k, v in (await request.form()).items()}
    public_url = f"{get_settings().public_base_url}{request.url.path}"
    if not is_valid_twilio_request(public_url, form, request.headers.get("X-Twilio-Signature")):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    logger.info(
        "call_status_update",
        call_sid=form.get("CallSid"),
        status=form.get("CallStatus"),
        duration=form.get("CallDuration", form.get("Duration", "0")),
    )
    return {"received": True}


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe (Cloud Run)."""
    return {"status": "ok", "service": "voxa"}
