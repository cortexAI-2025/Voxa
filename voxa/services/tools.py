"""Tools the model can call. Authorization lives here, in code, never in the prompt.

The customer identity comes from the call itself (caller number → CallContext), never from a model
argument, so no phrasing of a request can make a tool return another customer's data.
"""
from dataclasses import dataclass, field
from typing import Any

from core.config import get_settings
from core.logger import get_logger
from services.database import CustomerProfile, create_escalation

logger = get_logger(__name__)


@dataclass
class CallContext:
    call_sid: str
    caller_phone: str
    profile: CustomerProfile
    verified: bool = False
    failed_attempts: int = 0
    end_requested: bool = False  # set by end_call / escalate_to_agent; the engine hangs up after the goodbye
    end_reason: str = ""
    transcript: list[tuple[str, str]] = field(default_factory=list)  # (role, text)


class ToolRunner:
    """Bound async methods double as Gemini tools (docstrings + type hints become the schema)."""

    def __init__(self, ctx: CallContext) -> None:
        self.ctx = ctx
        self._settings = get_settings()

    @property
    def callables(self) -> list:
        return [self.get_customer_info, self.verify_caller, self.escalate_to_agent, self.end_call]

    def _locked(self) -> bool:
        return self._settings.require_caller_verification and not self.ctx.verified

    async def get_customer_info(self) -> dict[str, Any]:
        """Return what is known about the current caller: name, preferred language and recent interactions."""
        p = self.ctx.profile
        if self._locked():
            return {"status": "verification_required", "hint": "Ask for the date of birth, then call verify_caller."}
        if not p.known:
            return {"status": "unknown_customer", "hint": "New customer: collect their need, no account data exists."}
        return {"status": "ok", "name": p.name, "language": p.language, "recent_history": p.history[-5:]}

    async def verify_caller(self, birth_date: str) -> dict[str, Any]:
        """Verify the caller's identity from their date of birth, given in ISO format YYYY-MM-DD."""
        expected = self.ctx.profile.raw.get("birth_date")
        if not expected:
            return {"status": "verification_unavailable"}
        if self.ctx.failed_attempts >= self._settings.max_verification_attempts:
            return {"status": "locked", "hint": "Too many failures: call escalate_to_agent."}
        if birth_date.strip() == str(expected):
            self.ctx.verified = True
            return {"status": "verified"}
        self.ctx.failed_attempts += 1
        # Never reveal which detail was wrong.
        return {"status": "failed", "attempts_left": self._settings.max_verification_attempts - self.ctx.failed_attempts}

    async def escalate_to_agent(self, reason: str) -> dict[str, Any]:
        """Request a callback from a human agent (complaint, fraud, hardship, out-of-scope, or caller's wish)."""
        escalation_id = await create_escalation(self.ctx.call_sid, self.ctx.caller_phone, reason)
        if escalation_id is None:
            return {"status": "error", "hint": "Callback could not be recorded; apologise and do not promise one."}
        self.ctx.end_requested, self.ctx.end_reason = True, f"escalated: {reason}"
        return {"status": "ok", "hint": "Tell the caller a human agent will call back, say goodbye briefly."}

    async def end_call(self, reason: str) -> dict[str, Any]:
        """End the call once the caller is done. Say goodbye first; the line closes after your last words."""
        self.ctx.end_requested, self.ctx.end_reason = True, reason
        return {"status": "ok"}

    async def run(self, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
        """Manual dispatch (Live engine). Unknown tools and bad arguments return an error to the model."""
        fn = {f.__name__: f for f in self.callables}.get(name)
        if fn is None:
            return {"status": "error", "hint": f"unknown tool {name}"}
        try:
            result = await fn(**(args or {}))
        except TypeError as exc:
            result = {"status": "error", "hint": f"bad arguments: {exc}"}
        logger.info("tool_called", call_sid=self.ctx.call_sid, tool=name, status=result.get("status"))
        return result
