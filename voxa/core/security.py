"""Request authentication for Twilio webhooks and Media Stream WebSockets."""
import hashlib
import hmac

from twilio.request_validator import RequestValidator

from core.config import get_settings


def is_valid_twilio_request(url: str, params: dict[str, str], signature: str | None) -> bool:
    """Check the X-Twilio-Signature header of a webhook (HTTP) request."""
    settings = get_settings()
    if not settings.validate_twilio_signature:
        return True
    if not signature:
        return False
    return RequestValidator(settings.twilio_auth_token).validate(url, params, signature)


def make_stream_token(call_sid: str) -> str:
    """HMAC bound to a CallSid, handed to Twilio as a <Parameter> and checked on the WebSocket `start` event.

    Twilio does not allow a query string on <Stream url>, and WebSockets carry no form body to sign,
    so this token is what proves the stream belongs to a call that passed webhook validation.
    """
    key = get_settings().twilio_auth_token.encode()
    return hmac.new(key, call_sid.encode(), hashlib.sha256).hexdigest()[:32]


def verify_stream_token(call_sid: str, token: str) -> bool:
    if not get_settings().validate_twilio_signature:
        return True
    return hmac.compare_digest(make_stream_token(call_sid), token or "")
