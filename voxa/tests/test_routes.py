from fastapi.testclient import TestClient
from twilio.request_validator import RequestValidator

from app.main import app

client = TestClient(app)
URL = "https://voxa.example.com/voice/incoming"
FORM = {"CallSid": "CA123", "From": "+33612345678", "To": "+15550001111"}


def test_health():
    assert client.get("/status/health").json() == {"status": "ok", "service": "voxa"}


def test_incoming_rejects_unsigned():
    assert client.post("/voice/incoming", data=FORM).status_code == 403


def test_incoming_returns_twiml_stream():
    sig = RequestValidator("test-token").compute_signature(URL, FORM)
    r = client.post("/voice/incoming", data=FORM, headers={"X-Twilio-Signature": sig})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/xml")
    assert 'url="wss://voxa.example.com/voice/stream"' in r.text
    assert 'name="caller"' in r.text and 'value="+33612345678"' in r.text and 'name="token"' in r.text
