import base64

import pytest
from pydantic import ValidationError

from core.config import Settings
from core.security import make_stream_token, verify_stream_token
from services.ai_agent import sentences_from_stream
from services.audio_stream import chunk_audio, mulaw_rms, strip_wav_header


def test_config_rejects_bad_sid(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "XX" + "a" * 32)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_config_ws_url():
    assert Settings(_env_file=None).stream_ws_url == "wss://voxa.example.com/voice/stream"


def test_stream_token_roundtrip():
    token = make_stream_token("CA123")
    assert verify_stream_token("CA123", token)
    assert not verify_stream_token("CA999", token)
    assert not verify_stream_token("CA123", "")


def test_strip_wav_header_and_chunking():
    wav = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 16 + b"data\x04\x00\x00\x00" + b"\x01\x02\x03\x04"
    assert strip_wav_header(wav) == b"\x01\x02\x03\x04"
    assert strip_wav_header(b"raw") == b"raw"
    assert [len(c) for c in chunk_audio(b"x" * 7000, 3200)] == [3200, 3200, 600]


def test_mulaw_rms_silence_vs_loud():
    assert mulaw_rms(b"\xff" * 160) < 50
    assert mulaw_rms(b"\x00" * 160) > 5000


async def _aiter(items):
    for i in items:
        yield i


async def test_sentences_from_stream_regroups_deltas():
    out = [s async for s in sentences_from_stream(_aiter(["Bonjour Marie", ". Comment ", "puis-je vous ", "aider ?"]))]
    assert out == ["Bonjour Marie.", "Comment puis-je vous aider ?"]
