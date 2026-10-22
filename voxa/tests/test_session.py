import asyncio
import base64
import json

from core.security import make_stream_token
from services.audio_stream import CallSession

LOUD = base64.b64encode(b"\x00" * 160).decode()
QUIET = base64.b64encode(b"\xff" * 160).decode()


class FakeWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    def events(self, name):
        return [m for m in self.sent if m["event"] == name]


class FakeAgent:
    def __init__(self, call_sid, caller):
        self.replies: list[str] = []

    async def initialize(self): ...
    async def close(self): ...

    async def _say(self, sentences, delay=0.0):
        for s in sentences:
            await asyncio.sleep(delay)
            yield s

    def greeting(self):
        return self._say(["Bonjour.", "Comment puis-je vous aider ?"])

    def reply(self, text):
        self.replies.append(text)
        return self._say(["Bien sûr."])


class FakeSTT:
    def __init__(self, lang):
        self.q: asyncio.Queue = asyncio.Queue()
        self.frames = 0

    def start(self): ...
    def feed(self, frame): self.frames += 1
    async def close(self): ...

    async def results(self):
        while (item := await self.q.get()) is not None:
            yield item


class FakeTTS:
    async def synthesize(self, text):
        return b"\xff" * 4000


def make_session(ws):
    return CallSession(ws, agent_factory=FakeAgent, transcriber_factory=FakeSTT, synthesizer_factory=FakeTTS)


async def start(session, sid="CA1"):
    ok = await session.on_start(
        {"streamSid": "MZ1", "callSid": sid, "customParameters": {"caller": "+33600000000", "token": make_stream_token(sid)}}
    )
    await asyncio.sleep(0.05)
    return ok


async def test_greeting_is_streamed_and_marked():
    ws = FakeWS()
    s = make_session(ws)
    assert await start(s)
    assert len(ws.events("media")) == 4  # 2 sentences × 4000 B → 2 chunks each
    assert ws.events("mark")[0]["mark"]["name"] == f"turn-{s._turn}"
    assert s.speaking
    s.on_mark(f"turn-{s._turn}")
    assert not s.speaking
    await s.close()


async def test_bad_token_is_rejected():
    s = make_session(FakeWS())
    assert not await s.on_start({"streamSid": "MZ1", "callSid": "CA1", "customParameters": {"token": "nope"}})
    await s.close()


async def test_barge_in_clears_twilio_and_stops_audio():
    ws = FakeWS()
    s = make_session(ws)
    await start(s)
    assert s.speaking
    await s.on_media(LOUD)
    await s.on_media(LOUD)
    assert not ws.events("clear")  # 2 frames < min_frames (3): no false trigger
    await s.on_media(LOUD)
    assert len(ws.events("clear")) == 1
    assert not s.speaking
    n = len(ws.events("media"))
    await asyncio.sleep(0.05)
    assert len(ws.events("media")) == n  # nothing is sent after the clear
    await s.close()


async def test_quiet_frames_do_not_interrupt():
    ws = FakeWS()
    s = make_session(ws)
    await start(s)
    for _ in range(10):
        await s.on_media(QUIET)
    assert not ws.events("clear")
    await s.close()


async def test_final_transcript_triggers_reply():
    ws = FakeWS()
    s = make_session(ws)
    await start(s)
    s.on_mark(f"turn-{s._turn}")
    before = len(ws.events("media"))
    s._stt.q.put_nowait(("je voudrais un rendez-vous", True))
    await asyncio.sleep(0.1)
    assert s.agent.replies == ["je voudrais un rendez-vous"]
    assert len(ws.events("media")) > before
    await s.close()


async def test_interim_transcript_while_speaking_interrupts():
    ws = FakeWS()
    s = make_session(ws)
    await start(s)
    s._stt.q.put_nowait(("allô", False))
    await asyncio.sleep(0.05)
    assert len(ws.events("clear")) == 1
    await s.close()
