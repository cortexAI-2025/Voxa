import asyncio

from services.engines.cascade import CascadeEngine
from tests.fakes import FakeTransport, make_ctx, make_tools, settle

LOUD = b"\x00" * 160
QUIET = b"\xff" * 160


class FakeAgent:
    def __init__(self, ctx, tools): self.replies = []
    async def initialize(self): ...

    async def reply(self, text):
        self.replies.append(text)
        yield "Bien sûr."


class FakeSTT:
    def __init__(self, lang): self.q = asyncio.Queue()
    def start(self): ...
    def feed(self, frame): ...
    async def close(self): ...

    async def results(self):
        while (item := await self.q.get()) is not None:
            yield item


class FakeTTS:
    async def synthesize(self, text, cache=False): return b"\xff" * 4000


async def make_engine():
    tr = FakeTransport()
    ctx, tools = make_tools()
    finished = []

    async def finish(): finished.append(1)

    e = CascadeEngine(tr, ctx, tools, finish, agent_factory=FakeAgent, transcriber_factory=FakeSTT, synthesizer_factory=FakeTTS)
    await e.start()
    await settle()
    return e, tr, ctx, finished


async def test_greeting_is_static_personalised_and_marked():
    e, tr, ctx, _ = await make_engine()
    assert len(tr.of("audio")) == 2  # 4000 B → two 3200-B chunks max
    assert tr.of("mark")[0][1] == f"turn-{e._turn}"
    assert e.speaking
    e.on_mark(f"turn-{e._turn}")
    assert not e.speaking
    await e.close()


async def test_barge_in_needs_min_frames_then_clears_and_stops_audio():
    e, tr, *_ = await make_engine()
    await e.on_audio(LOUD)
    await e.on_audio(LOUD)
    assert not tr.of("clear")  # 2 < min_frames: no false trigger
    await e.on_audio(LOUD)
    assert len(tr.of("clear")) == 1 and not e.speaking
    n = len(tr.of("audio"))
    await settle()
    assert len(tr.of("audio")) == n
    await e.close()


async def test_quiet_frames_do_not_interrupt():
    e, tr, *_ = await make_engine()
    for _ in range(10):
        await e.on_audio(QUIET)
    assert not tr.of("clear")
    await e.close()


async def test_final_transcript_triggers_reply():
    e, tr, *_ = await make_engine()
    e.on_mark(f"turn-{e._turn}")
    before = len(tr.of("audio"))
    e._stt.q.put_nowait(("je voudrais un rendez-vous", True))
    await settle(0.1)
    assert e.agent.replies == ["je voudrais un rendez-vous"]
    assert len(tr.of("audio")) > before
    await e.close()


async def test_interim_backchannel_is_ignored_but_real_speech_interrupts():
    e, tr, *_ = await make_engine()
    e._stt.q.put_nowait(("mm", False))
    await settle()
    assert not tr.of("clear")
    e._stt.q.put_nowait(("attendez je voulais", False))
    await settle()
    assert len(tr.of("clear")) == 1
    await e.close()


async def test_goodbye_hangs_up_only_after_final_mark():
    e, tr, ctx, finished = await make_engine()
    ctx.end_requested = True
    e._stt.q.put_nowait(("merci au revoir", True))
    await settle(0.1)
    assert ("mark", "final") in tr.events and not finished  # audio sent, but not yet played
    e.on_mark("final")
    await settle()
    assert finished
    await e.close()
