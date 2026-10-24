import services.call_session as cs
from core.security import make_stream_token
from tests.fakes import FakeTransport, make_ctx

START = {"streamSid": "MZ1", "callSid": "CA1", "customParameters": {"caller": "+33600000000", "token": make_stream_token("CA1")}}


class DeadLive:
    def __init__(self, *a): ...
    async def start(self): raise RuntimeError("vertex unavailable")
    async def close(self): ...


class OkCascade:
    started_with = "unset"

    def __init__(self, *a): ...

    async def start(self, announce=None): OkCascade.started_with = announce
    async def close(self): ...
    async def on_audio(self, f): ...
    def on_mark(self, n): ...


def patch(monkeypatch):
    async def ctx_factory(sid, phone): return make_ctx()
    async def no_summary(ctx): ...
    monkeypatch.setattr(cs, "build_call_context", ctx_factory)
    monkeypatch.setattr(cs, "summarize_and_save", no_summary)
    monkeypatch.setattr(cs, "CascadeEngine", OkCascade)


async def test_bad_token_is_rejected():
    s = cs.CallSession(FakeTransport())
    assert not await s.on_start({**START, "customParameters": {"token": "nope"}})


async def test_live_failure_at_start_falls_back_to_cascade(monkeypatch):
    patch(monkeypatch)
    monkeypatch.setattr(cs, "LiveEngine", DeadLive)
    s = cs.CallSession(FakeTransport())
    assert await s.on_start(START)
    assert s.engine_name == "cascade"
    await s.close()


async def test_live_dying_mid_call_switches_to_cascade_with_an_apology(monkeypatch):
    patch(monkeypatch)
    s = cs.CallSession(FakeTransport())
    s.ctx, s._tools = make_ctx(), object()
    await s._switch_to_cascade()
    assert s.engine_name == "cascade" and OkCascade.started_with == cs.FALLBACK_NOTICE
    await s._switch_to_cascade()  # idempotent
    await s.close()


async def test_max_duration_hangs_up(monkeypatch):
    patch(monkeypatch)
    tr = FakeTransport()
    s = cs.CallSession(tr)
    s._settings = s._settings.model_copy(update={"max_call_seconds": 1})
    s.ctx = make_ctx()
    import asyncio
    s._watchdog = asyncio.create_task(s._enforce_max_duration())
    await asyncio.sleep(1.1)
    assert ("hangup",) in tr.events
    await s.close()
