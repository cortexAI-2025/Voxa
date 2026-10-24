import asyncio

from google.genai import types

from services.engines.live import LiveEngine
from tests.fakes import FakeTransport, make_tools, settle


class FakeLiveSession:
    def __init__(self):
        self.q: asyncio.Queue = asyncio.Queue()
        self.inputs: list[dict] = []
        self.tool_responses: list = []

    async def send_realtime_input(self, **kw): self.inputs.append(kw)
    async def send_tool_response(self, *, function_responses): self.tool_responses.extend(function_responses)

    async def receive(self):
        while True:
            msg = await self.q.get()
            yield msg
            if msg.server_content and msg.server_content.turn_complete:
                return


def audio_msg(n=480):
    part = types.Part(inline_data=types.Blob(data=b"\x00\x00" * n, mime_type="audio/pcm;rate=24000"))
    return types.LiveServerMessage(server_content=types.LiveServerContent(model_turn=types.Content(parts=[part])))


def done_msg():
    return types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))


async def make_engine(monkeypatch):
    tr = FakeTransport()
    ctx, tools = make_tools()
    finished, fatal = [], []
    session = FakeLiveSession()

    async def finish(): finished.append(1)
    async def on_fatal(): fatal.append(1)

    e = LiveEngine(tr, ctx, tools, finish, on_fatal)

    async def fake_connect(): e._session = session
    monkeypatch.setattr(e, "_connect", fake_connect)
    await e.start()
    return e, tr, ctx, session, finished, fatal


async def test_start_asks_model_to_say_the_static_greeting(monkeypatch):
    e, _, _, session, *_ = await make_engine(monkeypatch)
    assert "Bonjour Marie" in session.inputs[0]["text"]
    await e.close()


async def test_caller_audio_is_upsampled_to_16k_pcm(monkeypatch):
    e, _, _, session, *_ = await make_engine(monkeypatch)
    await e.on_audio(b"\xff" * 160)
    blob = session.inputs[-1]["audio"]
    assert blob.mime_type == "audio/pcm;rate=16000" and 600 <= len(blob.data) <= 680
    await e.close()


async def test_model_audio_is_streamed_then_marked(monkeypatch):
    e, tr, _, session, *_ = await make_engine(monkeypatch)
    session.q.put_nowait(audio_msg())
    session.q.put_nowait(done_msg())
    await settle()
    assert tr.of("audio") and e.speaking
    assert tr.of("mark")[-1][1] == f"turn-{e._turn}"
    e.on_mark(f"turn-{e._turn}")
    assert not e.speaking
    await e.close()


async def test_server_interruption_flushes_twilio(monkeypatch):
    e, tr, _, session, *_ = await make_engine(monkeypatch)
    session.q.put_nowait(audio_msg())
    await settle()
    session.q.put_nowait(types.LiveServerMessage(server_content=types.LiveServerContent(interrupted=True)))
    await settle()
    assert len(tr.of("clear")) == 1 and not e.speaking
    await e.close()


async def test_end_call_tool_hangs_up_after_goodbye_is_played(monkeypatch):
    e, tr, ctx, session, finished, _ = await make_engine(monkeypatch)
    call = types.FunctionCall(id="1", name="end_call", args={"reason": "client satisfait"})
    session.q.put_nowait(types.LiveServerMessage(tool_call=types.LiveServerToolCall(function_calls=[call])))
    session.q.put_nowait(audio_msg())  # the goodbye
    session.q.put_nowait(done_msg())
    await settle(0.1)
    assert session.tool_responses[0].response["status"] == "ok" and ctx.end_requested
    assert ("mark", "final") in tr.events and not finished
    e.on_mark("final")
    await settle()
    assert finished
    await e.close()


async def test_transcripts_are_collected_for_the_summary(monkeypatch):
    e, _, ctx, session, *_ = await make_engine(monkeypatch)
    session.q.put_nowait(types.LiveServerMessage(server_content=types.LiveServerContent(
        input_transcription=types.Transcription(text="mon solde ?"),
        output_transcription=types.Transcription(text="Votre solde est de cent euros."))))
    session.q.put_nowait(done_msg())
    await settle()
    assert ("client", "mon solde ?") in ctx.transcript and ("voxa", "Votre solde est de cent euros.") in ctx.transcript
    await e.close()
