import asyncio

from services.database import CustomerProfile
from services.tools import CallContext, ToolRunner


class FakeTransport:
    def __init__(self):
        self.call_sid, self.stream_sid = "CA1", "MZ1"
        self.events: list[tuple] = []

    async def send_audio(self, mulaw): self.events.append(("audio", mulaw))
    async def clear(self): self.events.append(("clear",))
    async def mark(self, name): self.events.append(("mark", name))
    async def hangup(self): self.events.append(("hangup",))

    def of(self, kind):
        return [e for e in self.events if e[0] == kind]


def make_ctx(**profile) -> CallContext:
    data = {"customer_id": "33600000000", "name": "Marie", "birth_date": "1990-05-14", **profile}
    return CallContext("CA1", "+33600000000", CustomerProfile(data))


def make_tools(ctx=None):
    ctx = ctx or make_ctx()
    return ctx, ToolRunner(ctx)


async def settle(delay=0.05):
    await asyncio.sleep(delay)
