import pytest

import services.tools as tools_mod
from tests.fakes import make_ctx, make_tools


def strict(runner):
    runner._settings = runner._settings.model_copy(update={"require_caller_verification": True})
    return runner


async def test_customer_info_open_when_verification_not_required():
    _, t = make_tools()
    info = await t.get_customer_info()
    assert info["status"] == "ok" and info["name"] == "Marie"


async def test_data_is_locked_until_verified():
    ctx, t = make_tools()
    strict(t)
    assert (await t.get_customer_info())["status"] == "verification_required"
    assert (await t.verify_caller("1990-05-14"))["status"] == "verified"
    assert (await t.get_customer_info())["status"] == "ok"


async def test_verification_lockout_and_no_hint_about_which_detail_failed():
    ctx, t = make_tools()
    results = [await t.verify_caller("2000-01-01") for _ in range(4)]
    assert [r["status"] for r in results] == ["failed", "failed", "failed", "locked"]
    assert (await t.verify_caller("1990-05-14"))["status"] == "locked"  # even the right answer is refused
    assert not ctx.verified


async def test_unknown_caller_gets_no_account_data():
    _, t = make_tools(make_ctx(customer_id=""))
    assert (await t.get_customer_info())["status"] == "unknown_customer"


async def test_tools_take_no_customer_id_so_model_cannot_pick_another_customer():
    _, t = make_tools()
    r = await t.run("get_customer_info", {"customer_id": "someone-else"})
    assert r["status"] == "error"  # unexpected argument is rejected, not honoured


async def test_escalation_marks_call_for_ending_only_on_success(monkeypatch):
    ctx, t = make_tools()
    async def fail(*a): return None
    monkeypatch.setattr(tools_mod, "create_escalation", fail)
    assert (await t.escalate_to_agent("fraude"))["status"] == "error"
    assert not ctx.end_requested

    async def ok(*a): return "esc1"
    monkeypatch.setattr(tools_mod, "create_escalation", ok)
    assert (await t.escalate_to_agent("fraude"))["status"] == "ok"
    assert ctx.end_requested and "fraude" in ctx.end_reason


async def test_unknown_tool_returns_error():
    _, t = make_tools()
    assert (await t.run("drop_database", {}))["status"] == "error"
