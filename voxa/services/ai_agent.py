"""Voxa's "brain": call context from Firestore, prompt assembly, Gemini text chat (cascade engine), summaries."""
from __future__ import annotations

import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from google import genai
from google.genai import types

from core.config import get_settings
from core.logger import get_logger
from services.database import append_call_summary, get_customer_profile
from services.prompts import SYSTEM_PROMPT
from services.tools import CallContext, ToolRunner

logger = get_logger(__name__)

# End of a speakable unit: sentence punctuation followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?…:;])\s+")
_MIN_CLAUSE_CHARS = 60  # flush a long clause at a comma so TTS can start early

_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
    """Lazily build the Vertex AI client (Application Default Credentials)."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = genai.Client(vertexai=True, project=settings.gcp_project_id, location=settings.gcp_location)
        logger.info("vertex_ai_initialized", location=settings.gcp_location)
    return _client


async def build_call_context(call_sid: str, caller_phone: str) -> CallContext:
    """Fetch the customer profile from Firestore BEFORE the model says anything."""
    profile = await get_customer_profile(caller_phone)
    logger.info("call_context_ready", call_sid=call_sid, customer=profile.name, known=profile.known)
    return CallContext(call_sid=call_sid, caller_phone=caller_phone, profile=profile)


def build_system_instruction(ctx: CallContext) -> str:
    return f"{SYSTEM_PROMPT}\n\n--- CONTEXTE CLIENT ---\n{ctx.profile.to_context_prompt()}"


def greeting_text(ctx: CallContext) -> str:
    """Static greeting: spoken without any LLM round-trip, so the caller hears Voxa immediately."""
    p = ctx.profile
    hello = f"Bonjour {p.name}, ici Voxa." if p.known else "Bonjour, ici Voxa."
    return f"{hello} Comment puis-je vous aider ?"


async def summarize_and_save(ctx: CallContext) -> None:
    """Persist a one-line call summary in Firestore (best effort, never raises)."""
    if len(ctx.transcript) < 2:
        return
    try:
        dialogue = "\n".join(f"{role}: {text}" for role, text in ctx.transcript)
        response = await get_genai_client().aio.models.generate_content(
            model=get_settings().gemini_model,
            contents=f"Résume cet appel en une phrase factuelle (français) :\n{dialogue}",
            config=types.GenerateContentConfig(max_output_tokens=80, temperature=0.2),
        )
        summary = (response.text or "").strip()
        if ctx.end_reason:
            summary = f"{summary} [{ctx.end_reason}]"
        if summary:
            await append_call_summary(ctx.caller_phone, summary, datetime.now(timezone.utc).date().isoformat())
    except Exception as exc:
        logger.warning("call_summary_failed", call_sid=ctx.call_sid, error=str(exc))


async def sentences_from_stream(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Regroup raw LLM deltas into speakable sentences (lowest TTS latency without cutting words)."""
    buffer = ""
    async for delta in chunks:
        buffer += delta
        while True:
            match = _SENTENCE_END.search(buffer)
            if match is None and len(buffer) >= _MIN_CLAUSE_CHARS and "," in buffer:
                cut = buffer.rfind(",") + 1
                sentence, buffer = buffer[:cut], buffer[cut:]
            elif match is not None:
                sentence, buffer = buffer[: match.start()], buffer[match.end():]
            else:
                break
            if sentence.strip():
                yield sentence.strip()
    if buffer.strip():
        yield buffer.strip()


class VoxaAgent:
    """Text chat session for the cascade engine (tools are executed automatically by the SDK)."""

    def __init__(self, ctx: CallContext, tools: ToolRunner) -> None:
        self.ctx = ctx
        self._tools = tools
        self._chat: genai.chats.AsyncChat | None = None

    async def initialize(self) -> None:
        settings = get_settings()
        greeting = greeting_text(self.ctx)
        self._chat = get_genai_client().aio.chats.create(
            model=settings.gemini_model,
            config=types.GenerateContentConfig(
                system_instruction=build_system_instruction(self.ctx),
                temperature=settings.gemini_temperature,
                max_output_tokens=settings.gemini_max_output_tokens,
                tools=self._tools.callables,  # async bound methods → schema from signature + docstring
            ),
            # The greeting is already spoken: put it in the history so the model knows.
            history=[
                types.Content(role="user", parts=[types.Part(text="[L'appel commence]")]),
                types.Content(role="model", parts=[types.Part(text=greeting)]),
            ],
        )
        self.ctx.transcript.append(("voxa", greeting))

    async def _stream_text(self, user_text: str) -> AsyncIterator[str]:
        assert self._chat is not None, "initialize() first"
        stream = await self._chat.send_message_stream(user_text)
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    async def reply(self, user_text: str) -> AsyncIterator[str]:
        """Yield the answer sentence by sentence as Gemini generates it."""
        self.ctx.transcript.append(("client", user_text))
        spoken: list[str] = []
        try:
            async for sentence in sentences_from_stream(self._stream_text(user_text)):
                spoken.append(sentence)
                yield sentence
        finally:  # also runs on cancellation (barge-in): keep only what was produced
            if spoken:
                self.ctx.transcript.append(("voxa", " ".join(spoken)))
