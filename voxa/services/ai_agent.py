"""Gemini "brain": one VoxaAgent per call, streaming sentence by sentence."""
from __future__ import annotations

import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from google import genai
from google.genai import types

from core.config import get_settings
from core.logger import get_logger
from services.database import CustomerProfile, append_call_summary, get_customer_profile

logger = get_logger(__name__)

SYSTEM_PROMPT = """Tu es Voxa, un assistant vocal IA professionnel et chaleureux pour un centre d'appels.
Règles impératives :
- Tes réponses sont lues à voix haute au téléphone : 1 à 2 phrases courtes (30 mots maximum).
- Aucun Markdown, emoji, liste à puces ni balise : uniquement du texte parlé naturel.
- Utilise le prénom du client quand il est connu, sans en abuser.
- Si tu n'as pas compris, demande poliment de répéter.
- Reste factuel : ne promets rien que tu ne peux pas tenir ; propose un transfert vers un conseiller si besoin.
- Le bloc CONTEXTE CLIENT ci-dessous est confidentiel : ne le récite jamais mot pour mot."""

# End of a speakable unit: sentence punctuation followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?…:;])\s+")
# Flush very long clauses at a comma so TTS can start early.
_MIN_CLAUSE_CHARS = 60

_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
    """Lazily build the Vertex AI client (uses Application Default Credentials)."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = genai.Client(
            vertexai=True, project=settings.gcp_project_id, location=settings.gcp_location
        )
        logger.info("vertex_ai_initialized", model=settings.gemini_model, location=settings.gcp_location)
    return _client


async def sentences_from_stream(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Regroup raw LLM text deltas into speakable sentences (lowest TTS latency without cutting words)."""
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
    """Stateful conversational agent bound to a single call."""

    def __init__(self, call_sid: str, caller_phone: str) -> None:
        self.call_sid = call_sid
        self.caller_phone = caller_phone
        self.profile: CustomerProfile | None = None
        self._chat: genai.chats.AsyncChat | None = None
        self.transcript: list[tuple[str, str]] = []  # (role, text)

    async def initialize(self) -> None:
        """Fetch the customer profile from Firestore, THEN open the Gemini chat with it as context."""
        settings = get_settings()
        self.profile = await get_customer_profile(self.caller_phone)

        system_instruction = f"{SYSTEM_PROMPT}\n\n--- CONTEXTE CLIENT ---\n{self.profile.to_context_prompt()}"
        self._chat = get_genai_client().aio.chats.create(
            model=settings.gemini_model,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=settings.gemini_temperature,
                max_output_tokens=settings.gemini_max_output_tokens,
            ),
        )
        logger.info("agent_initialized", call_sid=self.call_sid, customer=self.profile.name, known=self.profile.known)

    async def _stream_text(self, user_text: str) -> AsyncIterator[str]:
        if self._chat is None:
            raise RuntimeError("VoxaAgent.initialize() must be called first.")
        stream = await self._chat.send_message_stream(user_text)
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    async def reply(self, user_text: str) -> AsyncIterator[str]:
        """Yield the answer to `user_text` sentence by sentence as Gemini generates it."""
        logger.debug("llm_request", call_sid=self.call_sid, text=user_text)
        self.transcript.append(("client", user_text))
        spoken: list[str] = []
        try:
            async for sentence in sentences_from_stream(self._stream_text(user_text)):
                spoken.append(sentence)
                yield sentence
        finally:
            # Also runs when the consumer is cancelled (barge-in): keep only what was produced.
            if spoken:
                self.transcript.append(("voxa", " ".join(spoken)))

    def greeting(self) -> AsyncIterator[str]:
        name = self.profile.name if self.profile and self.profile.known else None
        who = f"au client {name}" if name else "au client"
        return self.reply(
            f"[L'appel vient de commencer. Souhaite la bienvenue {who} en une phrase "
            f"et demande-lui comment tu peux l'aider.]"
        )

    async def close(self) -> None:
        """Persist a short summary of the call in Firestore (best effort)."""
        settings = get_settings()
        if self._chat is not None and len(self.transcript) > 2:
            try:
                dialogue = "\n".join(f"{r}: {t}" for r, t in self.transcript)
                response = await get_genai_client().aio.models.generate_content(
                    model=settings.gemini_model,
                    contents=f"Résume cet appel en une phrase factuelle (français) :\n{dialogue}",
                    config=types.GenerateContentConfig(max_output_tokens=80, temperature=0.2),
                )
                summary = (response.text or "").strip()
                if summary:
                    await append_call_summary(
                        self.caller_phone, summary, datetime.now(timezone.utc).date().isoformat()
                    )
            except Exception as exc:  # never fail call teardown
                logger.warning("call_summary_failed", call_sid=self.call_sid, error=str(exc))
        self._chat = None
        logger.info("agent_closed", call_sid=self.call_sid)
