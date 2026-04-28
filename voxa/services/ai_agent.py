from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import vertexai
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    Part,
)

from core.config import get_settings
from core.logger import get_logger
from services.database import CustomerProfile, get_customer_profile

logger = get_logger(__name__)

_SYSTEM_PROMPT = """Tu es Voxa, un assistant vocal IA professionnel et empathique pour un centre d'appels.
Règles impératives :
- Réponds TOUJOURS en moins de 2 phrases courtes (max 30 mots) pour minimiser la latence TTS.
- Adapte ton ton au profil du client fourni dans le contexte.
- Ne génère JAMAIS de Markdown, balises HTML, ou listes à puces : ta sortie sera lue à voix haute.
- Si tu ne comprends pas, demande poliment de répéter.
- Reste factuel ; ne promets rien que tu ne peux pas tenir."""

_model: GenerativeModel | None = None


def _init_vertex() -> GenerativeModel:
    global _model
    if _model is None:
        settings = get_settings()
        vertexai.init(project=settings.gcp_project_id, location=settings.gcp_location)
        _model = GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=_SYSTEM_PROMPT,
            generation_config=GenerationConfig(
                temperature=settings.gemini_temperature,
                max_output_tokens=settings.gemini_max_output_tokens,
                candidate_count=1,
            ),
        )
        logger.info("vertex_ai_initialized", model=settings.gemini_model)
    return _model


class VoxaAgent:
    """
    Stateful conversational agent for a single call session.

    Lifecycle: one instance per WebSocket / Twilio call.
    The instance holds the chat history so Gemini maintains context.
    """

    def __init__(self, call_sid: str, caller_phone: str) -> None:
        self.call_sid = call_sid
        self.caller_phone = caller_phone
        self._profile: CustomerProfile | None = None
        self._chat: Any = None  # vertexai ChatSession

    # ── Initialisation ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Load customer profile from Firestore, then open a chat session."""
        model = _init_vertex()

        # Fetch customer context (non-blocking, max 3 s)
        self._profile = await get_customer_profile(self.caller_phone)

        # Inject customer context as the first user turn so Gemini has it
        context_turn = Content(
            role="user",
            parts=[Part.from_text(
                f"[CONTEXTE SYSTÈME - ne pas lire à voix haute]\n"
                f"{self._profile.to_context_prompt()}"
            )],
        )
        context_reply = Content(
            role="model",
            parts=[Part.from_text("Contexte reçu. Je suis prêt à répondre.")],
        )

        self._chat = model.start_chat(history=[context_turn, context_reply])
        logger.info(
            "agent_initialized",
            call_sid=self.call_sid,
            customer=self._profile.name,
        )

    # ── Streaming inference ───────────────────────────────────────────────────

    async def stream_response(self, user_text: str) -> AsyncIterator[str]:
        """
        Yield text chunks from Gemini as they arrive (streaming generation).
        Each chunk should be forwarded to the TTS pipeline immediately to
        minimise time-to-first-audio.
        """
        if self._chat is None:
            raise RuntimeError("VoxaAgent.initialize() must be called first.")

        logger.debug("llm_request", call_sid=self.call_sid, text=user_text)

        loop = asyncio.get_running_loop()

        # vertexai streaming is synchronous; run it in a thread pool so we
        # never block the event loop.
        response_stream = await loop.run_in_executor(
            None,
            lambda: self._chat.send_message(user_text, stream=True),
        )

        for chunk in response_stream:
            text = chunk.text
            if text:
                logger.debug("llm_chunk", call_sid=self.call_sid, chunk=text)
                yield text

    async def get_greeting(self) -> AsyncIterator[str]:
        """Generate a personalised opening greeting."""
        name = self._profile.name if self._profile else "Client"
        greeting_prompt = (
            f"Accueille chaleureusement {name} en une phrase courte "
            f"et demande-lui comment tu peux l'aider."
        )
        async for chunk in self.stream_response(greeting_prompt):
            yield chunk

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def close(self) -> None:
        logger.info("agent_closed", call_sid=self.call_sid)
        self._chat = None
