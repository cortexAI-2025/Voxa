from __future__ import annotations

import asyncio
from typing import Any

from google.cloud import firestore
from google.cloud.firestore_v1.async_client import AsyncClient

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

_client: AsyncClient | None = None


def _get_client() -> AsyncClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = firestore.AsyncClient(project=settings.gcp_project_id)
    return _client


class CustomerProfile:
    def __init__(self, data: dict[str, Any]) -> None:
        self.customer_id: str = data.get("customer_id", "")
        self.name: str = data.get("name") or "Client"
        self.phone: str = data.get("phone", "")
        self.language: str = data.get("language", "fr-FR")
        self.history: list[dict[str, Any]] = data.get("history", [])
        self.preferences: dict[str, Any] = data.get("preferences", {})
        self.raw: dict[str, Any] = data
        # Unknown callers get a stub profile with only a phone number.
        self.known: bool = bool(data.get("customer_id"))

    def to_context_prompt(self) -> str:
        history_lines = "\n".join(
            f"- [{h.get('date', '')}] {h.get('summary', '')}"
            for h in self.history[-5:]  # last 5 interactions only
        )
        return (
            f"Client : {self.name if self.known else 'inconnu (nouveau client)'}\n"
            f"Langue préférée : {self.language}\n"
            f"Historique récent :\n{history_lines or '  Aucun historique.'}"
        )


async def get_customer_profile(phone_number: str) -> CustomerProfile:
    settings = get_settings()
    client = _get_client()
    collection = settings.firestore_collection_customers

    try:
        # Normalise E.164 → document ID (strip leading +)
        doc_id = phone_number.lstrip("+")
        doc_ref = client.collection(collection).document(doc_id)
        snapshot = await asyncio.wait_for(doc_ref.get(), timeout=3.0)

        if snapshot.exists:
            data = snapshot.to_dict() or {}
            data["customer_id"] = snapshot.id
            logger.info("customer_profile_loaded", phone=phone_number, name=data.get("name"))
            return CustomerProfile(data)

        logger.info("customer_not_found", phone=phone_number)
        return CustomerProfile({"phone": phone_number})

    except asyncio.TimeoutError:
        logger.warning("firestore_timeout", phone=phone_number)
        return CustomerProfile({"phone": phone_number})
    except Exception as exc:
        logger.error("firestore_error", phone=phone_number, error=str(exc))
        return CustomerProfile({"phone": phone_number})


async def append_call_summary(phone_number: str, summary: str, date: str) -> None:
    settings = get_settings()
    client = _get_client()
    collection = settings.firestore_collection_customers
    doc_id = phone_number.lstrip("+")
    doc_ref = client.collection(collection).document(doc_id)

    try:
        await asyncio.wait_for(
            doc_ref.set(
                {
                    "history": firestore.ArrayUnion(
                        [{"date": date, "summary": summary}]
                    )
                },
                merge=True,
            ),
            timeout=5.0,
        )
        logger.info("call_summary_saved", phone=phone_number)
    except Exception as exc:
        logger.error("firestore_write_error", phone=phone_number, error=str(exc))
