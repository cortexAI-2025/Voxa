import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Google Cloud ──────────────────────────────────────────────────────────
    gcp_project_id: str = Field(..., min_length=1, description="GCP project ID")
    gcp_location: str = Field(default="us-central1")
    google_application_credentials: str | None = Field(
        default=None,
        description="Service account key path (local dev). On Cloud Run use the attached service account.",
    )

    # ── Gemini / Vertex AI ────────────────────────────────────────────────────
    gemini_model: str = Field(default="gemini-2.0-flash")
    gemini_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    gemini_max_output_tokens: int = Field(default=256, gt=0, le=8192)

    # ── Voice engine ──────────────────────────────────────────────────────────
    # "live"    : Gemini Live native audio (audio in → audio out, lowest latency, server-side VAD/barge-in)
    # "cascade" : Google STT → Gemini text → Google TTS (full control, used as automatic fallback)
    voice_engine: Literal["live", "cascade"] = Field(default="live")
    live_model: str = Field(default="gemini-live-2.5-flash-native-audio")
    live_voice: str = Field(default="Aoede")
    live_language_code: str = Field(default="fr-FR")

    # ── Speech (STT / TTS, cascade engine) ────────────────────────────────────
    stt_language_code: str = Field(default="fr-FR")
    tts_language_code: str = Field(default="fr-FR")
    tts_voice_name: str = Field(default="fr-FR-Neural2-A")

    # ── Firestore ─────────────────────────────────────────────────────────────
    firestore_collection_customers: str = Field(default="customers")

    # ── Twilio ────────────────────────────────────────────────────────────────
    twilio_account_sid: str = Field(..., min_length=34, max_length=34)
    twilio_auth_token: str = Field(..., min_length=1)
    twilio_phone_number: str = Field(..., pattern=r"^\+\d{7,15}$")
    validate_twilio_signature: bool = Field(
        default=True, description="Reject webhooks not signed by Twilio. Disable only for local tests."
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8080, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    public_base_url: str = Field(..., description="Publicly reachable base URL (https://…, no trailing slash)")

    # ── Call policy ───────────────────────────────────────────────────────────
    max_call_seconds: int = Field(default=900, gt=0, description="Hard limit; the call is ended politely after this")
    goodbye_grace_seconds: float = Field(default=8.0, gt=0, description="Max wait for goodbye audio before hanging up")
    require_caller_verification: bool = Field(
        default=False, description="Withhold history / account data until verify_caller succeeds"
    )
    max_verification_attempts: int = Field(default=3, ge=1)
    firestore_collection_escalations: str = Field(default="escalations")

    # ── Audio / barge-in (cascade engine) ──────────────────────────────────────────────────────
    # Twilio Media Streams are µ-law 8 kHz mono, 20 ms (160 bytes) per frame.
    barge_in_energy_threshold: int = Field(default=500, ge=0, le=32768)
    barge_in_min_frames: int = Field(
        default=3, ge=1, description="Consecutive loud 20 ms frames required to trigger barge-in"
    )

    barge_in_min_chars: int = Field(
        default=3, ge=0, description="Interim transcript must have this many characters to interrupt (ignores 'mm')"
    )

    @field_validator("twilio_account_sid")
    @classmethod
    def _validate_account_sid(cls, v: str) -> str:
        if not v.startswith("AC"):
            raise ValueError("TWILIO_ACCOUNT_SID must start with 'AC'")
        return v

    @field_validator("public_base_url")
    @classmethod
    def _normalize_public_url(cls, v: str) -> str:
        v = v.rstrip("/")
        if not v.startswith(("https://", "http://")):
            raise ValueError("PUBLIC_BASE_URL must start with https:// or http://")
        return v

    @model_validator(mode="after")
    def _validate_gcp_credentials(self) -> "Settings":
        path = self.google_application_credentials
        if path:
            if not os.path.isfile(path):
                raise ValueError(f"GOOGLE_APPLICATION_CREDENTIALS file not found: {path}")
            # google-auth reads the env var, not our settings object.
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", path)
        return self

    @property
    def stream_ws_url(self) -> str:
        return self.public_base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/voice/stream"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
