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
    gcp_project_id: str = Field(..., description="GCP project ID")
    gcp_location: str = Field(default="us-central1")
    google_application_credentials: str | None = Field(default=None)

    # ── Gemini / Vertex AI ────────────────────────────────────────────────────
    gemini_model: str = Field(default="gemini-2.0-flash-001")
    gemini_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    gemini_max_output_tokens: int = Field(default=1024, gt=0, le=8192)

    # ── Firestore ─────────────────────────────────────────────────────────────
    firestore_collection_customers: str = Field(default="customers")

    # ── Twilio ────────────────────────────────────────────────────────────────
    twilio_account_sid: str = Field(..., min_length=34, max_length=34)
    twilio_auth_token: str = Field(..., min_length=1)
    twilio_phone_number: str = Field(..., pattern=r"^\+\d{7,15}$")

    # ── App ───────────────────────────────────────────────────────────────────
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8080, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    public_base_url: str = Field(..., description="Publicly reachable base URL (no trailing slash)")

    # ── Audio ─────────────────────────────────────────────────────────────────
    audio_sample_rate_in: int = Field(default=8000)
    audio_sample_rate_out: int = Field(default=8000)
    barge_in_energy_threshold: int = Field(default=300, ge=0, le=32768)
    barge_in_silence_frames: int = Field(default=20, gt=0)

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("twilio_account_sid")
    @classmethod
    def validate_account_sid(cls, v: str) -> str:
        if not v.startswith("AC"):
            raise ValueError("TWILIO_ACCOUNT_SID must start with 'AC'")
        return v

    @field_validator("public_base_url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @model_validator(mode="after")
    def validate_gcp_credentials(self) -> "Settings":
        import os
        if self.google_application_credentials:
            if not os.path.isfile(self.google_application_credentials):
                raise ValueError(
                    f"GOOGLE_APPLICATION_CREDENTIALS file not found: "
                    f"{self.google_application_credentials}"
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
