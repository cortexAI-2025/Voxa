# Voxa — AI voice agent

Inbound phone calls (Twilio Media Streams) answered by Gemini on Vertex AI, with Firestore customer context.

## Architecture

```
Twilio ──POST /voice/incoming──► TwiML <Connect><Stream>  ──► WS /voice/stream
                                                                   │
                  TwilioTransport (framing, µ-law, clear/mark, REST hang-up)
                                                                   │
                  CallSession  (auth token, Firestore context, tools, engine choice, watchdog)
                         ┌─────────────────────┴─────────────────────┐
                 LiveEngine (default)                          CascadeEngine (fallback)
       Gemini Live native audio, server VAD            Google STT → Gemini text → Google TTS
       8k↔16k/24k stateful resampling                  energy gate + interim-STT barge-in
```

| Concern | Where | Notes |
|---|---|---|
| Transport vs. logic split | `services/transport.py`, `call_session.py` | Engines never see Twilio JSON |
| Low latency | `engines/live.py` | One audio-to-audio stream; `turn_latency` logged per turn |
| Automatic fallback | `call_session.py` | Live fails at start **or** mid-call → the same call continues on the cascade engine |
| Barge-in | both engines | Live: server `interrupted` → Twilio `clear`. Cascade: loud frames / interim STT ≥ N chars, race-free turn counter |
| Authorization in code | `services/tools.py` | Tools take no customer id; identity comes from the caller number. Optional DOB verification with lockout |
| Clean hang-up | `engines/base.py` | `end_call` / `escalate_to_agent` → goodbye audio → Twilio `mark` echo → REST hang-up (timer as safety net) |
| Escalation | `database.create_escalation` | Writes a callback request to Firestore (`escalations`) |
| Call limits | `call_session.py` | `MAX_CALL_SECONDS` watchdog |
| Context before first word | `ai_agent.build_call_context` | Firestore profile loaded before the engine starts |

## Run locally
```bash
cp .env.example .env            # fill in values
pip install -r requirements-dev.txt
pytest                          # 35 tests, no GCP credentials needed
uvicorn app.main:app --port 8080
ngrok http 8080                 # set PUBLIC_BASE_URL, then point the Twilio number's
                                # "A call comes in" webhook to  POST {PUBLIC_BASE_URL}/voice/incoming
```
`VOICE_ENGINE=cascade` forces the STT → LLM → TTS pipeline.

## Deploy (Cloud Run)
```bash
gcloud run deploy voxa --source . --region us-central1 --timeout 3600 --min-instances 1 \
  --set-env-vars GCP_PROJECT_ID=...,PUBLIC_BASE_URL=...,TWILIO_ACCOUNT_SID=...,TWILIO_PHONE_NUMBER=... \
  --set-secrets TWILIO_AUTH_TOKEN=twilio-auth-token:latest
```
Service account roles: `roles/aiplatform.user`, `roles/datastore.user`, plus Speech-to-Text and Text-to-Speech API access.

Firestore: `customers/{phone without '+'}` → `name`, `language`, `birth_date` (YYYY-MM-DD, optional), `history[]`.
