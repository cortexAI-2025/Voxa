# Voxa — AI voice agent (MVP)

Inbound phone calls (Twilio Media Streams) answered by Gemini on Vertex AI, with Firestore customer context.

```
Twilio ─POST /voice/incoming─► TwiML <Connect><Stream> ─► WS /voice/stream
  µ-law 8 kHz ─► Google STT (streaming) ─► final transcript ─► VoxaAgent
  (Firestore profile + Gemini stream) ─► sentences ─► Google TTS (µ-law 8 kHz) ─► Twilio
Barge-in: caller speech while Voxa talks ─► {"event":"clear"} + cancel response
```

## Run locally
```bash
cp .env.example .env            # fill in values
pip install -r requirements-dev.txt
pytest                          # no GCP credentials needed
uvicorn app.main:app --port 8080
ngrok http 8080                 # set PUBLIC_BASE_URL, then point the Twilio number's
                                # "A call comes in" webhook to  POST {PUBLIC_BASE_URL}/voice/incoming
```

## Deploy (Cloud Run)
```bash
gcloud run deploy voxa --source . --region us-central1 --timeout 3600 --min-instances 1 \
  --set-env-vars GCP_PROJECT_ID=...,PUBLIC_BASE_URL=...,TWILIO_ACCOUNT_SID=...,TWILIO_PHONE_NUMBER=... \
  --set-secrets TWILIO_AUTH_TOKEN=twilio-auth-token:latest
```
The service account needs `roles/aiplatform.user`, `roles/datastore.user` and Speech/TTS API access.
Firestore documents live in `customers/{phone without '+'}` with `name`, `language`, `history[]`.
