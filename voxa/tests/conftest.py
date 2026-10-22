import os

os.environ.update(
    GCP_PROJECT_ID="test-project",
    TWILIO_ACCOUNT_SID="AC" + "a" * 32,
    TWILIO_AUTH_TOKEN="test-token",
    TWILIO_PHONE_NUMBER="+15550001111",
    PUBLIC_BASE_URL="https://voxa.example.com",
)
