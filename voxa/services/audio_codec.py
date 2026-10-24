"""Telephony audio conversion. Twilio: µ-law 8 kHz. Gemini Live: PCM16 16 kHz in, 24 kHz out."""
import audioop

TWILIO_RATE = 8000
LIVE_IN_RATE = 16000
LIVE_OUT_RATE = 24000
FRAME_BYTES = 160  # 20 ms of µ-law at 8 kHz
OUT_CHUNK_BYTES = 3200  # 400 ms of µ-law per outbound media message


class Resampler:
    """Stateful mono PCM16 resampler.

    Phone audio arrives in 20 ms frames; resampling each frame in isolation clicks at the boundaries,
    so the filter state is carried across calls (audioop.ratecv).
    """

    def __init__(self, in_rate: int, out_rate: int) -> None:
        self._in, self._out = in_rate, out_rate
        self._state = None

    def convert(self, pcm: bytes) -> bytes:
        out, self._state = audioop.ratecv(pcm, 2, 1, self._in, self._out, self._state)
        return out

    def reset(self) -> None:
        self._state = None


class PhoneToModel:
    """µ-law 8 kHz frame → PCM16 16 kHz (Gemini Live input)."""

    def __init__(self) -> None:
        self._up = Resampler(TWILIO_RATE, LIVE_IN_RATE)

    def __call__(self, mulaw: bytes) -> bytes:
        return self._up.convert(audioop.ulaw2lin(mulaw, 2))


class ModelToPhone:
    """PCM16 24 kHz (Gemini Live output) → µ-law 8 kHz."""

    def __init__(self) -> None:
        self._down = Resampler(LIVE_OUT_RATE, TWILIO_RATE)

    def __call__(self, pcm24k: bytes) -> bytes:
        return audioop.lin2ulaw(self._down.convert(pcm24k), 2)

    def reset(self) -> None:
        self._down.reset()


def mulaw_rms(frame: bytes) -> int:
    """RMS energy (16-bit PCM scale) of a µ-law frame."""
    return audioop.rms(audioop.ulaw2lin(frame, 2), 2) if frame else 0


def strip_wav_header(audio: bytes) -> bytes:
    """Google TTS wraps MULAW in a WAV container; Twilio wants the raw samples."""
    if audio[:4] == b"RIFF":
        idx = audio.find(b"data")
        if idx != -1:
            return audio[idx + 8:]
    return audio


def chunk_audio(audio: bytes, size: int = OUT_CHUNK_BYTES) -> list[bytes]:
    return [audio[i:i + size] for i in range(0, len(audio), size)]
