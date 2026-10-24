from services.audio_codec import ModelToPhone, PhoneToModel, Resampler


def test_phone_to_model_doubles_sample_rate():
    conv = PhoneToModel()
    total = sum(len(conv(b"\xff" * 160)) for _ in range(50))  # 50 × 20 ms
    assert abs(total - 50 * 640) <= 64  # 160 samples @8k → 320 samples @16k, 2 bytes each


def test_model_to_phone_resamples_24k_to_8k_ulaw():
    conv = ModelToPhone()
    total = sum(len(conv(b"\x00\x00" * 480)) for _ in range(50))  # 50 × 20 ms @24 kHz
    assert abs(total - 50 * 160) <= 16


def test_resampler_state_makes_chunking_irrelevant():
    pcm = bytes((i * 7) % 256 for i in range(4800))
    whole = Resampler(24000, 8000).convert(pcm)
    r = Resampler(24000, 8000)
    parts = b"".join(r.convert(pcm[i:i + 960]) for i in range(0, len(pcm), 960))
    assert whole == parts  # stateless per-frame conversion would differ at the boundaries
