"""Masterplan 3.2 — proves Fast mode's transcribe_audio() actually passes the
Settings-derived temperature all the way to the real openai-whisper call, and
that its returned "reproducibility" label matches what was really used.

Real Whisper Medium model, real 30-second Afrikaans clip (the existing
bench_30s.wav also used by test_accurate_memory.py) — real transcriptions,
not mocks. model.transcribe() itself is wrapped (not replaced) with a spy
that records the kwargs it was actually called with, then calls straight
through to the real implementation — this still does real decoding work and
returns a real transcript; only the assertion boundary is instrumented.

The UI-level wiring (Settings -> config -> this call site) is proven
separately in test_accurate_ui.py section 7, mirroring the split
test_reproducibility_settings.py's own docstring already documents for
Accurate mode.

Run with the full-requirements venv:
    ~/whisper_afrikaans_spike/venv/bin/python3 test_fast_mode_reproducibility.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stillscript as ds  # noqa: E402

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


CLIP = os.path.expanduser("~/whisper_afrikaans_spike/bench_30s.wav")


# ════════════════════════════════════════════════════════════════════════════
print("=== 1. transcribe_audio() passes the real temperature value to the "
      "real openai-whisper call site, for both settings ===")
# ════════════════════════════════════════════════════════════════════════════
model = ds.get_model("medium")  # real load, cached by get_model() for reuse below
real_transcribe = model.transcribe
received_calls = []


def spy_transcribe(*args, **kwargs):
    received_calls.append(kwargs)
    return real_transcribe(*args, **kwargs)


model.transcribe = spy_transcribe
try:
    for choice, expected_temp in (
        ("Consistent", ds.FAST_TEMPERATURE_CONSISTENT),
        ("Best effort", ds.FAST_TEMPERATURE_BEST_EFFORT),
    ):
        received_calls.clear()
        result = ds.transcribe_audio(
            CLIP, language="af", task="transcribe", model_name="medium",
            temperature=expected_temp,
        )
        check(f"reproducibility='{choice}' -> a real transcript was produced",
              bool(result.get("text", "").strip()), result.get("text", "")[:80])
        check(f"...and the REAL model.transcribe() call received "
              f"temperature={expected_temp!r} (the actual openai-whisper call "
              f"site, not just our own code's belief)",
              received_calls and received_calls[0].get("temperature") == expected_temp,
              received_calls)
        check(f"...and transcribe_audio()'s own returned dict labels it "
              f"'{choice}', sourced from the value actually used",
              result.get("reproducibility") == choice, result.get("reproducibility"))
finally:
    model.transcribe = real_transcribe


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 2. omitting temperature preserves the pre-3.2 default (Best "
      "effort, openai-whisper's own default) — existing callers unaffected ===")
# ════════════════════════════════════════════════════════════════════════════
received_calls.clear()
model.transcribe = spy_transcribe
try:
    result = ds.transcribe_audio(CLIP, language="af", task="transcribe", model_name="medium")
finally:
    model.transcribe = real_transcribe
check("a caller that omits temperature (e.g. test_accurate_memory.py's "
      "existing call) reaches the real call site with the Best-effort tuple, "
      "unchanged from before this item",
      received_calls and received_calls[0].get("temperature") == ds.FAST_TEMPERATURE_BEST_EFFORT,
      received_calls)
check("...and the result is still labelled 'Best effort'",
      result.get("reproducibility") == "Best effort", result.get("reproducibility"))


print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
