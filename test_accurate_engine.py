"""Self-contained smoke test for the Accurate-mode engine (masterplan 2.1).

Verifies the things that would otherwise fail silently:
  1. The engine loads the merged large-v3 Afrikaans model and produces output.
  2. It goes through transformers `generate()` — and CTranslate2 /
     faster-whisper are never imported (the broken path for this model).
  3. `condition_on_prev_tokens=True` actually reaches generate().
  4. The result shape matches the Fast seam, so diarization stays happy.
  5. Importing the product module does not drag in torch/transformers,
     i.e. Fast mode is untouched by this work.

Run with the spike venv (it has torch + transformers):
    ~/whisper_afrikaans_spike/venv/bin/python3 test_accurate_engine.py

Uses the existing 30-second benchmark clip, so it costs ~1-2 min, not hours.
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

AUDIO = os.path.expanduser("~/whisper_afrikaans_spike/bench_30s.wav")

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


print("\n=== 1. Fast-mode isolation: importing the engine must not need torch ===")
import accurate_engine  # noqa: E402

check("accurate_engine imports without pulling in torch",
      "torch" not in sys.modules,
      "torch absent from sys.modules after import")
check("accurate_engine imports without pulling in transformers",
      "transformers" not in sys.modules)

print("\n=== 2. Model directory resolution ===")
resolved = accurate_engine.resolve_model_dir()
print(f"  resolved model dir: {resolved}")
ok, detail = accurate_engine.check_model_available()
check("model directory present and complete", ok, detail)
if not ok:
    print("\nCannot continue without the model. Aborting.")
    sys.exit(1)

check("audio fixture present", os.path.isfile(AUDIO), AUDIO)
if not os.path.isfile(AUDIO):
    sys.exit(1)

print("\n=== 3. Config constants (Wave 0.5 verified) ===")
check("condition_on_prev_tokens default is True",
      accurate_engine.CONDITION_ON_PREV_TOKENS is True)
check("default language is 'af'", accurate_engine.DEFAULT_LANGUAGE == "af")

print("\n=== 4. Spy on generate() to prove the kwargs that actually reach it ===")
import torch  # noqa: E402
from transformers import WhisperForConditionalGeneration  # noqa: E402

captured = {}
_real_generate = WhisperForConditionalGeneration.generate


def _spy_generate(self, *args, **kwargs):
    captured.update(kwargs)
    return _real_generate(self, *args, **kwargs)


WhisperForConditionalGeneration.generate = _spy_generate

print("\n=== 5. Transcribe the 30s clip ===")
t0 = time.time()
result = accurate_engine.transcribe(AUDIO, language="af", task="transcribe")
elapsed = time.time() - t0
WhisperForConditionalGeneration.generate = _real_generate

audio_len = 30.0
print(f"  wall clock: {elapsed:.1f}s for ~{audio_len:.0f}s audio "
      f"(RTF ~{elapsed / audio_len:.2f}x)")

print("\n=== 6. Assertions ===")
check("condition_on_prev_tokens=True reached generate()",
      captured.get("condition_on_prev_tokens") is True,
      f"got {captured.get('condition_on_prev_tokens')!r}")
check("return_timestamps=True reached generate() (long-form path)",
      captured.get("return_timestamps") is True)
check("temperature fallback ladder passed",
      isinstance(captured.get("temperature"), tuple),
      f"got {captured.get('temperature')!r}")
check("language forced to af", captured.get("language") == "af")
check("task=transcribe", captured.get("task") == "transcribe")

check("CTranslate2 never imported", "ctranslate2" not in sys.modules)
check("faster_whisper never imported", "faster_whisper" not in sys.modules)
check("transformers WAS used (the correct path)", "transformers" in sys.modules)

text = result.get("text", "")
segments = result.get("segments", [])
check("produced non-empty text", bool(text.strip()), f"{len(text)} chars")
check("result has Fast-seam shape (text + segments)",
      "text" in result and "segments" in result)
check("segments non-empty", len(segments) > 0, f"{len(segments)} segments")
if segments:
    s = segments[0]
    check("segments carry start/end/text (diarization contract)",
          all(k in s for k in ("start", "end", "text")),
          f"keys={sorted(s.keys())}")
    check("segment start/end are floats",
          isinstance(s["start"], float) and isinstance(s["end"], float))
check("engine label recorded", bool(result.get("engine")), result.get("engine", ""))

print("\n=== 7. Output ===")
print(f"  text: {text.strip()[:400]}")
print(f"  first segments:")
for s in segments[:3]:
    print(f"    [{s['start']:.2f}-{s['end']:.2f}] {s['text'][:80]}")

print("\n" + "=" * 60)
if failures:
    print(f"RESULT: FAIL — {len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("RESULT: PASS — engine works, uses transformers generate(), "
      "condition_on_prev_tokens=True, CT2 never touched.")
