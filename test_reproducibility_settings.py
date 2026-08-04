"""Tests for masterplan 2.9 (reproducibility choice — "Consistent" vs
"Best effort", Accurate mode) and masterplan 3.2 (the same choice, extended
to Fast mode) that don't need a real Tkinter app or a real generate()/
transcribe() call.

WHAT IS AND IS NOT HERE
────────────────────────
Pure/fast logic only: accurate_engine._describe_reproducibility()'s and
stillscript._describe_fast_reproducibility()'s label derivation,
load_config()/save_config()'s persistence and validation (the one shared
config key both modes now read), and build_provenance()/
format_provenance_lines()'s rendering of the field for both modes. No torch,
no real model load, no real audio.

The REAL UI wiring — that a real Settings-window change actually reaches
transcribe_audio_accurate() / transcribe_audio() as the correct temperature
value at the real call site, and that it persists across a simulated app
reload — is proven separately in test_accurate_ui.py (section 1b and
section 4's extension for Accurate mode; section 7 for Fast mode), which
already has the real-Tkinter-app-under-Xvfb harness this would otherwise
have to duplicate. The real openai-whisper call boundary for Fast mode
specifically (that transcribe_audio() passes temperature all the way into a
real model.transcribe() call, using real audio) is proven separately again,
in test_fast_mode_reproducibility.py.

Run with the full-requirements venv (needs accurate_engine importable, same
as the other accurate-mode test files):
    <appvenv>/bin/python3 test_reproducibility_settings.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stillscript as ds  # noqa: E402
import accurate_engine  # noqa: E402

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


# ════════════════════════════════════════════════════════════════════════════
print("=== 1. accurate_engine constants match what masterplan 1.4/2.9 specify ===")
# ════════════════════════════════════════════════════════════════════════════
check("TEMPERATURE_CONSISTENT is plain 0.0 (deterministic, no sampling)",
      accurate_engine.TEMPERATURE_CONSISTENT == 0.0,
      accurate_engine.TEMPERATURE_CONSISTENT)
check("TEMPERATURE_BEST_EFFORT is the fallback ladder, not a single value",
      accurate_engine.TEMPERATURE_BEST_EFFORT == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
      accurate_engine.TEMPERATURE_BEST_EFFORT)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 2. _describe_reproducibility() labels the ACTUAL value used, not "
      "a re-derived guess ===")
# ════════════════════════════════════════════════════════════════════════════
check("TEMPERATURE_CONSISTENT -> 'Consistent'",
      accurate_engine._describe_reproducibility(accurate_engine.TEMPERATURE_CONSISTENT)
      == "Consistent")
check("TEMPERATURE_BEST_EFFORT -> 'Best effort'",
      accurate_engine._describe_reproducibility(accurate_engine.TEMPERATURE_BEST_EFFORT)
      == "Best effort")
check("a value that is neither constant is labelled honestly as custom, not "
      "silently mislabelled as one of the two named Settings choices",
      accurate_engine._describe_reproducibility(0.5) == "Custom (temperature=0.5)",
      accurate_engine._describe_reproducibility(0.5))


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 2b. Fast mode's own constants/label derivation (masterplan 3.2) "
      "mirror accurate_engine's exactly ===")
# ════════════════════════════════════════════════════════════════════════════
check("FAST_TEMPERATURE_CONSISTENT is plain 0.0, same as Accurate mode's",
      ds.FAST_TEMPERATURE_CONSISTENT == accurate_engine.TEMPERATURE_CONSISTENT
      == 0.0, ds.FAST_TEMPERATURE_CONSISTENT)
check("FAST_TEMPERATURE_BEST_EFFORT is the identical fallback tuple, not a "
      "separately-invented value",
      ds.FAST_TEMPERATURE_BEST_EFFORT == accurate_engine.TEMPERATURE_BEST_EFFORT
      == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), ds.FAST_TEMPERATURE_BEST_EFFORT)
check("_describe_fast_reproducibility(FAST_TEMPERATURE_CONSISTENT) -> "
      "'Consistent'",
      ds._describe_fast_reproducibility(ds.FAST_TEMPERATURE_CONSISTENT)
      == "Consistent")
check("_describe_fast_reproducibility(FAST_TEMPERATURE_BEST_EFFORT) -> "
      "'Best effort'",
      ds._describe_fast_reproducibility(ds.FAST_TEMPERATURE_BEST_EFFORT)
      == "Best effort")
check("a value that is neither Fast-mode constant is labelled honestly as "
      "custom, same discipline as Accurate mode's version",
      ds._describe_fast_reproducibility(0.5) == "Custom (temperature=0.5)",
      ds._describe_fast_reproducibility(0.5))


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 3. load_config()/save_config() persistence and validation ===")
# ════════════════════════════════════════════════════════════════════════════
# Real ~/.stillscript_config.json — snapshot and restore exactly, same care as
# every other test file that touches real user state in this project.
_had_config = os.path.exists(ds.CONFIG_PATH)
_original_config_raw = None
if _had_config:
    with open(ds.CONFIG_PATH) as f:
        _original_config_raw = f.read()

try:
    try:
        os.remove(ds.CONFIG_PATH)
    except OSError:
        pass
    check("default (no config file at all) is 'Consistent', matching "
          "masterplan 1.4's decision",
          ds.load_config()["reproducibility"] == "Consistent")

    # Round-trip both real values through an actual file write + a FRESH
    # read — the fresh read is what makes this a real "simulated restart"
    # check, not just an in-memory assertion.
    for choice in ("Consistent", "Best effort"):
        cfg = ds.load_config()
        cfg["reproducibility"] = choice
        ds.save_config(cfg)
        reloaded = ds.load_config()  # fresh disk read = a simulated relaunch
        check(f"'{choice}' survives a real save_config() -> fresh load_config() "
              f"round-trip", reloaded["reproducibility"] == choice, reloaded)

    # A partial/old config file (missing the key entirely) must not crash and
    # must fall back to the default — same discipline as last_language.
    with open(ds.CONFIG_PATH, "w") as f:
        json.dump({"api_key": ""}, f)
    check("a config file missing the 'reproducibility' key entirely still "
          "defaults to 'Consistent'",
          ds.load_config()["reproducibility"] == "Consistent")

    # An invalid/corrupted value must not propagate into the running app.
    with open(ds.CONFIG_PATH, "w") as f:
        json.dump({"reproducibility": "yolo"}, f)
    check("an invalid persisted value falls back to 'Consistent' rather than "
          "propagating garbage into the engine call",
          ds.load_config()["reproducibility"] == "Consistent")
finally:
    if _had_config:
        with open(ds.CONFIG_PATH, "w") as f:
            f.write(_original_config_raw)
    else:
        try:
            os.remove(ds.CONFIG_PATH)
        except OSError:
            pass


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 4. build_provenance()/format_provenance_lines() render the field "
      "for BOTH modes now (masterplan 3.2 made it non-Accurate-only) ===")
# ════════════════════════════════════════════════════════════════════════════
# When a caller doesn't pass reproducibility at all, the dict/footer are
# unchanged from before 2.6/2.9/3.2 — proves the field really is additive,
# not that Fast mode's real call site never passes it (it now does; see
# test_accurate_ui.py section 7 for that).
fast_provenance_bare = ds.build_provenance(
    mode="Fast (Vinnig)", language_label="Afrikaans", task="transcribe",
    diarized=False,
)
check("build_provenance() omits 'reproducibility' entirely (not even None) "
      "when a caller doesn't pass it, whichever mode",
      "reproducibility" not in fast_provenance_bare, fast_provenance_bare)
fast_lines_bare = ds.format_provenance_lines(fast_provenance_bare)
check("...and the rendered footer correspondingly has no 'Reproducibility:' "
      "line in that case",
      not any(line.startswith("Reproducibility:") for line in fast_lines_bare),
      fast_lines_bare)

for mode_label, choice in (
    ("Accurate (Akkuraat)", "Consistent"),
    ("Accurate (Akkuraat)", "Best effort"),
    ("Fast (Vinnig)", "Consistent"),
    ("Fast (Vinnig)", "Best effort"),
):
    prov = ds.build_provenance(
        mode=mode_label, language_label="Afrikaans", task="transcribe",
        diarized=False, reproducibility=choice,
    )
    check(f"{mode_label} provenance carries reproducibility='{choice}' "
          f"unmodified when a caller passes it", prov.get("reproducibility") == choice)
    lines = ds.format_provenance_lines(prov)
    check(f"...and the rendered footer has a matching 'Reproducibility: {choice}' "
          f"line", f"Reproducibility: {choice}" in lines, lines)

# End-to-end through each engine's own label derivation, not just a
# hand-typed string — proves the whole chain (temperature -> engine's label
# -> build_provenance -> footer line) agrees with itself, for both engines.
for describe_fn, mode_label, temp, expected_choice in (
    (accurate_engine._describe_reproducibility, "Accurate (Akkuraat)",
     accurate_engine.TEMPERATURE_CONSISTENT, "Consistent"),
    (accurate_engine._describe_reproducibility, "Accurate (Akkuraat)",
     accurate_engine.TEMPERATURE_BEST_EFFORT, "Best effort"),
    (ds._describe_fast_reproducibility, "Fast (Vinnig)",
     ds.FAST_TEMPERATURE_CONSISTENT, "Consistent"),
    (ds._describe_fast_reproducibility, "Fast (Vinnig)",
     ds.FAST_TEMPERATURE_BEST_EFFORT, "Best effort"),
):
    label = describe_fn(temp)
    prov = ds.build_provenance(
        mode=mode_label, language_label="Afrikaans", task="transcribe",
        diarized=False, reproducibility=label,
    )
    lines = ds.format_provenance_lines(prov)
    check(f"[{mode_label}] temperature={temp!r} -> engine label '{label}' -> "
          f"footer line 'Reproducibility: {expected_choice}', the full real "
          f"chain agrees",
          f"Reproducibility: {expected_choice}" in lines, lines)


# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
