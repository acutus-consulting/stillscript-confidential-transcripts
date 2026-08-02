"""UI wiring tests for Accurate mode (masterplan 2.3) — DanScribe_v2.py.

Drives the REAL Tkinter/customtkinter app under a real X display (Xvfb via
`xvfb-run`, NOT whatever :0 happens to be — that may be a real desktop
session and must never be touched by an automated test), not a source-level
check of button state. The only things monkeypatched are what a human would
otherwise have to click: the OS file-open dialog and the blocking messagebox
popups (showinfo/showerror/askretrycancel) — there is no human in this test,
and under Xvfb those would hang the run forever rather than tell us anything.

WHAT IS REAL vs MOCKED, section by section (each section says so again inline):
  1-2   Real app construction + button/mode-toggle state. Nothing mocked.
  3     Real consent dialog, built from a REAL describe_download() network
        call against the real Hub — the numbers on screen are the numbers the
        Hub actually reports right now, not a remembered figure.
        ensure_accurate_model() itself IS mocked here: actually downloading
        5.75 GiB is not something a UI test should wait for. The download
        mechanism's real behaviour, including real cancellation against the
        real Hub, is proven separately in test_accurate_model_download.py
        sections 24-26 — this section's job is to prove the UI reacts
        correctly to what that function can do (progress, cancel, fail,
        succeed), not to re-prove the function itself.
  4     Subsequent-activation short-circuit: uses the REAL, already-downloaded,
        already-verified model already on this machine
        (~/.danscribe_models/accurate-af-large-v3, left on disk by earlier
        real-download testing in this project) and the REAL
        ensure_accurate_model() — nothing about the download path is mocked.
        Only transcribe_audio_accurate() is mocked, since Wave 2.1's own test
        suite already covers the engine's correctness; this section exists to
        prove the WIRING reaches it with the right model_dir, not to re-run
        transcription correctness.
  5     Subsequent activation again, this time with diarization ON, IF the
        same real downloaded model is present: the real ensure_accurate_model()
        short-circuit, real self._diarize() running against the real clip
        (masterplan 2.12: this is now the pyannote neural pipeline, ~33s on the
        30s clip plus a one-time model download — it is no longer the cheap
        librosa/sklearn pass this note used to describe), but
        transcribe_audio_accurate() is mocked, same as section 4. A real transformers.generate() call was
        tried here first and dropped: CPU inference timing proved unpredictable
        enough to blow past a 600s test timeout and then crash the process on
        teardown (a daemon thread still inside native torch code when the
        script tried to exit) — fragility with no payoff, since transcription
        correctness is already covered by test_accurate_engine.py. Skipped
        (not faked) if the model isn't there.

Run with the full-requirements venv (needs customtkinter + PIL + the Accurate
stack together, which no earlier venv in this project had at the same time):
    xvfb-run -a <appvenv>/bin/python3 test_accurate_ui.py
"""

import json
import logging
import os
import queue
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

if not os.environ.get("DISPLAY"):
    print("No DISPLAY set — run this under `xvfb-run -a`, not directly.")
    sys.exit(1)

import DanScribe_v2 as ds  # noqa: E402
import accurate_model_download as amd  # noqa: E402
from tkinter import filedialog, messagebox  # noqa: E402

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


def pump_until_gen(predicate, timeout=25, interval=0.02):
    """Generator version of the same wait: yields (instead of blocking) until
    predicate() is true or timeout elapses. Must be driven by _driver() below
    while app.mainloop() is genuinely running on the main thread — that is
    what lets a background worker thread's cross-thread self._ui(...) calls
    (self.after(0, ...)) succeed at all. A plain app.update() polling loop on
    a thread that never calls mainloop() looked equivalent but isn't: Tk
    refuses cross-thread after() posting unless the owning thread is actually
    inside mainloop() at that moment, and raises "main thread is not in main
    loop" otherwise — confirmed by hitting exactly that error before this
    generator-driven design replaced a bare polling loop.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        yield
    return False


def all_widget_texts(widget):
    texts = []
    try:
        texts.append(str(widget.cget("text")))
    except Exception:  # noqa: BLE001 - not every widget has a "text" option
        pass
    for child in widget.winfo_children():
        texts.extend(all_widget_texts(child))
    return texts


# ── message-box interception: the two things a human would click ────────────
# Recorded rather than silently swallowed, so tests can assert WHAT was shown,
# not just that something was. askretrycancel's answer is scripted per call
# via a queue so different scenarios can drive it differently.
mb_calls = {"showinfo": [], "showerror": [], "askretrycancel": []}
askretrycancel_answers = queue.Queue()

_real_showinfo = messagebox.showinfo
_real_showerror = messagebox.showerror
_real_askretrycancel = messagebox.askretrycancel
_real_askopenfilename = filedialog.askopenfilename


def _fake_showinfo(title, message, **kw):
    mb_calls["showinfo"].append((title, message))


def _fake_showerror(title, message, **kw):
    mb_calls["showerror"].append((title, message))


def _fake_askretrycancel(title, message, **kw):
    mb_calls["askretrycancel"].append((title, message))
    return askretrycancel_answers.get(timeout=10)


messagebox.showinfo = _fake_showinfo
messagebox.showerror = _fake_showerror
messagebox.askretrycancel = _fake_askretrycancel

_next_file_path = {"value": None}


def _fake_askopenfilename(**kw):
    return _next_file_path["value"]


filedialog.askopenfilename = _fake_askopenfilename


def reset_mb():
    mb_calls["showinfo"].clear()
    mb_calls["showerror"].clear()
    mb_calls["askretrycancel"].clear()


# ── dialog capture: get a handle on the REAL dialog instances the app creates,
#    without faking their behaviour — subclassing, not replacing. ────────────
captured = {"consent": [], "progress": []}
_RealConsentDialog = ds.AccurateConsentDialog
_RealProgressDialog = ds.AccurateDownloadProgressDialog


class _CapturingConsentDialog(_RealConsentDialog):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        captured["consent"].append(self)


class _CapturingProgressDialog(_RealProgressDialog):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        captured["progress"].append(self)


ds.AccurateConsentDialog = _CapturingConsentDialog
ds.AccurateDownloadProgressDialog = _CapturingProgressDialog


TEST_SCRATCH = tempfile.mkdtemp(prefix="stillscript_ui_test_")
BENCH_CLIP = os.path.expanduser("~/whisper_afrikaans_spike/bench_30s.wav")


# ════════════════════════════════════════════════════════════════════════════
print("=== 1. The app constructs, and the Accurate button is live ===")
# ════════════════════════════════════════════════════════════════════════════
app = ds.DanScribeApp()
app.withdraw()  # off-screen; state and event handling are unaffected

check("Accurate button exists", hasattr(app, "accurate_btn"))
check("Accurate button is NOT disabled",
      app.accurate_btn.cget("state") != "disabled", app.accurate_btn.cget("state"))

texts = all_widget_texts(app)
check("no leftover 'Coming soon' text anywhere in the built UI",
      not any("coming soon" in t.lower() for t in texts),
      [t for t in texts if "coming soon" in t.lower()])
check("the Accurate button's own label reads correctly",
      "Accurate" in app.accurate_btn.cget("text"), app.accurate_btn.cget("text"))


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 2. Mode toggle changes state, both by calling the handler and by "
      "a real button click ===")
# ════════════════════════════════════════════════════════════════════════════
app._select_fast_mode()
check("fast mode selected", app.mode_var.get() == "fast")
check("fast button shows the active colour", app.fast_btn.cget("fg_color") == "#1f538d")
check("accurate button shows the inactive colour", app.accurate_btn.cget("fg_color") == "gray30")
check("fast mode's scope label shows the Fast-mode notice, not Accurate's (masterplan 2.11)",
      app.mode_scope_label.cget("text") == ds.FAST_MODE_SCOPE_NOTE,
      app.mode_scope_label.cget("text"))

app._select_accurate_mode()
check("accurate mode selected", app.mode_var.get() == "accurate")
check("accurate button shows the active colour", app.accurate_btn.cget("fg_color") == "#1f538d")
check("fast button shows the inactive colour", app.fast_btn.cget("fg_color") == "gray30")
check("accurate mode's scope label shows the Accurate-mode notice (masterplan 2.11)",
      app.mode_scope_label.cget("text") == ds.ACCURATE_MODE_SCOPE_NOTE,
      app.mode_scope_label.cget("text"))
check("...and neither notice claims a mode is disabled or blocked — this is "
      "disclosure, not a gate",
      "disabled" not in ds.FAST_MODE_SCOPE_NOTE.lower()
      and "disabled" not in ds.ACCURATE_MODE_SCOPE_NOTE.lower()
      and "blocked" not in ds.FAST_MODE_SCOPE_NOTE.lower()
      and "blocked" not in ds.ACCURATE_MODE_SCOPE_NOTE.lower())

app.fast_btn.invoke()  # a REAL click — CTkButton.invoke() runs its configured command
check("invoking the Fast button (a real click) selects fast mode",
      app.mode_var.get() == "fast")
check("...and a real click updates the scope label too, not just mode_var",
      app.mode_scope_label.cget("text") == ds.FAST_MODE_SCOPE_NOTE)
app.accurate_btn.invoke()
check("invoking the Accurate button (a real click) selects accurate mode",
      app.mode_var.get() == "accurate")
check("...and a real click updates the scope label too, not just mode_var",
      app.mode_scope_label.cget("text") == ds.ACCURATE_MODE_SCOPE_NOTE)

# Both modes must remain fully usable regardless of the notice shown —
# masterplan 2.11 is disclosure, not an enforcement/gating mechanism.
check("Fast button still enabled after showing its own scope notice",
      app.fast_btn.cget("state") != "disabled")
check("Accurate button still enabled after showing its own scope notice",
      app.accurate_btn.cget("state") != "disabled")

# The runs below save real transcript files into the user's real output
# directory (~/Documents/DanScribe_Transcriptions), same as the shipped app
# would. Snapshot what's there BEFORE any accurate-mode run, so only files
# this test run itself created are removed at the end — nothing pre-existing
# is ever touched.
_output_dir_before = set(os.listdir(app._get_output_dir()))


# ════════════════════════════════════════════════════════════════════════════
def test_sequence():
    """Generator: everything from section 3 onward. Driven by _driver(),
    scheduled via app.after() while app.mainloop() runs for real on the main
    thread — see pump_until_gen()'s docstring for why that is necessary."""
    print("\n=== 3. First activation: consent (real data) -> download (mocked) ===")
    # ════════════════════════════════════════════════════════════════════════════
    # is_model_ready() must say False for this whole section, on a scratch root
    # nothing has ever downloaded into.
    os.environ["STILLSCRIPT_MODEL_ROOT"] = os.path.join(TEST_SCRATCH, "empty_root")
    os.makedirs(os.environ["STILLSCRIPT_MODEL_ROOT"], exist_ok=True)
    check("sanity: the scratch model root is genuinely not ready",
          amd.is_model_ready() is False)

    real_describe = amd.describe_download
    real_ensure_accurate_model = amd.ensure_accurate_model
    real_expected_info = real_describe()  # the REAL number, fetched independently,
                                           # to cross-check the dialog against.


    def start_accurate_run(clip_path):
        app.mode_var.set("accurate")
        app._select_accurate_mode()
        _next_file_path["value"] = clip_path
        reset_mb()
        app.start_transcription()


    # ── 3a. Decline ───────────────────────────────────────────────────────────
    mock_calls = []


    def mock_ensure_never_called(**kw):
        mock_calls.append(kw)
        raise AssertionError("ensure_accurate_model() must not run before consent")


    amd.ensure_accurate_model = mock_ensure_never_called
    try:
        start_accurate_run(BENCH_CLIP)
        ok = (yield from pump_until_gen(lambda: len(captured["consent"]) >= 1))
        check("the consent dialog appears on first activation", ok)

        dlg = captured["consent"][-1]
        check("the dialog shows the REAL size from describe_download(), not a "
              "hardcoded or stale figure",
              abs(dlg.size_gb - real_expected_info["total_bytes"] / (1024 ** 3)) < 0.05,
              f"dialog={dlg.size_gb:.3f} GB, live describe_download()="
              f"{real_expected_info['total_bytes'] / (1024 ** 3):.3f} GB")

        all_dialog_text = " ".join(all_widget_texts(dlg)).lower()
        check("mentions it needs an internet connection",
              "internet connection" in all_dialog_text)
        check("gives a realistic (hours-possible) time estimate, not 'a few minutes'",
              "hours" in all_dialog_text and "few minutes" not in all_dialog_text)
        check("states plainly that only the model is downloaded, not recordings "
              "(not buried — check it isn't the smallest/faintest text on screen)",
              "audio" in all_dialog_text and "never" in all_dialog_text)
        check("states plainly that Accurate mode always transcribes as Afrikaans, "
              "regardless of the Audio Language menu (masterplan 2.11)",
              "afrikaans" in all_dialog_text and "audio language" in all_dialog_text)
        check("names the concrete risk on mixed-language audio — slower, and a "
              "higher chance of errors needing full (not light) review",
              "slower" in all_dialog_text and "english" in all_dialog_text
              and "review" in all_dialog_text)
        check("does NOT claim mixed-language audio is unusable outright — Wave 2.8's "
              "own evidence shows large stretches, including English, come through "
              "correctly; the honest framing is risk/review depth, not a blanket ban",
              "not currently suitable" not in all_dialog_text
              and "does not work" not in all_dialog_text
              and "doesn't work" not in all_dialog_text)

        dlg.decline_btn.invoke()
        (yield from pump_until_gen(lambda: app.main_btn.cget("state") == "normal"))
        check("declining leaves the button usable again",
              app.main_btn.cget("state") == "normal")
        check("declining does NOT call the download function",
              len(mock_calls) == 0, str(len(mock_calls)))
        check("declining shows no error dialog — the user said no, that's not a failure",
              not mb_calls["showerror"] and not mb_calls["showinfo"])
    finally:
        pass  # the next sub-test replaces amd.ensure_accurate_model unconditionally

    # ── 3b. Confirm -> progress updates -> cancel ───────────────────────────────
    captured["consent"].clear()
    captured["progress"].clear()
    cancel_calls = []


    def mock_ensure_cancels(*, progress_callback=None, cancel_event=None, **kw):
        cancel_calls.append({"progress_callback": progress_callback, "cancel_event": cancel_event})
        if progress_callback:
            progress_callback(amd.DownloadProgress(
                phase="downloading", message="Downloading the Accurate-mode language model…",
                downloaded_bytes=50 * 2 ** 20, total_bytes=200 * 2 ** 20, fraction=0.25,
                bytes_per_second=2 * 2 ** 20, eta_seconds=75,
            ))
        # Real download checks cancel_event between chunks; simulate that polling
        # loop here rather than raising immediately, so the test also proves the
        # UI's Cancel click (not just a pre-set event) is what triggers this.
        for _ in range(200):
            if cancel_event is not None and cancel_event.is_set():
                raise amd.AccurateModelDownloadCancelled()
            time.sleep(0.05)
        raise TimeoutError("test: Cancel was never observed within ~10s")


    amd.ensure_accurate_model = mock_ensure_cancels
    try:
        start_accurate_run(BENCH_CLIP)
        ok = (yield from pump_until_gen(lambda: len(captured["consent"]) >= 1))
        check("consent dialog appears again on a second first-activation attempt", ok)
        captured["consent"][-1].confirm_btn.invoke()

        ok = (yield from pump_until_gen(lambda: len(captured["progress"]) >= 1))
        check("confirming opens the download progress dialog", ok)
        pdlg = captured["progress"][-1]

        ok = (yield from pump_until_gen(lambda: "MiB" in pdlg.detail_label.cget("text")))
        check("a progress sample reached the dialog", ok, pdlg.detail_label.cget("text"))
        check("the progress bar reflects the emitted DownloadProgress sample's fraction",
              abs(pdlg.bar.get() - 0.25) < 0.01, pdlg.bar.get())
        check("the detail line shows byte counts from the real DownloadProgress shape",
              "MiB" in pdlg.detail_label.cget("text"), pdlg.detail_label.cget("text"))

        pdlg.cancel_btn.invoke()
        check("the Cancel click sets the cancel_event the download function received",
              cancel_calls[0]["cancel_event"].is_set())

        ok = (yield from pump_until_gen(lambda: "cancelled" in app.status_label.cget("text").lower()))
        check("the app reaches the 'cancelled' status", ok, app.status_label.cget("text"))
        check("the progress dialog is gone", not pdlg.winfo_exists())
        check("cancelling shows NO error dialog — a deliberate stop is not a failure",
              not mb_calls["showerror"])
        check("...and no success dialog either", not mb_calls["showinfo"])
        check("the main button is usable again after a cancel",
              app.main_btn.cget("state") == "normal")
    finally:
        pass

    # ── 3c. Failure, decline retry ───────────────────────────────────────────
    captured["consent"].clear()
    captured["progress"].clear()
    fail_calls = []


    def mock_ensure_fails(*, progress_callback=None, cancel_event=None, **kw):
        fail_calls.append(1)
        raise amd.AccurateModelDownloadError(
            "The Accurate-mode language model could not be downloaded. Please try again.",
            can_retry=True,
        )


    amd.ensure_accurate_model = mock_ensure_fails
    start_accurate_run(BENCH_CLIP)
    (yield from pump_until_gen(lambda: len(captured["consent"]) >= 1))
    askretrycancel_answers.put(False)  # user picks "Cancel" on the retry prompt
    captured["consent"][-1].confirm_btn.invoke()

    ok = (yield from pump_until_gen(lambda: len(mb_calls["askretrycancel"]) >= 1))
    check("a retryable failure asks the user via askretrycancel, using the "
          "error's own plain-language message verbatim",
          ok and mb_calls["askretrycancel"][0][1] ==
          "The Accurate-mode language model could not be downloaded. Please try again.")
    (yield from pump_until_gen(lambda: app.main_btn.cget("state") == "normal"))
    check("declining the retry leaves exactly one download attempt made",
          len(fail_calls) == 1, str(len(fail_calls)))
    check("the button is usable again", app.main_btn.cget("state") == "normal")

    # ── 3d. Failure, accept retry -> succeeds -> transcribes ────────────────────
    captured["consent"].clear()
    captured["progress"].clear()
    fail_then_succeed_calls = []
    FAKE_MODEL_DIR = os.path.join(TEST_SCRATCH, "fake_downloaded_model")
    os.makedirs(FAKE_MODEL_DIR, exist_ok=True)


    def mock_ensure_fails_then_succeeds(*, progress_callback=None, cancel_event=None, **kw):
        fail_then_succeed_calls.append(1)
        if len(fail_then_succeed_calls) == 1:
            raise amd.AccurateModelDownloadError("Network hiccup. Please try again.", can_retry=True)
        if progress_callback:
            progress_callback(amd.DownloadProgress(
                phase="done", message="The Accurate-mode language model is ready.",
                fraction=1.0,
            ))
        return FAKE_MODEL_DIR


    transcribe_calls = []
    FAKE_ENGINE_LABEL = "StillScript Accurate — Whisper large-v3 + Afrikaans adapter (test double)"


    def mock_transcribe_audio_accurate(path, *, language, task, model_dir=None, **kw):
        transcribe_calls.append({
            "path": path, "language": language, "task": task, "model_dir": model_dir,
        })
        return {
            "text": "Toets transkripsie vanaf die Akkuraat-modus-toets.",
            "segments": [{"start": 0.0, "end": 3.0,
                          "text": "Toets transkripsie vanaf die Akkuraat-modus-toets."}],
            "language": language,
            "engine": FAKE_ENGINE_LABEL,
        }


    real_transcribe_audio_accurate = ds.transcribe_audio_accurate
    amd.ensure_accurate_model = mock_ensure_fails_then_succeeds
    ds.transcribe_audio_accurate = mock_transcribe_audio_accurate

    try:
        app.diarize_var.set(False)  # keep this run simple; diarization is Wave 2.1's concern
        start_accurate_run(BENCH_CLIP)
        (yield from pump_until_gen(lambda: len(captured["consent"]) >= 1))
        askretrycancel_answers.put(True)  # user picks "Retry"
        captured["consent"][-1].confirm_btn.invoke()

        (yield from pump_until_gen(lambda: len(mb_calls["askretrycancel"]) >= 1))
        ok = (yield from pump_until_gen(lambda: len(mb_calls["showinfo"]) >= 1, timeout=40))
        check("after accepting the retry, the second attempt succeeds and "
              "transcription completes", ok)
        check("the download function was called exactly twice (fail, then succeed)",
              len(fail_then_succeed_calls) == 2, str(len(fail_then_succeed_calls)))
        check("transcribe_audio_accurate() was called with the model_dir "
              "ensure_accurate_model() actually returned — not a guessed path",
              transcribe_calls and transcribe_calls[0]["model_dir"] == FAKE_MODEL_DIR,
              transcribe_calls[-1] if transcribe_calls else None)
        check("the output box shows the (mocked) transcript",
              "Toets transkripsie" in app.output_box.get("1.0", "end"))

        # Provenance correctness — the bug this task also had to fix: an Accurate
        # transcript's footer must name the ACCURATE engine, not the Fast one.
        saved_files = [f for f in os.listdir(app._get_output_dir())
                      if f.endswith(".txt") and "transcript" in f]
        saved_files.sort(key=lambda f: os.path.getmtime(os.path.join(app._get_output_dir(), f)))
        check("a transcript .txt was saved", bool(saved_files))
        if saved_files:
            content = open(os.path.join(app._get_output_dir(), saved_files[-1]),
                           encoding="utf-8").read()
            check("provenance footer names the ACCURATE engine, not Fast/Medium",
                  f"Engine: {FAKE_ENGINE_LABEL}" in content, content[-400:])
            check("provenance footer's Mode line says Accurate",
                  "Mode: Accurate (Akkuraat)" in content, content[-400:])
            check("Fast mode's engine label does NOT leak into an Accurate transcript",
                  ds.FAST_MODE_ENGINE_LABEL not in content)
    finally:
        ds.transcribe_audio_accurate = real_transcribe_audio_accurate


    # ════════════════════════════════════════════════════════════════════════════
    print("\n=== 4. Subsequent activation: REAL ensure_accurate_model() "
          "short-circuits, no consent dialog ===")
    # ════════════════════════════════════════════════════════════════════════════
    # Restore the REAL ensure_accurate_model() — section 3 mocked it several
    # times over and the last mock (3d's fail-then-succeed stub) would happily
    # keep returning its fake path forever if left in place. This section's
    # whole point is to prove the REAL function short-circuits correctly, so
    # it must not still be talking to a stub.
    amd.ensure_accurate_model = real_ensure_accurate_model

    # Point at the REAL managed model root (unset the scratch override) so this
    # exercises the actual, already-downloaded, already-verified model this
    # machine has from earlier real-download testing — not a fixture.
    os.environ.pop("STILLSCRIPT_MODEL_ROOT", None)
    os.environ.pop("STILLSCRIPT_ACCURATE_MODEL_DIR", None)
    real_model_ready = amd.is_model_ready()

    if not real_model_ready:
        print("  [SKIP] no real downloaded model on this machine "
              "(~/.danscribe_models/accurate-af-large-v3) — section 4 needs it, "
              "not faking a multi-GB directory just to pass.")
    else:
        real_target_dir = str(amd.managed_model_dir())
        check("sanity: the real managed model is verified-ready", real_model_ready)

        def consent_must_not_appear(*a, **kw):
            raise AssertionError(
                "AccurateConsentDialog was constructed on a subsequent activation "
                "— the model is already downloaded, this must never happen")

        ds.AccurateConsentDialog = consent_must_not_appear

        transcribe_calls_4 = []

        def mock_transcribe_4(path, *, language, task, model_dir=None, **kw):
            transcribe_calls_4.append(model_dir)
            return {"text": "kort toets", "segments": [{"start": 0, "end": 1, "text": "kort toets"}],
                    "language": language, "engine": FAKE_ENGINE_LABEL}

        real_transcribe = ds.transcribe_audio_accurate
        ds.transcribe_audio_accurate = mock_transcribe_4
        try:
            app.diarize_var.set(False)
            reset_mb()
            start_accurate_run(BENCH_CLIP)
            ok = (yield from pump_until_gen(lambda: len(mb_calls["showinfo"]) >= 1, timeout=60))
            check("a subsequent activation completes without ever showing consent", ok)
            check("...and it used the REAL, already-verified model directory "
                  "(ensure_accurate_model() genuinely short-circuited under this "
                  "button's real call path, not assumed)",
                  transcribe_calls_4 and transcribe_calls_4[0] == real_target_dir,
                  transcribe_calls_4)
        finally:
            ds.transcribe_audio_accurate = real_transcribe
            ds.AccurateConsentDialog = _CapturingConsentDialog


    # ════════════════════════════════════════════════════════════════════════════
    print("\n=== 5. Subsequent activation with diarization ON (mocked engine, "
          "same as every other section) ===")
    # ════════════════════════════════════════════════════════════════════════════
    # transcribe_audio_accurate() is mocked here too, deliberately. A real
    # transformers.generate() call was tried and dropped: CPU inference timing
    # is unpredictable enough that it blew past a 600s test timeout and then
    # crashed the process on teardown while a daemon thread was still inside
    # native torch code — fragility with no payoff, since transcription
    # correctness is Wave 2.1's job (test_accurate_engine.py) and already
    # covered there. What Wave 2.3 actually needs to prove is UI wiring, and
    # section 4 already proves the real ensure_accurate_model() short-circuit
    # end to end. Section 4 ran with diarization OFF, so this section's only
    # remaining job is the one path that leaves genuinely untested: does the
    # Accurate-mode branch correctly call self._diarize() and thread real
    # speaker labels through to the saved transcript, the same way Fast mode
    # already does. _diarize() itself runs for real here — only the engine call
    # that would produce the segments is mocked.
    #
    # NOTE (masterplan 2.12): _diarize() is no longer cheap. It used to be
    # librosa/sklearn on a 30s clip with no torch; it now runs the pyannote
    # neural pipeline, measured at ~33s on this same 30s clip, plus a one-time
    # ~50s model download on a machine that has never run it. The timeout below
    # was raised from 60s for exactly that reason — if this section starts
    # timing out again, check whether the diarization model is present before
    # assuming the UI wiring broke.
    if not real_model_ready:
        print("  [SKIP] same reason as section 4 — no real model on this machine.")
    else:
        # Segments spanning the real clip, well clear of each other, so
        # _diarize()'s clustering has more than one time window to work with
        # (it needs at least max_speakers segments or it falls back to the
        # pause-based path instead of pitch/MFCC clustering).
        diarize_segments = [
            {"start": 2.0, "end": 8.0, "text": "Segment een."},
            {"start": 10.0, "end": 16.0, "text": "Segment twee."},
            {"start": 18.0, "end": 24.0, "text": "Segment drie."},
            {"start": 25.0, "end": 29.0, "text": "Segment vier."},
        ]
        transcribe_calls_5 = []

        def mock_transcribe_5(path, *, language, task, model_dir=None, **kw):
            transcribe_calls_5.append(model_dir)
            return {
                "text": " ".join(s["text"] for s in diarize_segments),
                "segments": diarize_segments,
                "language": language,
                "engine": FAKE_ENGINE_LABEL,
            }

        real_transcribe_5 = ds.transcribe_audio_accurate
        ds.transcribe_audio_accurate = mock_transcribe_5
        try:
            app.diarize_var.set(True)
            app.num_speakers_var.set("2")
            reset_mb()
            start_accurate_run(BENCH_CLIP)
            ok = (yield from pump_until_gen(lambda: len(mb_calls["showinfo"]) >= 1
                                            or mb_calls["showerror"], timeout=300))
            check("a subsequent activation with diarization ON completes "
                  "(mocked engine, real ensure_accurate_model() short-circuit, "
                  "real _diarize())",
                  ok and not mb_calls["showerror"],
                  mb_calls["showerror"] if mb_calls["showerror"]
                  else ("" if ok else "timed out"))
            check("...and it used the REAL, already-verified model directory here too",
                  transcribe_calls_5 and transcribe_calls_5[0] == real_target_dir,
                  transcribe_calls_5)
            if ok and not mb_calls["showerror"]:
                check("Accurate mode's diarize switch actually reached "
                      "self._diarize() — speaker labels appear in the output, "
                      "not just the raw mocked text",
                      app.diarized_segments is not None
                      and len(app.diarized_segments) > 0,
                      app.diarized_segments)
                preview = app.output_box.get("1.0", "end")
                check("the on-screen preview shows a 'Speaker N:' label, not "
                      "the bare unlabelled text",
                      "Speaker" in preview, preview[:200])
        finally:
            ds.transcribe_audio_accurate = real_transcribe_5
            app.diarize_var.set(False)




_test_gen = test_sequence()


def _driver():
    try:
        next(_test_gen)
        app.after(15, _driver)
    except StopIteration:
        app.quit()
    except Exception as e:  # noqa: BLE001 - a crash here must still end the run cleanly
        import traceback
        traceback.print_exc()
        failures.append(f"CRASH in test sequence: {type(e).__name__}: {e}")
        app.quit()


app.after(15, _driver)
app.mainloop()

# ════════════════════════════════════════════════════════════════════════════
# Clean up ONLY the files this run created (see the snapshot taken in
# section 1) — never anything that was already in the user's real output dir.
_output_dir_after = set(os.listdir(app._get_output_dir()))
for name in _output_dir_after - _output_dir_before:
    try:
        os.remove(os.path.join(app._get_output_dir(), name))
    except OSError:
        pass

app.destroy()
shutil.rmtree(TEST_SCRATCH, ignore_errors=True)
messagebox.showinfo = _real_showinfo
messagebox.showerror = _real_showerror
messagebox.askretrycancel = _real_askretrycancel
filedialog.askopenfilename = _real_askopenfilename

print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
