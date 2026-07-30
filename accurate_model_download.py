"""Download-on-demand for the Accurate-mode model (masterplan 2.1a.2).

WHY THIS EXISTS
───────────────
Masterplan 2.1a decided the Accurate model ships as a download rather than a
bundle: 5.75 GiB of weights cannot go inside the installer (GitHub Releases caps
a single file at 2 GiB), and torch is already in the exe, so the real packaging
delta is only ~36-40 MiB of libraries. The weights therefore have to arrive on
the user's machine at some point after installation, and this module is that
"some point".

accurate_engine.py loads from a directory and, until now, defaulted to Danie's
dev spike path (`~/whisper_afrikaans_spike/merged_afrikaans_fp32`). A user who
installs the app has no such directory. This module produces the directory.

SCOPE — THIS IS A FUNCTION, NOT A FLOW
──────────────────────────────────────
Nothing here draws a window, blocks on a prompt, or decides whether the user
wants to spend 5.75 GiB of their bandwidth. That is masterplan 2.3/2.4. This
module gives 2.3 one callable — `ensure_accurate_model()` — plus enough
pre-flight information (`describe_download()`) to ask the question honestly
before starting.

    from accurate_model_download import ensure_accurate_model, describe_download

    info = describe_download()            # size + file count, for the consent dialog
    path = ensure_accurate_model(progress_callback=my_ui_hook)
    result = accurate_engine.transcribe(audio, model_dir=path)

WHAT IT GUARANTEES
──────────────────
1. The revision is PINNED (`fedc252…`), never "main". Same reasoning as the
   guard's pinned fingerprint: a model that can change under us is a model we
   cannot make promises about. If the Hub repo gains a new commit, this app
   keeps fetching the one that was verified. Re-pinning is a deliberate,
   reviewable edit to REVISION here plus a re-run of
   pin_accurate_fingerprint.py — not something that happens by itself.

2. Anything this module leaves on disk in its managed directory has passed FULL
   guard verification (all 1259 tensors hashed), not the 9-probe default. See
   "FULL VERIFICATION IS NOT OPTIONAL HERE" below.

3. A directory that failed verification is deleted, not left behind. There is no
   state in which a later run finds a half-downloaded model and accepts it.

FULL VERIFICATION IS NOT OPTIONAL HERE
──────────────────────────────────────
accurate_guard defaults to nine probe tensors and offers full verification as an
opt-in (STILLSCRIPT_ACCURATE_FULL_VERIFY). That default is a reasonable trade
for a model that has been sitting on a local disk: the probes already catch a
wrong or stock model, and full mode only adds coverage of corruption hiding in
an unprobed tensor.

A fresh download is exactly the case where that extra coverage is the whole
point. Truncation and byte-level corruption in transit are the failure modes
here, and they land wherever they land — a nine-tensor sample of a 1259-tensor
file will miss most of them. So `ensure_accurate_model()` calls the guard with
`full=True` explicitly, which beats the environment variable in the guard's own
resolution order. The ~4 seconds this costs is noise against a download measured
in hours.

This does NOT change the guard's global default. Making full mode the default
everywhere is masterplan 2.2a and remains a separate decision.

THE XET PROBLEM (learned the hard way, uploading this exact model)
─────────────────────────────────────────────────────────────────
Hugging Face serves LFS content through "Xet", a chunked content-addressed
transfer backend (cas-server.xethub.hf.co). It is faster when it works and it
fails outright with connection errors when it does not — a known, recurring
problem independent of the user's connection quality (huggingface/xet-core
issues #311, #407, #592). Publishing this model to the Hub hit it repeatedly.
The reliable workaround is HF_HUB_DISABLE_XET=1, which forces the older plain
HTTPS path through cas-bridge.

A legal secretary, a psychologist, or a journalist must never see a traceback
containing "xethub", and must never be told to set an environment variable. So
the fallback is automatic: attempt the normal (Xet-enabled) download; if it
fails with something that looks like a transport failure, check that the Hub
itself is still reachable, and if it is, retry once with Xet disabled. Only if
that also fails does the user see anything, and what they see is a sentence
about their internet connection.

The check is deliberately narrow. "No internet at all" and "disk full" are real
answers and must surface as themselves — retrying those with a different
transfer backend just wastes the user's time and buries the real cause. See
`_is_retryable_transport_failure()`.

Note on how the switch is thrown: `huggingface_hub.constants.HF_HUB_DISABLE_XET`
is read from the environment once, at import time, so setting os.environ after
huggingface_hub is imported does nothing. The constant itself is re-read on every
download, though, so `_xet_disabled()` sets both the module attribute (what
actually takes effect) and the environment variable (so anything imported later,
or any subprocess, agrees).

RESUME — READ THIS BEFORE ASSUMING IT WORKS
───────────────────────────────────────────
Measured against huggingface_hub 1.24.0 on 2026-07-28, by SIGKILLing a real
download of this model part-way through model.safetensors and restarting it.
The result is two different answers depending on what "interrupted" means:

  - A network blip WITHIN one run is handled, and no bytes are lost. On the
    plain-HTTPS path `http_get` reissues the request with a `Range` header and
    carries on from where it stopped; the Xet path has its own internal retry
    wrapper. This is the common case and it works.

  - Losing the PROCESS (crash, quit, reboot, power cut) loses that file's
    bytes. huggingface_hub 1.x writes each file to a process-unique
    `<etag>.<uuid>.incomplete` and unlinks it in a `finally`
    (file_download.py, PR #4228), so a restart has nothing to resume from.
    Confirmed by measurement, not just by reading the code: 56 MiB of
    model.safetensors transferred, killed, restarted — the second run began
    that file at zero. Both transports behave this way; it is not a Xet quirk.
    The Xet chunk cache does not rescue it either (chunks are consumed into the
    reconstruction buffer, not retained: 96 KiB of cache after 32 MiB moved).

  - Resume across runs therefore works per FILE. Finished files are not fetched
    again — verified, and the resumed run correctly reports absolute progress
    rather than restarting the bar.

For a single 5.75 GiB file that split was useless: the twelve small files
(5.5 MiB) resumed and the one that mattered — 99.9% of the payload — did not. At
the ~0.6 MiB/s measured here, closing the app meant losing up to 2.7 hours.

Pinning an older, resuming huggingface_hub is NOT an option: transformers 5.14.1
requires huggingface-hub>=1.5.0,<2.0, and per-byte resume was removed before 1.5.

CHUNKING IS THE ANSWER (masterplan 2.1a.3)
──────────────────────────────────────────
Rather than hand-write ranged HTTP against Xet's reconstruction behaviour — our
own transfer path, for a model whose entire value rests on being verifiable —
the file was split. `shard_accurate_model.py` published `chunks/` alongside the
original: thirty ~200 MiB byte slices plus a manifest of sizes and sha256s.

Now file-level resume, the mechanism that already works, is enough. An
interruption costs at most one partial chunk (~200 MiB, ~6 minutes) instead of
everything. Nothing clever is required and nothing new has to be trusted: each
chunk is verified against the manifest on arrival, the reassembled file is
verified against the manifest's whole-file sha256, and the guard then runs in
full mode over the result exactly as before. Three independent checks, all of
which must agree, before a byte of it is used for a transcript.

The split is byte-level, deliberately not tensor-aware. Re-serialising the
weights would change the bytes and invalidate the pinned fingerprint; plain byte
slices concatenate back to something bit-identical by construction.

LAYOUTS
───────
`layout="chunked"` (default) fetches the parts and reassembles them.
`layout="single"` is the 2.1a.2 path, still working and still pinned to
REVISION, kept as a rollback while the chunked path proves itself in the field.
Both are served by the same repo; see CHUNKED_REVISION for which commit is which.

DISK
────
The chunked path needs about 11.5 GiB free while it installs — the parts and the
reassembled model exist at the same time — settling to 5.75 GiB once the parts
are discarded. That is the deliberate price of requirement (d): the parts are
kept until the guard has passed, so a failure at the last step costs a retry of
the assembly rather than a re-download of everything. `_check_disk_space()`
refuses up front rather than filling the disk halfway through.

DEPENDENCIES
────────────
`huggingface_hub` only, imported lazily inside the functions — same discipline
as accurate_engine and accurate_guard. Fast mode's runtime must not acquire a
new import just because this module exists on disk. The guard call pulls in
torch/safetensors, but only after a download has actually happened.
"""

import contextlib
import errno
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger("danscribe.accurate.download")

# The Accurate-mode error hierarchy lives in accurate_guard so the UI can import
# and handle these without pulling in torch/transformers.
from accurate_guard import (  # noqa: F401  (re-export)
    AccurateEngineUnavailable,
    AccurateModelGuardError,
    verify_merged_model,
)


# ─────────────────────────────────────────────
#  WHAT WE FETCH
# ─────────────────────────────────────────────
REPO_ID = "DanieClar/stillscript-whisper-large-v3-afrikaans"

# PINNED. Not "main". The guard's fingerprint is pinned to one specific merge,
# so the download has to be pinned to the same one — a floating revision would
# turn a legitimate model update into a guard failure the user cannot explain,
# and (worse) would let the weights change without anyone re-verifying them.
REVISION = "fedc2529d295a9ebf527142e1965cfc8c04c516f"
REVISION_SHORT = REVISION[:7]

# ── The chunked layout (masterplan 2.1a.3) ──────────────────────────────────
# A SECOND pinned revision, deliberately not the same constant as REVISION even
# though the newer commit contains both layouts. Keeping them apart records what
# was actually verified against what:
#
#   REVISION          the commit whose single-file model.safetensors 2.1a.2
#                     downloaded and full-verified. Only the legacy path uses it.
#   CHUNKED_REVISION  the commit that added chunks/ alongside that file. The
#                     chunked path pins this.
#
# When the monolithic file is eventually deleted from the repo, REVISION and the
# legacy path go together in the same change, and a single grep finds both. One
# shared constant would have hidden that.
CHUNKED_REVISION = "6baf2473d04da504f039ade149512d891e4a7ca5"

# Repo-side layout, kept in step with shard_accurate_model.py.
ORIGINAL_WEIGHTS_FILENAME = "model.safetensors"
REPO_CHUNK_DIR = "chunks"
MANIFEST_NAME = f"{ORIGINAL_WEIGHTS_FILENAME}.manifest.json"
MANIFEST_SCHEMA = 1

# Which layout to fetch. "chunked" is the default; "single" is 2.1a.2's path,
# kept until the chunked path has proven itself on real installations.
DEFAULT_LAYOUT = "chunked"

# Cheap sanity check before we trust a directory. The guard is what actually
# proves the model is right; this only decides whether there is anything worth
# handing to the guard. Kept in step with accurate_engine._REQUIRED_MODEL_FILES.
REQUIRED_MODEL_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")


# ─────────────────────────────────────────────
#  WHERE IT GOES
# ─────────────────────────────────────────────
# DanScribe keeps its per-user state as dot-entries directly in the home
# directory — `~/.danscribe.log` and `~/.danscribe_config.json` (DanScribe_v2.py
# lines 28 and 75). This follows that convention rather than inventing a second
# one: `~/.danscribe_models/` matches the `~/.danscribe*` glob that masterplan
# 4.1 already has to sweep when the product is renamed to StillScript, so the
# rename stays one coordinated change instead of two.
#
# Deliberately NOT the huggingface_hub shared cache (~/.cache/huggingface): the
# app must be able to say exactly where the 5.75 GiB went and delete it on
# request, and the model directory the guard and from_pretrained() see should be
# a plain directory of real files, not a tree of blobs and symlinks.
#
# Deliberately NOT %LOCALAPPDATA% on Windows either, for now. It is arguably the
# more correct Windows location for 5.75 GiB, but the app's config and log do
# not use it, and having the model somewhere the rest of the app's state is not
# is how paths get lost in a rename. Revisit as part of 4.1 if desired — it is a
# one-function change, confined to default_model_root().
MODEL_ROOT_ENV_VAR = "STILLSCRIPT_MODEL_ROOT"

# Shared with accurate_engine.resolve_model_dir(), so an override set for one is
# honoured by the other.
MODEL_DIR_ENV_VAR = "STILLSCRIPT_ACCURATE_MODEL_DIR"

_MANAGED_DIR_NAME = "accurate-af-large-v3"

# Written into the managed directory once, after a download has passed FULL
# verification. Its presence is the only thing that lets a later run skip
# re-verifying. No stamp means "not finished" — which is what makes an
# interrupted download safe to leave on disk for a resume attempt.
STAMP_FILENAME = ".stillscript_download.json"

# huggingface_hub's own metadata lives here inside local_dir. Not part of the
# model; excluded when we count "bytes that are really downloaded".
_HF_INTERNAL_DIRNAME = ".cache"


# ─────────────────────────────────────────────
#  TRANSPORT BEHAVIOUR
# ─────────────────────────────────────────────
# How many times to attempt the transfer in total, across both transports.
# Attempt 1 uses Xet (if available); the first retryable failure switches Xet off
# for every remaining attempt, because a backend that just refused is not likely
# to work better three seconds later. Remaining attempts exist for ordinary
# network flakiness, which matters more than usual here given that an interrupted
# large file restarts from zero (see RESUME above).
MAX_TRANSPORT_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (5, 20)

# How often the reporter thread emits a progress event. Fast enough to feel
# live, slow enough that a Tk UI marshalling each one to the main thread does
# not care.
PROGRESS_INTERVAL_SECONDS = 0.5

# Reachability probe timeout, used to tell "Xet is broken" apart from "this
# machine is offline".
_REACHABILITY_TIMEOUT = 10

_HUB_ROOT = "https://huggingface.co"


class AccurateModelDownloadError(AccurateEngineUnavailable):
    """Raised when the Accurate model could not be obtained.

    A subclass of AccurateEngineUnavailable so existing handlers catch it, while
    code that wants to offer a "retry download" button can tell it apart from
    "this machine cannot run Accurate mode at all".

    The message is written to be shown to a non-technical user as-is. Anything a
    developer needs is on the attributes instead:

      `detail`          — the technical cause, for the log
      `guard_failures`  — the guard's failed-check list, when verification failed
      `can_retry`       — whether trying again might plausibly help
    """

    def __init__(self, message, detail=None, guard_failures=None, can_retry=True):
        super().__init__(message)
        self.detail = detail
        self.guard_failures = guard_failures or []
        self.can_retry = can_retry


class AccurateModelDownloadCancelled(AccurateModelDownloadError):
    """Raised when a caller-supplied `cancel_event` was set mid-download.

    A distinct subclass, not a generic AccurateModelDownloadError, so a UI can
    tell "the user cancelled on purpose" apart from "something went wrong" —
    catch this one first and skip the error dialog entirely; the user already
    knows what they did.

    Never raised for anything the user didn't ask for. Whatever chunks had
    already landed on disk are untouched — cancelling is deliberately not a
    failure path, so none of the guard-failure cleanup in
    ensure_accurate_model() runs, and a later attempt resumes exactly where
    this one stopped.
    """

    def __init__(self):
        super().__init__(
            "Download cancelled.",
            detail="cancel_event was set",
            can_retry=True,
        )


def _check_cancelled(cancel_event):
    """Raise AccurateModelDownloadCancelled if the caller asked to stop.

    Checked between whole chunks (~200 MiB units), not mid-byte — the same
    granularity an involuntary interruption already has (see RESUME in the
    module docstring), so a deliberate cancel is never worse than a crash.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise AccurateModelDownloadCancelled()


# ─────────────────────────────────────────────
#  PROGRESS
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class DownloadProgress:
    """One progress sample, handed to the caller's `progress_callback`.

    Every field is safe to render directly. `fraction` is clamped to [0, 1] and
    `eta_seconds` is None rather than an absurd number when there is not yet
    enough history to estimate one.

    `phase` is one of:
      "preparing"  — asking the Hub what the download consists of
      "downloading"— bytes moving
      "verifying"  — full guard pass over the downloaded weights
      "done"       — verified and ready
    """

    phase: str
    message: str
    downloaded_bytes: int = 0
    total_bytes: int = 0
    fraction: float = 0.0
    bytes_per_second: float = 0.0
    eta_seconds: float = None

    def as_dict(self):
        return asdict(self)


def _emit(callback, progress):
    """Deliver one progress sample, absorbing anything the callback throws.

    A UI bug in a progress hook must not be able to abort a three-hour download.
    """
    if callback is None:
        return
    try:
        callback(progress)
    except Exception:  # noqa: BLE001 - deliberately swallowing caller errors
        logger.exception("Accurate-model download progress callback raised; ignoring.")


class _ByteTracker:
    """Thread-safe accumulator for bytes transferred, plus rate/ETA.

    huggingface_hub reports progress by driving tqdm objects. `snapshot_download`
    creates two byte-counting bars from the class we pass it — one counting bytes
    received from the network, one counting bytes written to disk (they differ on
    the Xet path, where reconstruction lags transfer) — and a third counting
    files. We take the larger of the two byte counters rather than deciding which
    bar is which by name or description: on the plain HTTP path only one of them
    moves at all, on the Xet path both do and either is a fair answer to "how far
    along is this", and neither depends on strings that a huggingface_hub upgrade
    could rename.

    `baseline` is the bytes already on disk when the attempt started, so a resumed
    download reports absolute progress rather than restarting the bar at zero.

    CHUNKED DOWNLOADS need one more thing. There, each chunk is a separate
    `hf_hub_download` call that creates its own pair of bars, so "max across
    counters" would report the size of the largest single chunk instead of the
    running total — the bar would climb to 200 MiB and then reset, thirty times.
    `file_done(size)` folds a finished chunk into a persistent total and clears
    the per-file counters, so the max applies within one file and the sum applies
    across them. The snapshot path never calls it and behaves exactly as before,
    because there huggingface_hub has already aggregated every file into the same
    two bars.
    """

    def __init__(self, total_bytes, baseline=0):
        self._lock = threading.Lock()
        self._counters = {}
        self._completed = 0
        self.total_bytes = total_bytes
        self.baseline = baseline
        self._started = time.monotonic()

    def add(self, key, amount):
        if not amount:
            return
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + amount

    def file_done(self, size):
        """Fold a completed file into the running total and reset the live bars."""
        with self._lock:
            self._completed += size
            self._counters.clear()

    def transferred(self):
        with self._lock:
            return self._completed + max(self._counters.values(), default=0)

    def downloaded(self):
        done = self.baseline + self.transferred()
        # Clamp: on the Xet path the network-transfer counter can briefly run
        # ahead of what is really on disk, and a bar past 100% looks broken.
        return min(done, self.total_bytes) if self.total_bytes else done

    def sample(self):
        """Return (downloaded_bytes, bytes_per_second, eta_seconds).

        The rate is the average over this attempt so far, not an instantaneous
        one. Both of the obvious alternatives were tried against the real
        download and are worse:

          - A short-window instantaneous rate reads as a stalled download. Xet
            reconstructs in ~64 MiB blocks, so bytes-on-disk sits flat for tens
            of seconds and then jumps. An instantaneous rate spends most of that
            time at 0.00 MiB/s with no ETA, which for a multi-hour download is
            the single most alarming thing a progress bar can say.
          - Exponential smoothing only slows that decay down; it still bottoms
            out during a long flat stretch.

        Over a transfer measured in hours the cumulative average is also simply
        the better predictor. Bytes already on disk from an earlier attempt are
        excluded from the rate — they took no time in THIS attempt, and counting
        them makes the first sample report an absurd speed.
        """
        elapsed = max(1e-6, time.monotonic() - self._started)
        moved = self.transferred()
        rate = moved / elapsed

        done = self.downloaded()
        remaining = max(0, (self.total_bytes or 0) - done)
        # No ETA until enough has moved to mean anything — an estimate derived
        # from the first half-second of a 5.75 GiB download is noise.
        eta = remaining / rate if (remaining and rate > 1 and moved > 1 << 20) else None
        return done, rate, eta


def _make_tqdm_class(tracker):
    """Build the tqdm stand-in `snapshot_download` will instantiate.

    Subclassing huggingface_hub's tqdm (rather than duck-typing a fake) is the
    conservative choice: `thread_map` iterates the object it constructs, and the
    Xet reporter pokes at `.n`, `.total`, `.refresh()` and friends. Inheriting
    means every one of those keeps working no matter which code path runs.

    `disable=True` is forced so nothing is ever written to the console — this
    library is being driven from a desktop GUI, and a frozen Windows build may
    have no usable stderr at all. Disabled tqdm skips its own bookkeeping in
    `update()`, so the accounting happens here first, before delegating.
    """
    from huggingface_hub.utils import tqdm as hf_tqdm

    class _TrackingTqdm(hf_tqdm):
        def __init__(self, *args, **kwargs):
            self._sink_key = kwargs.get("name") or f"bar-{id(self)}"
            self._counts_bytes = kwargs.get("unit") == "B"
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            if self._counts_bytes:
                tracker.add(self._sink_key, int(n or 0))
            return super().update(n)

        def update_transfer(self, n=1):
            # Present so huggingface_hub routes Xet's network-transfer counter
            # here instead of opening a second visible bar. Counted under its own
            # key; _ByteTracker takes the max across keys.
            tracker.add(f"{self._sink_key}::transfer", int(n or 0))

        def set_transfer_postfix_str(self, postfix, refresh=False):
            pass

    return _TrackingTqdm


class _ProgressReporter:
    """Emits progress on a fixed cadence from one dedicated thread.

    Callbacks are driven by a timer rather than by huggingface_hub's own update
    frequency, for two reasons. A stalled transfer stops calling back entirely,
    and a UI that stops receiving events cannot tell "stalled" from "finished".
    And the Xet path calls back from several worker threads at once, which would
    otherwise push a marshalling problem onto every caller.

    The guarantee to callers is that progress callbacks are NEVER CONCURRENT —
    exactly one is in flight at a time. It is deliberately not "always the same
    thread": the phase changes around the download ("preparing", "verifying",
    "done") are emitted by whichever thread called ensure_accurate_model(). The
    reporter's own thread is started on entry and joined on exit, so it cannot
    overlap with those.
    """

    def __init__(self, tracker, callback, message):
        self._tracker = tracker
        self._callback = callback
        self._message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="stillscript-dl-progress", daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)
        return False

    def _run(self):
        while not self._stop.wait(PROGRESS_INTERVAL_SECONDS):
            self.emit_now()

    def emit_now(self):
        done, rate, eta = self._tracker.sample()
        total = self._tracker.total_bytes
        _emit(self._callback, DownloadProgress(
            phase="downloading",
            message=self._message,
            downloaded_bytes=int(done),
            total_bytes=int(total),
            fraction=min(1.0, done / total) if total else 0.0,
            bytes_per_second=rate,
            eta_seconds=eta,
        ))


# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
def default_model_root():
    """Parent directory for downloaded models. See the note above on why here."""
    override = (os.environ.get(MODEL_ROOT_ENV_VAR) or "").strip()
    if override:
        return Path(override)
    return Path.home() / ".danscribe_models"


def managed_model_dir():
    """The directory this module owns, downloads into, and may delete."""
    return default_model_root() / _MANAGED_DIR_NAME


def resolve_model_dir(model_dir=None):
    """Where the Accurate model should live.

    Same resolution order as accurate_engine.resolve_model_dir() — explicit
    argument, then the environment variable, then the default — so a machine
    configured with STILLSCRIPT_ACCURATE_MODEL_DIR behaves consistently across
    both modules. The difference is only the last step: the engine falls back to
    the dev spike path, this falls back to the managed download location.
    """
    explicit = model_dir or os.environ.get(MODEL_DIR_ENV_VAR)
    return Path(explicit) if explicit else managed_model_dir()


def is_managed(model_dir):
    """True when `model_dir` is the directory this module owns.

    Everything destructive is gated on this. A user who points
    STILLSCRIPT_ACCURATE_MODEL_DIR at a hand-built merge has told us where their
    model is, not given us permission to delete it.
    """
    try:
        return Path(model_dir).resolve() == managed_model_dir().resolve()
    except OSError:
        return False


def has_model_files(model_dir):
    """Cheap "is there a model here at all" check. Says nothing about validity."""
    d = Path(model_dir)
    return d.is_dir() and all((d / f).is_file() for f in REQUIRED_MODEL_FILES)


# ─────────────────────────────────────────────
#  COMPLETION STAMP
# ─────────────────────────────────────────────
# Files-on-disk is not a sufficient completion test: an interrupted download can
# leave config.json and a truncated model.safetensors, which passes
# has_model_files() and would then be handed to the engine. The stamp is written
# only after a full guard pass, so "stamp present and matching" is the single
# fact that means finished-and-verified. Anything else is treated as unfinished
# and re-downloaded (which resumes what it can) and re-verified.
def _stamp_path(model_dir):
    return Path(model_dir) / STAMP_FILENAME


def read_stamp(model_dir):
    """Return the completion stamp dict, or None if absent/unreadable."""
    try:
        with open(_stamp_path(model_dir)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_stamp(model_dir, repo_id, revision, guard_report, total_bytes):
    stamp = {
        "repo_id": repo_id,
        "revision": revision,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "full_verification": True,
        "guard": guard_report,
        "total_bytes": total_bytes,
    }
    tmp = _stamp_path(model_dir).with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(stamp, fh, indent=2)
    os.replace(tmp, _stamp_path(model_dir))
    return stamp


# Revisions whose weights are the same bytes. `chunks/` was published as a new
# commit on top of the one 2.1a.2 pinned, and the chunks reassemble to a file
# with the same sha256 (33bdc94e…) as the model.safetensors already there — so a
# model verified against either commit IS the approved model.
#
# This set exists to stop a stamp comparison from being a provenance test when it
# should be an identity test. Without it, upgrading a machine that already
# finished a 2.1a.2 download would find `revision != CHUNKED_REVISION`, conclude
# the model was stale, and re-fetch 5.75 GiB over ~3 hours to arrive at a
# byte-identical file. Anything added here must have been shown to carry the same
# weights; the guard is what proves it, and it still runs.
EQUIVALENT_REVISIONS = frozenset({REVISION, CHUNKED_REVISION})


def is_verified_download(model_dir, repo_id=REPO_ID, revision=None):
    """True when this directory holds a completed, fully-verified download.

    The repo must match, and the stamped revision must be one that carries the
    approved weights (see EQUIVALENT_REVISIONS). A stamp naming some other
    revision means the app now expects different weights, and the old ones are
    not the approved model any more.
    """
    stamp = read_stamp(model_dir)
    if not stamp:
        return False
    # Equivalence only extends between revisions we have actually shown to carry
    # the same weights. Asking about some other revision is a strict question and
    # gets a strict answer — otherwise a caller pinning a genuinely new model
    # would be told an old download satisfies it.
    if revision is None or revision in EQUIVALENT_REVISIONS:
        accepted = EQUIVALENT_REVISIONS
    else:
        accepted = {revision}
    return (
        stamp.get("repo_id") == repo_id
        and stamp.get("revision") in accepted
        and has_model_files(model_dir)
    )


def is_model_ready(model_dir=None, repo_id=REPO_ID):
    """True when `ensure_accurate_model()` would short-circuit without a network call.

    For a caller (masterplan 2.3's UI) that needs to know, BEFORE doing
    anything, whether this is a first activation (show the download consent
    dialog) or a later one (go straight to transcription) — without either
    duplicating ensure_accurate_model()'s own logic or paying for a describe_
    download() Hub round trip just to find out.

    Mirrors the exact two conditions ensure_accurate_model() early-returns on:
    a verified managed download, or a directory the caller pointed us at that
    already has model files. If those conditions ever change, change them
    here too — kept as a separate, named predicate (rather than inlined only
    inside ensure_accurate_model) specifically so a UI has something safe to
    call ahead of time.
    """
    target = Path(resolve_model_dir(model_dir))
    if is_verified_download(target, repo_id):
        return True
    return not is_managed(target) and has_model_files(target)


# ─────────────────────────────────────────────
#  PRE-FLIGHT
# ─────────────────────────────────────────────
def describe_download(repo_id=REPO_ID, revision=None, layout=DEFAULT_LAYOUT):
    """What the user is about to be asked to download. Requires network.

    Returns {"repo_id", "revision", "layout", "total_bytes", "file_count", "files"}.

    Intended for 2.3's consent dialog, which has to state the size, the time and
    the fact that an internet connection is needed BEFORE anything starts
    (masterplan 2.1a.2 and the waitlist entry on first-run download).

    MUST be layout-aware, and this is not cosmetic. Since 2.1a.3 the repo holds
    BOTH layouts — the original single model.safetensors and the chunks that
    reassemble into an identical copy of it — so summing every file in the repo
    reports ~11.5 GiB, roughly double what any one download actually costs.
    Telling a user their download is twice its real size is exactly the kind of
    number that makes them cancel.

    Note this reports bytes on disk; the Xet transport may move slightly fewer
    bytes over the wire thanks to deduplication, so it is an upper bound on the
    transfer, not a prediction of it.
    """
    from huggingface_hub import HfApi

    if revision is None:
        revision = CHUNKED_REVISION if layout == "chunked" else REVISION

    info = HfApi().repo_info(repo_id, revision=revision, files_metadata=True)
    chunk_prefix = f"{REPO_CHUNK_DIR}/"

    if layout == "chunked":
        # Everything except the monolithic file. The chunks sum to exactly its
        # size, so this is the small files plus one copy of the weights.
        wanted = [s for s in info.siblings if s.rfilename != ORIGINAL_WEIGHTS_FILENAME]
    else:
        wanted = [s for s in info.siblings if not s.rfilename.startswith(chunk_prefix)]

    files = [(s.rfilename, s.size or 0) for s in wanted]
    return {
        "repo_id": repo_id,
        "revision": info.sha,
        "layout": layout,
        "total_bytes": sum(size for _, size in files),
        "file_count": len(files),
        "files": files,
    }


def _bytes_on_disk(model_dir):
    """Bytes of real model content already present, excluding hf_hub's metadata.

    Used as the baseline so a resumed download's progress bar starts where the
    previous attempt got to instead of at zero.
    """
    root = Path(model_dir)
    if not root.is_dir():
        return 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if _HF_INTERNAL_DIRNAME in dirnames:
            dirnames.remove(_HF_INTERNAL_DIRNAME)
        for name in filenames:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            # Symlinks would double-count against their target.
            if not os.path.islink(os.path.join(dirpath, name)):
                total += st.st_size
    return total


def _prune_stale_incomplete(model_dir):
    """Delete leftover `.incomplete` files from attempts that were killed.

    huggingface_hub normally removes these itself, but it does so in a `finally`
    — which never runs if the process is killed or the machine loses power.
    Since a process-unique `.incomplete` can never be resumed (see RESUME in the
    module docstring), every one of these is dead weight, and they accumulate one
    per crashed attempt. At 5.75 GiB a file, that fills a disk quickly.

    Returns the number of bytes reclaimed. Only ever called on the managed
    directory, and only before an attempt starts.
    """
    download_dir = Path(model_dir) / _HF_INTERNAL_DIRNAME / "huggingface" / "download"
    if not download_dir.is_dir():
        return 0
    reclaimed = 0
    for path in download_dir.rglob("*.incomplete"):
        try:
            size = path.stat().st_size
            path.unlink()
            reclaimed += size
        except OSError:
            continue
    if reclaimed:
        logger.info(
            "Removed %.1f MiB of unresumable leftovers from a previous "
            "interrupted download.", reclaimed / 2 ** 20,
        )
    return reclaimed


def _hub_reachable():
    """Can we reach huggingface.co at all?

    This is what separates "the Xet backend is having one of its episodes" from
    "this machine is not on the internet". Only the former is worth retrying with
    a different transport; telling an offline user we are trying a different
    storage backend would be both wrong and useless.

    Uses urllib rather than huggingface_hub's HTTP session on purpose: this runs
    on the failure path, and it should not depend on whatever inside the library
    just broke.
    """
    import urllib.error
    import urllib.request

    request = urllib.request.Request(f"{_HUB_ROOT}/api/whoami-v2", method="HEAD")
    try:
        urllib.request.urlopen(request, timeout=_REACHABILITY_TIMEOUT)
        return True
    except urllib.error.HTTPError:
        # Any HTTP status at all means we spoke to the Hub. 401 is the expected
        # answer here for an unauthenticated request, and it means "reachable".
        return True
    except Exception:  # noqa: BLE001 - URLError, socket errors, TLS failures, ...
        return False


# ─────────────────────────────────────────────
#  ERROR CLASSIFICATION
# ─────────────────────────────────────────────
# Substrings that mark a failure as coming from the Xet transfer backend.
_XET_MARKERS = ("xet", "cas-server", "cas_server", "cas-bridge", "reconstruction")

# Substrings that mark a failure as a real local problem. These must surface as
# themselves: retrying a full disk on a different transport just wastes hours.
_FATAL_MARKERS = (
    "no space left",
    "not enough free disk space",
    "disk space",
    "quota exceeded",
    "read-only file system",
)

_FATAL_ERRNOS = {errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EACCES, errno.EPERM}

# Exception names that mean the request was answered and the answer was "no".
# Matched by name so this module does not have to import them (and so a
# huggingface_hub upgrade that moves them does not break the check).
_PERMANENT_ERROR_NAMES = {
    "RepositoryNotFoundError",
    "RevisionNotFoundError",
    "GatedRepoError",
    "DisabledRepoError",
    "EntryNotFoundError",
    "RemoteEntryNotFoundError",
    "HFValidationError",
    "BadRequestError",
}

# Exception types that are transport failures by construction.
_TRANSPORT_BASE_TYPES = (ConnectionError, TimeoutError)

_TRANSPORT_ERROR_NAMES = {
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "ReadError",
    "WriteError",
    "RemoteProtocolError",
    "PoolTimeout",
    "TransportError",
    "ProtocolError",
    "IncompleteRead",
    "ChunkedEncodingError",
    "SSLError",
    "gaierror",
}


def _exception_chain(exc):
    """Walk __cause__/__context__, so a wrapped cause is still visible."""
    seen = []
    current = exc
    while current is not None and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def _is_retryable_transport_failure(exc):
    """Should this failure be retried on the non-Xet transport?

    True only for things that look like the transfer backend giving up. Refusals
    from the Hub (wrong repo, wrong revision, gated) and local problems (disk
    full, permissions) return False and propagate — retrying them differently
    would change nothing except how long the user waits to find out.

    The Xet failures we are targeting arrive from Rust as fairly plain
    exceptions whose text names the backend, so the message is checked alongside
    the type.
    """
    chain = _exception_chain(exc)
    blob = " | ".join(f"{type(e).__name__}: {e}" for e in chain).lower()

    for e in chain:
        if isinstance(e, OSError) and e.errno in _FATAL_ERRNOS:
            return False
    if any(marker in blob for marker in _FATAL_MARKERS):
        return False
    if any(type(e).__name__ in _PERMANENT_ERROR_NAMES for e in chain):
        return False

    if any(marker in blob for marker in _XET_MARKERS):
        return True
    if any(isinstance(e, _TRANSPORT_BASE_TYPES) for e in chain):
        return True
    if any(type(e).__name__ in _TRANSPORT_ERROR_NAMES for e in chain):
        return True
    return False


class _xet_disabled:
    """Force the plain-HTTPS transfer path for the duration of the block.

    Both switches are thrown on purpose. `huggingface_hub.constants` snapshots
    the environment variable at import time, so setting os.environ alone is too
    late once the library is loaded — the module attribute is the one that takes
    effect. The environment variable is still set so that anything imported
    afterwards (or a subprocess) sees the same decision.
    """

    def __enter__(self):
        from huggingface_hub import constants

        self._constants = constants
        self._prev_attr = constants.HF_HUB_DISABLE_XET
        self._prev_env = os.environ.get("HF_HUB_DISABLE_XET")
        constants.HF_HUB_DISABLE_XET = True
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        return self

    def __exit__(self, *exc):
        self._constants.HF_HUB_DISABLE_XET = self._prev_attr
        if self._prev_env is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = self._prev_env
        return False


# ─────────────────────────────────────────────
#  THE DOWNLOAD
# ─────────────────────────────────────────────
def _snapshot(repo_id, revision, target, tqdm_class, max_workers):
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id,
        revision=revision,
        local_dir=str(target),
        max_workers=max_workers,
        tqdm_class=tqdm_class,
    )


def _fetch_with_fallback(fetch, tracker_factory, progress_callback, message, label,
                         cancel_event=None):
    """Run `fetch(tqdm_class, tracker)` under the retry + Xet-fallback policy.

    This is the whole of 2.1a.2's transport behaviour in one place: attempt on
    Xet, classify any failure, distinguish "the Xet backend is having one of its
    episodes" from "this machine is offline", switch transports once and for
    good, and retry with backoff. The snapshot path and the chunked path both go
    through here, so there is exactly one implementation of it to get right and
    to keep tested.

    `fetch` is called with the tqdm stand-in for the current attempt; it may be
    called more than once, so it must be safe to re-run against whatever the
    previous attempt left on disk. It is also where `cancel_event` actually gets
    checked (inside `_fetch_chunks`, between chunks) — this function only checks
    it between ATTEMPTS, so a cancel during the retry backoff sleep stops the
    retry loop rather than starting another one.

    Returns (result, transport) where transport is "xet" or "https".
    """
    disable_xet = False
    last_error = None

    for attempt in range(1, MAX_TRANSPORT_ATTEMPTS + 1):
        _check_cancelled(cancel_event)
        tracker = tracker_factory()
        tqdm_class = _make_tqdm_class(tracker)
        transport = "https" if disable_xet else "xet"
        logger.info(
            "Accurate model %s attempt %d/%d via %s (%d bytes already on disk)",
            label, attempt, MAX_TRANSPORT_ATTEMPTS, transport, tracker.baseline,
        )

        context = _xet_disabled() if disable_xet else contextlib.nullcontext()
        try:
            with context:
                with _ProgressReporter(tracker, progress_callback, message) as reporter:
                    result = fetch(tqdm_class, tracker)
                # Outside the reporter block: its thread has been joined, so this
                # final 100% sample cannot race a timed one.
                reporter.emit_now()
            logger.info("Accurate model %s finished via %s", label, transport)
            return result, transport
        except KeyboardInterrupt:
            raise
        except AccurateModelDownloadCancelled:
            # A deliberate stop, not a transport problem: must propagate exactly
            # as raised, never reclassified into a "could not be downloaded"
            # friendly-error or retried.
            raise
        except Exception as exc:  # noqa: BLE001 - classified immediately below
            last_error = exc
            retryable = _is_retryable_transport_failure(exc)
            logger.warning(
                "Accurate model download attempt %d/%d via %s failed (%s): %s",
                attempt, MAX_TRANSPORT_ATTEMPTS, transport,
                "retryable" if retryable else "not retryable", exc,
            )
            if not retryable:
                raise _friendly_transport_error(exc) from exc
            if attempt == MAX_TRANSPORT_ATTEMPTS:
                break

            # A transport failure with the Hub itself unreachable is not the Xet
            # problem — it is the user's connection. Say so rather than silently
            # burning the remaining attempts on a machine that is offline.
            if not _hub_reachable():
                raise AccurateModelDownloadError(
                    "The download stopped because the internet connection was lost. "
                    "Check your connection and try again — the parts that finished "
                    "will not need to be downloaded again.",
                    detail=f"{type(exc).__name__}: {exc}",
                ) from exc

            if not disable_xet:
                # First retryable failure with the Hub still up: this is the Xet
                # signature. Switch transports for good rather than alternating.
                disable_xet = True
                logger.warning(
                    "Switching to the plain HTTPS transfer path for the remaining "
                    "attempts (Hugging Face's Xet backend appears to be failing)."
                )
            time.sleep(RETRY_BACKOFF_SECONDS[min(attempt - 1, len(RETRY_BACKOFF_SECONDS) - 1)])

    raise AccurateModelDownloadError(
        "The Accurate-mode language model could not be downloaded. This is "
        "usually a temporary problem with the download server or your internet "
        "connection. Please try again later — nothing is wrong with your "
        "recordings or with StillScript.",
        detail=f"{type(last_error).__name__}: {last_error}",
    ) from last_error


def _download_snapshot(repo_id, revision, target, total_bytes, progress_callback, max_workers,
                       cancel_event=None):
    """LEGACY single-file path (2.1a.2). Fetch the whole repo in one snapshot.

    Kept working, unchanged in behaviour, while the chunked layout proves itself
    in the field. See LAYOUTS in the module docstring for when this is used and
    when it can go.

    CANCELLATION GRANULARITY IS COARSER HERE than on the chunked path. This
    layout has no chunk boundary to check between, so `cancel_event` is only
    checked before the single snapshot_download() call starts — once that is
    running, cancelling has no effect until it finishes or fails on its own.
    This is the pre-existing single-file transport's real limitation, not a
    gap introduced here; it is one more reason 2.3 defaults to layout="chunked".
    """
    target.mkdir(parents=True, exist_ok=True)
    # Once, before the first attempt — not per attempt, so we never race a
    # transfer this loop itself started.
    if is_managed(target):
        _prune_stale_incomplete(target)
    _check_cancelled(cancel_event)

    return _fetch_with_fallback(
        fetch=lambda tqdm_class, _tracker: _snapshot(
            repo_id, revision, target, tqdm_class, max_workers),
        tracker_factory=lambda: _ByteTracker(total_bytes, baseline=_bytes_on_disk(target)),
        progress_callback=progress_callback,
        message="Downloading the Accurate-mode language model…",
        label="download",
        cancel_event=cancel_event,
    )


# ─────────────────────────────────────────────
#  THE CHUNKED DOWNLOAD (masterplan 2.1a.3)
# ─────────────────────────────────────────────
def _hf_file(repo_id, revision, filename, target_dir, tqdm_class=None):
    """Download one file from the repo into `target_dir`, returning its path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id,
        filename=filename,
        revision=revision,
        local_dir=str(target_dir),
        tqdm_class=tqdm_class,
    )


def fetch_manifest(repo_id=REPO_ID, revision=CHUNKED_REVISION, cache_dir=None):
    """Read the chunk manifest for `revision`. Requires network.

    The manifest is the contract between shard_accurate_model.py and this
    module: chunk order, per-chunk size and sha256, and the sha256 of the whole
    reassembled file. Everything downstream is verified against it, so it is
    fetched at the pinned revision like everything else.
    """
    import tempfile

    scratch = cache_dir or tempfile.mkdtemp(prefix="stillscript_manifest_")
    path = _hf_file(repo_id, revision, f"{REPO_CHUNK_DIR}/{MANIFEST_NAME}", scratch)
    with open(path) as fh:
        manifest = json.load(fh)

    # A manifest we cannot understand is not a manifest we may act on.
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise AccurateModelDownloadError(
            "This version of StillScript cannot read the model download "
            "information from the server. Please update StillScript.",
            detail=f"manifest schema {manifest.get('schema')!r}, expected {MANIFEST_SCHEMA}",
            can_retry=False,
        )
    for key in ("original_filename", "total_bytes", "sha256", "chunks"):
        if key not in manifest:
            raise AccurateModelDownloadError(
                "The model download information from the server is incomplete. "
                "Please try again later.",
                detail=f"manifest missing {key!r}",
            )
    return manifest


def _sha256_file(path, chunk=8 * 1024 * 1024, progress=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
            if progress is not None:
                progress(len(block))
    return h.hexdigest()


def _chunk_ok(path, entry):
    """Is this chunk fully present and exactly right?

    Size first because it is free and rules out the common case (a chunk that
    was never fetched, or a truncated one). sha256 second because size alone
    would accept a chunk whose bytes are wrong, and a wrong byte here becomes a
    wrong byte in the model.
    """
    try:
        if os.path.getsize(path) != entry["bytes"]:
            return False
    except OSError:
        return False
    return _sha256_file(path) == entry["sha256"]


def _existing_chunks(chunk_dir, manifest, progress_callback=None):
    """Return the set of chunk names already present and verified.

    On a fresh run this is empty and costs nothing. On a restart after a kill it
    is where the saving comes from, so it is worth the ~20 s of hashing: 20
    seconds against re-fetching gigabytes.
    """
    present = set()
    total = len(manifest["chunks"])
    for index, entry in enumerate(manifest["chunks"], start=1):
        path = os.path.join(chunk_dir, entry["name"])
        if not os.path.exists(path):
            continue
        if progress_callback is not None:
            _emit(progress_callback, DownloadProgress(
                phase="verifying",
                message="Checking which parts are already downloaded…",
                fraction=index / total,
            ))
        if _chunk_ok(path, entry):
            present.add(entry["name"])
        else:
            # Present but wrong: a corrupt or truncated leftover is worse than
            # nothing, because the skip logic would otherwise trust its size.
            logger.warning("Discarding bad chunk %s", entry["name"])
            try:
                os.remove(path)
            except OSError:
                pass
    return present


def _fetch_chunks(repo_id, revision, manifest, target, chunk_dir, present, tqdm_class, tracker,
                  cancel_event=None):
    """Download every chunk not already present. Verifies each on arrival.

    `cancel_event` is checked once per loop iteration, BEFORE starting the next
    chunk's `_hf_file()` call — never mid-chunk. This is the real cancellation
    point for the chunked layout: a chunk already in flight always finishes (or
    fails on its own), and only the NEXT one is skipped. Whatever chunks are
    already in `present` (this call and previous ones) are left on disk exactly
    as they are — cancelling never deletes anything, by construction: this
    function has no delete-on-cancel code path, only the ordinary
    delete-on-corruption one a few lines below.
    """
    for entry in manifest["chunks"]:
        name = entry["name"]
        if name in present:
            continue
        _check_cancelled(cancel_event)
        # local_dir is the model directory, and the repo path is "chunks/<name>",
        # so huggingface_hub writes straight into <target>/chunks/ — which is
        # where we want it. No moving files around afterwards.
        path = _hf_file(repo_id, revision, f"{REPO_CHUNK_DIR}/{name}", target, tqdm_class)
        if not _chunk_ok(path, entry):
            # Verify immediately rather than at the end: a bad chunk caught now
            # is one chunk to re-fetch, and the retry loop above will do exactly
            # that on the next attempt.
            try:
                os.remove(path)
            except OSError:
                pass
            raise AccurateModelDownloadError(
                "Part of the language model did not download correctly. "
                "Please try again.",
                detail=f"chunk {name} failed verification immediately after download",
            )
        tracker.file_done(entry["bytes"])
        present.add(name)
    return present


def _reassemble(chunk_dir, manifest, destination, progress_callback=None):
    """Concatenate the chunks into `destination`, verifying the result.

    Writes to a temporary file and renames only once the sha256 matches, so a
    half-written model.safetensors never exists at the real path — the guard, or
    a later run, must never be able to find one.
    """
    total = manifest["total_bytes"]
    tmp = f"{destination}.reassembling"
    h = hashlib.sha256()
    written = 0
    started = time.monotonic()

    logger.info("Reassembling %d chunks into %s", len(manifest["chunks"]), destination)
    try:
        with open(tmp, "wb") as out:
            for entry in manifest["chunks"]:
                path = os.path.join(chunk_dir, entry["name"])
                with open(path, "rb") as src:
                    while True:
                        block = src.read(8 * 1024 * 1024)
                        if not block:
                            break
                        out.write(block)
                        h.update(block)
                        written += len(block)
                        if progress_callback is not None and written % (64 * 1024 * 1024) < len(block):
                            _emit(progress_callback, DownloadProgress(
                                phase="assembling",
                                message="Putting the language model together…",
                                downloaded_bytes=written, total_bytes=total,
                                fraction=written / total if total else 0.0,
                            ))
        if written != total:
            raise AccurateModelDownloadError(
                "The language model was not put together correctly. Please try again.",
                detail=f"reassembled {written} bytes, manifest says {total}",
            )
        if h.hexdigest() != manifest["sha256"]:
            raise AccurateModelDownloadError(
                "The language model was not put together correctly. Please try again.",
                detail=(f"reassembled sha256 {h.hexdigest()} != manifest "
                        f"{manifest['sha256']}"),
            )
    except Exception:
        # The partial output is worthless and must not survive. The CHUNKS are
        # deliberately left alone — they were each verified on arrival, so the
        # problem is almost certainly here, not in them, and re-fetching 5.75 GiB
        # to fix a failed concatenation would be absurd.
        _unlink_quietly(tmp)
        raise

    os.replace(tmp, destination)
    logger.info("Reassembled %s (%d bytes) in %.1fs, sha256 verified",
                destination, written, time.monotonic() - started)
    return destination


def _unlink_quietly(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def _discard_chunks(target):
    """Delete the chunk staging directory. Returns bytes reclaimed.

    Called only after the reassembled model has passed the guard. Before that
    point the chunks are the difference between a five-minute retry and a
    three-hour one.
    """
    chunk_dir = Path(target) / REPO_CHUNK_DIR
    if not chunk_dir.is_dir():
        return 0
    reclaimed = sum(p.stat().st_size for p in chunk_dir.rglob("*") if p.is_file())
    shutil.rmtree(chunk_dir, ignore_errors=True)
    return reclaimed


def _download_chunked(repo_id, revision, target, progress_callback, max_workers,
                      cancel_event=None):
    """Fetch the model as chunks, verify, and reassemble. Returns (path, transport).

    The staging directory lives inside the model directory so that deleting the
    model deletes the parts too, and is hidden so nothing walking the model
    directory mistakes a raw byte slice for a weights file.

    `cancel_event`, if given, is a `threading.Event`. Setting it stops the
    download between chunks (see `_fetch_chunks`) and raises
    AccurateModelDownloadCancelled — every chunk fetched so far is left on disk
    untouched, so a later call to `ensure_accurate_model()` with the same
    `cancel_event` unset (or a fresh one) resumes rather than restarts. Checked
    here too, before any network call, so a cancel_event set immediately after
    the consent dialog is honoured without even asking the Hub for the manifest.
    """
    target.mkdir(parents=True, exist_ok=True)
    if is_managed(target):
        _prune_stale_incomplete(target)
    _check_cancelled(cancel_event)

    _emit(progress_callback, DownloadProgress(
        phase="preparing", message="Checking the download…"))
    manifest = fetch_manifest(repo_id, revision, cache_dir=str(target))

    chunk_dir = target / REPO_CHUNK_DIR
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Peak disk is chunks + the reassembled file, both at full size, because the
    # chunks are kept until the guard has passed (see the module docstring).
    _check_disk_space(target, manifest["total_bytes"] * 2)

    present = _existing_chunks(str(chunk_dir), manifest, progress_callback)
    already = sum(e["bytes"] for e in manifest["chunks"] if e["name"] in present)
    if present:
        logger.info("Resuming: %d/%d chunks already present (%.2f GiB)",
                    len(present), len(manifest["chunks"]), already / 2 ** 30)

    # The small files (config.json, tokenizer, …) come from the same commit.
    # Everything about the weights is excluded — the monolithic file because we
    # are deliberately not fetching 5.75 GiB twice, the chunks because they are
    # handled below with their own resume logic.
    def fetch(tqdm_class, tracker):
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id,
            revision=revision,
            local_dir=str(target),
            max_workers=max_workers,
            tqdm_class=tqdm_class,
            ignore_patterns=[ORIGINAL_WEIGHTS_FILENAME, f"{REPO_CHUNK_DIR}/*"],
        )
        return _fetch_chunks(repo_id, revision, manifest, target, str(chunk_dir),
                             present, tqdm_class, tracker, cancel_event=cancel_event)

    total_bytes = manifest["total_bytes"]
    _, transport = _fetch_with_fallback(
        fetch=fetch,
        tracker_factory=lambda: _ByteTracker(
            total_bytes,
            baseline=sum(e["bytes"] for e in manifest["chunks"] if e["name"] in present),
        ),
        progress_callback=progress_callback,
        message="Downloading the Accurate-mode language model…",
        label="chunked download",
        cancel_event=cancel_event,
    )

    destination = str(target / manifest["original_filename"])
    _reassemble(str(chunk_dir), manifest, destination, progress_callback)
    # Return the DIRECTORY, not the reassembled file. Everything downstream —
    # ensure_accurate_model()'s return value, the guard, and
    # accurate_engine.transcribe(model_dir=…) / from_pretrained() — works on the
    # model directory, and _download_snapshot() returns a directory too. Handing
    # back the file path here made ensure_accurate_model() return
    # ".../accurate-af-large-v3/model.safetensors", which would have broken Wave
    # 2.3 at the first call. Caught by the real end-to-end download; the unit
    # tests missed it because their stubs echoed the same wrong shape.
    return str(target), transport


def _check_disk_space(target, needed):
    """Fail early and clearly rather than filling the disk halfway through."""
    try:
        free = shutil.disk_usage(str(target)).free
    except OSError:
        return
    if free >= needed:
        return
    raise AccurateModelDownloadError(
        "There is not enough free space on this computer to download the "
        f"Accurate-mode language model. It needs about {needed / 2 ** 30:.0f} GB "
        f"free while it installs (about {needed / 2 ** 31:.0f} GB once finished), "
        f"and there is {free / 2 ** 30:.1f} GB available. Free up some space and "
        "try again.",
        detail=f"need {needed} bytes, have {free}",
        can_retry=False,
    )


def _friendly_transport_error(exc):
    """Turn a non-retryable failure into something a user can act on."""
    chain = _exception_chain(exc)
    blob = " | ".join(f"{type(e).__name__}: {e}" for e in chain).lower()
    names = {type(e).__name__ for e in chain}

    if any(isinstance(e, OSError) and e.errno in _FATAL_ERRNOS for e in chain) or \
            any(marker in blob for marker in _FATAL_MARKERS):
        return AccurateModelDownloadError(
            "There is not enough free space on this computer to download the "
            "Accurate-mode language model. It needs about 6 GB. Free up some "
            "space and try again.",
            detail=f"{type(exc).__name__}: {exc}",
            can_retry=False,
        )
    if names & {"RepositoryNotFoundError", "RevisionNotFoundError", "GatedRepoError",
                "DisabledRepoError", "EntryNotFoundError", "RemoteEntryNotFoundError"}:
        return AccurateModelDownloadError(
            "The Accurate-mode language model is not available for download at "
            "the moment. This is a problem on our side, not yours — please "
            "report it, and use Fast mode in the meantime.",
            detail=f"{type(exc).__name__}: {exc}",
            can_retry=False,
        )
    return AccurateModelDownloadError(
        "The Accurate-mode language model could not be downloaded. Please try "
        "again later.",
        detail=f"{type(exc).__name__}: {exc}",
    )


def remove_downloaded_model(model_dir):
    """Delete a managed download. Refuses to touch anything it does not own.

    The refusal matters: `resolve_model_dir()` can return a path the user chose,
    and a corrupted-download cleanup must never turn into deleting a directory
    somebody pointed us at.

    Returns True only if the directory is genuinely gone afterwards. Claiming a
    deletion that did not happen is the one outcome worse than not deleting: the
    caller would tell the user the bad download was removed while a
    half-verified model sits there waiting for the next run to find it. Windows
    makes this a real possibility — a 5.75 GiB model.safetensors that is still
    memory-mapped by a previous load cannot be unlinked.
    """
    if not is_managed(model_dir):
        logger.warning("Refusing to delete %s — not the managed download directory.", model_dir)
        return False
    shutil.rmtree(model_dir, ignore_errors=True)
    if os.path.exists(model_dir):
        logger.error(
            "Could not delete the downloaded model directory %s — it is still "
            "on disk. It has no completion stamp, so it will not be trusted, "
            "but it is taking up space.", model_dir,
        )
        return False
    logger.info("Deleted downloaded model directory %s", model_dir)
    return True


def ensure_accurate_model(
    *,
    model_dir=None,
    progress_callback=None,
    repo_id=REPO_ID,
    revision=None,
    force=False,
    max_workers=8,
    layout=DEFAULT_LAYOUT,
    cancel_event=None,
):
    """Make sure the Accurate model is on this machine, and return its directory.

    This is the one function masterplan 2.3 needs. It is safe to call every time
    Accurate mode is selected: when the model is already present and verified it
    returns immediately without touching the network.

        path = ensure_accurate_model(progress_callback=hook)
        accurate_engine.transcribe(audio_path, model_dir=path)

    Arguments
      model_dir          Where the model should be. Defaults to
                         STILLSCRIPT_ACCURATE_MODEL_DIR, then the managed
                         location (~/.danscribe_models/accurate-af-large-v3).
      progress_callback  Called with DownloadProgress objects, roughly twice a
                         second. Optional. Calls are never concurrent, so the
                         hook may assume it is not re-entered; during the
                         transfer they arrive on a background thread, so a UI
                         must marshal to its own. Exceptions raised by the hook
                         are logged and ignored — a UI bug cannot kill a
                         three-hour download.
      force              Re-download even if a verified copy is present.
      layout             "chunked" (default, masterplan 2.1a.3) fetches ~200 MiB
                         parts and reassembles them, so an interruption costs at
                         most one part. "single" is 2.1a.2's one-file path, kept
                         as a fallback until the chunked path has proven itself.
      repo_id, revision  Overridable for tests. Revision defaults to the pin that
                         matches `layout`; do not pass "main".
      cancel_event       Optional threading.Event. Set it to stop an in-progress
                         download early. On the chunked layout (the default)
                         this takes effect between chunks — whatever chunks have
                         already landed are left on disk, untouched, so a later
                         call resumes rather than restarts. Raises
                         AccurateModelDownloadCancelled, which callers should
                         catch BEFORE AccurateModelDownloadError so a deliberate
                         cancel is never shown as a failure. Checked before any
                         network call too, so setting it before this function is
                         even entered is honoured immediately. Never checked once
                         `verify_merged_model()` starts — a download that has
                         already finished should be allowed to verify and land
                         cleanly rather than being thrown away on a late click.

    Behaviour
      - Already downloaded and verified → returns at once.
      - A directory the user supplied (not our managed one) that already has
        model files → accepted as-is and returned. We do not manage, verify-then-
        delete, or re-download somebody else's directory; accurate_engine's own
        guard still runs against it at load time.
      - Otherwise → download (resuming what it can), then verify with the guard
        in FULL mode, then stamp.

    Raises
      AccurateModelDownloadError with a message written for a non-technical user.
      A guard failure on freshly downloaded weights deletes the download first,
      so a later run cannot pick up a partially-verified model.
    """
    if layout not in ("chunked", "single"):
        raise ValueError(f"unknown layout {layout!r}")
    if revision is None:
        revision = CHUNKED_REVISION if layout == "chunked" else REVISION

    target = Path(resolve_model_dir(model_dir))
    managed = is_managed(target)

    if not force and is_verified_download(target, repo_id, revision):
        logger.info("Accurate model already present and verified at %s", target)
        _emit(progress_callback, DownloadProgress(
            phase="done", message="The Accurate-mode language model is ready.",
            fraction=1.0,
        ))
        return str(target)

    if not managed and has_model_files(target):
        # An explicitly configured directory. Not ours to verify-and-delete.
        logger.info("Using the configured Accurate model directory %s (not managed here).", target)
        _emit(progress_callback, DownloadProgress(
            phase="done", message="The Accurate-mode language model is ready.",
            fraction=1.0,
        ))
        return str(target)

    if force and managed:
        remove_downloaded_model(target)

    _emit(progress_callback, DownloadProgress(
        phase="preparing", message="Checking the download…",
    ))
    # Checked before the first network call, not just inside the download
    # dispatch below — a cancel_event set before this function was even
    # entered (or reused from a previous, already-cancelled attempt by
    # mistake) must not cost a Hub round trip.
    _check_cancelled(cancel_event)

    try:
        info = describe_download(repo_id, revision, layout=layout)
    except Exception as exc:  # noqa: BLE001
        if not _hub_reachable():
            raise AccurateModelDownloadError(
                "StillScript needs to download the Accurate-mode language model "
                "the first time you use it, and could not reach the internet. "
                "Connect to the internet and try again. (Fast mode works "
                "offline.)",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        raise _friendly_transport_error(exc) from exc

    total_bytes = info["total_bytes"]
    logger.info(
        "Accurate model download (%s layout): %s @ %s, %d files, %.2f GiB -> %s",
        layout, repo_id, revision[:7], info["file_count"], total_bytes / 2 ** 30, target,
    )

    if layout == "chunked":
        path, transport = _download_chunked(
            repo_id, revision, target, progress_callback, max_workers,
            cancel_event=cancel_event,
        )
    else:
        path, transport = _download_snapshot(
            repo_id, revision, target, total_bytes, progress_callback, max_workers,
            cancel_event=cancel_event,
        )

    # ── Full verification (masterplan 2.2, in the one mode that is
    # unconditionally right for a fresh download — see the module docstring).
    _emit(progress_callback, DownloadProgress(
        phase="verifying",
        message="Checking the downloaded model is complete and correct…",
        downloaded_bytes=total_bytes, total_bytes=total_bytes, fraction=1.0,
    ))
    try:
        guard_report = verify_merged_model(str(target), full=True)
    except AccurateModelGuardError as exc:
        logger.error("Downloaded Accurate model failed FULL verification: %s", exc)
        # Three genuinely different situations, and the user needs to be told
        # which one they are in — "we cleaned it up, retry" is a lie if the
        # directory is still sitting there.
        if not managed:
            message = (
                "The model in this folder did not pass verification and cannot be "
                f"used:\n\n{target}\n\nStillScript did not download this folder, "
                "so it has been left alone. Remove or replace it and try again."
            )
        elif layout == "chunked":
            # Delete ONLY the reassembled file. Every chunk was verified against
            # the manifest as it arrived, so the downloaded parts are almost
            # certainly fine and the fault is in the assembly or on disk.
            # Throwing away 5.75 GiB of verified parts to re-fetch them over
            # three hours would be the wrong reflex.
            _unlink_quietly(str(target / ORIGINAL_WEIGHTS_FILENAME))
            message = (
                "The Accurate-mode language model did not pass its final check "
                "and has been removed. The downloaded parts have been kept, so "
                "trying again will be much quicker than the first download."
            )
        elif remove_downloaded_model(target):
            message = (
                "The Accurate-mode language model did not download correctly and "
                "has been removed. This is almost always a network problem. "
                "Please try again."
            )
        else:
            message = (
                "The Accurate-mode language model did not download correctly, and "
                "the incomplete copy could not be deleted automatically — another "
                "program may still be using it. Close StillScript, delete this "
                f"folder, and try again:\n\n{target}"
            )
        raise AccurateModelDownloadError(
            message,
            detail=str(exc),
            guard_failures=list(getattr(exc, "failures", []) or []),
        ) from exc

    logger.info(
        "Accurate model downloaded via %s and FULL-verified: %s", transport, guard_report,
    )

    # Only now are the chunks expendable. Until the guard has passed they are the
    # cheap way back from any failure, so this is the one and only place they get
    # deleted — and it happens after verification, never before.
    if layout == "chunked":
        reclaimed = _discard_chunks(target)
        if reclaimed:
            logger.info("Removed %.2f GiB of downloaded parts now that the model "
                        "is verified.", reclaimed / 2 ** 30)

    if managed:
        _write_stamp(target, repo_id, revision, guard_report, total_bytes)

    _emit(progress_callback, DownloadProgress(
        phase="done",
        message="The Accurate-mode language model is ready.",
        downloaded_bytes=total_bytes, total_bytes=total_bytes, fraction=1.0,
    ))
    return str(path if path else target)
