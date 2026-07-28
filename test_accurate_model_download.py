"""Tests for the Accurate-mode download mechanism (masterplan 2.1a.2).

Run with the spike venv (it has huggingface_hub + torch + safetensors):
    ~/whisper_afrikaans_spike/venv/bin/python3 test_accurate_model_download.py

WHAT IS AND IS NOT SIMULATED
────────────────────────────
The two things this task exists to prove — the Xet fallback and resume — are
tested against the real network, not mocked end to end:

  * The Xet fallback test injects a real Xet failure (the actual error text
    xet-core produces) and then lets everything downstream run for real: the
    classification, the reachability probe, the transport switch, and a genuine
    `snapshot_download` over the network on the retry. Only the backend's
    refusal is synthetic, because there is no way to make Hugging Face's Xet
    service fail on demand.

  * The resume test really downloads a repo, really deletes part of it, and
    really downloads again, asserting from the byte counter that only the
    missing part crossed the wire.

Both use `hf-internal-testing/tiny-random-WhisperForConditionalGeneration`
(~8 MiB) rather than the 5.75 GiB production model, so the suite runs in under a
minute. The production model is exercised separately by a real full download —
see the notes in the delivery summary.

The guard-failure test uses a synthetic corrupt model built the same way
test_accurate_guard.py builds its fixtures, so it needs no download at all.

SAFETY
──────
Several tests exercise deletion of the managed model directory. STILLSCRIPT_MODEL_ROOT
is redirected to a temp directory before anything else imports the module, and
the redirection is asserted before the first destructive test runs. A real
download in ~/.danscribe_models must never be at risk from running the tests.
"""

import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── SAFETY: redirect the managed root BEFORE importing the module ────────────
TEST_ROOT = tempfile.mkdtemp(prefix="stillscript_dl_test_")
os.environ["STILLSCRIPT_MODEL_ROOT"] = TEST_ROOT
os.environ.pop("STILLSCRIPT_ACCURATE_MODEL_DIR", None)
os.environ.pop("STILLSCRIPT_ACCURATE_FULL_VERIFY", None)

import logging  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

import accurate_guard  # noqa: E402
import accurate_model_download as amd  # noqa: E402
from accurate_fingerprint import ACCURATE_FINGERPRINT  # noqa: E402

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


# The whole suite is meaningless (and dangerous) if this is not true.
assert str(amd.managed_model_dir()).startswith(TEST_ROOT), (
    f"managed_model_dir() is {amd.managed_model_dir()}, not under {TEST_ROOT} — "
    f"refusing to run destructive tests against a real download."
)

TINY_REPO = "hf-internal-testing/tiny-random-WhisperForConditionalGeneration"


def reset_managed():
    shutil.rmtree(amd.managed_model_dir(), ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 1. Paths follow the app's existing convention ===")
# ════════════════════════════════════════════════════════════════════════════
saved_root = os.environ.pop("STILLSCRIPT_MODEL_ROOT")
check("default root is ~/.danscribe_models (matches ~/.danscribe.log / "
      "~/.danscribe_config.json, so masterplan 4.1's rename sweep catches it)",
      str(amd.default_model_root()) == os.path.join(os.path.expanduser("~"), ".danscribe_models"),
      str(amd.default_model_root()))
os.environ["STILLSCRIPT_MODEL_ROOT"] = saved_root

check("STILLSCRIPT_MODEL_ROOT overrides the root",
      str(amd.default_model_root()) == TEST_ROOT)

os.environ["STILLSCRIPT_ACCURATE_MODEL_DIR"] = "/some/where/else"
check("STILLSCRIPT_ACCURATE_MODEL_DIR wins over the managed default "
      "(same resolution order as accurate_engine)",
      str(amd.resolve_model_dir()) == "/some/where/else")
check("an explicit argument wins over the environment",
      str(amd.resolve_model_dir("/explicit")) == "/explicit")
del os.environ["STILLSCRIPT_ACCURATE_MODEL_DIR"]

check("resolve_model_dir() falls back to the managed download location",
      amd.resolve_model_dir() == amd.managed_model_dir())


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 2. Only the managed directory may ever be deleted ===")
# ════════════════════════════════════════════════════════════════════════════
foreign = os.path.join(TEST_ROOT, "not_ours")
os.makedirs(foreign, exist_ok=True)
open(os.path.join(foreign, "precious.bin"), "w").write("do not delete me")

check("is_managed() is False for a directory the user pointed us at",
      amd.is_managed(foreign) is False)
check("remove_downloaded_model() refuses a directory it does not own",
      amd.remove_downloaded_model(foreign) is False)
check("...and the directory is still there",
      os.path.isfile(os.path.join(foreign, "precious.bin")))

reset_managed()
os.makedirs(amd.managed_model_dir(), exist_ok=True)
open(os.path.join(amd.managed_model_dir(), "x"), "w").write("x")
check("remove_downloaded_model() deletes the managed directory",
      amd.remove_downloaded_model(amd.managed_model_dir()) is True
      and not os.path.exists(amd.managed_model_dir()))


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 3. Failure classification: retry the transport, surface the rest ===")
# ════════════════════════════════════════════════════════════════════════════
# The real text xet-core produces when the CAS backend refuses a connection.
# This is the case the whole fallback exists for; if the classifier stops
# recognising it, the fallback silently stops happening.
XET_CONNECTION_ERROR = ConnectionError(
    "Data transfer error: error sending request for url "
    "(https://cas-server.xethub.hf.co/v1/reconstructions/"
    "04c9711818700e4b7e0826dd0f9ed0f967ce1b010549f557981ea22403bf4634): "
    "client error (Connect)"
)
XET_RUNTIME_ERROR = RuntimeError(
    "Xet Runtime Error: CAS service error : reqwest::Error "
    "{ kind: Request, url: \"https://cas-server.xethub.hf.co/v1/chunks\", "
    "source: hyper_util::client::legacy::Error(Connect, ConnectError(...)) }"
)


class _FakeRepoNotFound(Exception):
    pass


_FakeRepoNotFound.__name__ = "RepositoryNotFoundError"

for label, exc, expected in (
    ("a Xet ConnectionError is retryable", XET_CONNECTION_ERROR, True),
    ("a Xet RuntimeError naming cas-server is retryable", XET_RUNTIME_ERROR, True),
    ("a bare ConnectionError is retryable", ConnectionError("connection reset"), True),
    ("a timeout is retryable", TimeoutError("read timed out"), True),
    ("disk full (ENOSPC) is NOT retryable",
     OSError(28, "No space left on device"), False),
    ("huggingface_hub's own disk-space message is NOT retryable",
     ValueError("Not enough free disk space to download the file."), False),
    ("a permission error is NOT retryable", OSError(13, "Permission denied"), False),
    ("a missing repo is NOT retryable", _FakeRepoNotFound("404"), False),
    ("an ordinary bug is NOT retryable", ValueError("bad argument"), False),
):
    check(label, amd._is_retryable_transport_failure(exc) is expected)

# A Xet failure wrapped in something else must still be seen.
try:
    try:
        raise XET_CONNECTION_ERROR
    except ConnectionError as inner:
        raise RuntimeError("download failed") from inner
except RuntimeError as wrapped:
    check("a Xet failure wrapped in another exception is still recognised",
          amd._is_retryable_transport_failure(wrapped) is True)

# A disk-full error must win even if a Xet mention is somewhere in the chain:
# retrying a full disk on a different backend is hours wasted for nothing.
try:
    try:
        raise OSError(28, "No space left on device")
    except OSError as inner:
        raise RuntimeError("xet transfer aborted") from inner
except RuntimeError as wrapped:
    check("disk-full beats a Xet mention in the same chain",
          amd._is_retryable_transport_failure(wrapped) is False)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 4. _xet_disabled() really disables Xet ===")
# ════════════════════════════════════════════════════════════════════════════
# This is the mechanism the whole fallback rests on, and the obvious way to
# write it (set os.environ and hope) does NOT work: huggingface_hub reads
# HF_HUB_DISABLE_XET into a module constant at import time.
from huggingface_hub import constants as hf_constants  # noqa: E402
from huggingface_hub.utils._runtime import is_xet_available  # noqa: E402

try:
    import hf_xet  # noqa: F401

    have_xet = True
except ImportError:
    have_xet = False

if have_xet:
    check("hf_xet is installed, so Xet is live before the switch",
          is_xet_available() is True)

prev_env = os.environ.get("HF_HUB_DISABLE_XET")
with amd._xet_disabled():
    check("inside the block, huggingface_hub reports Xet unavailable",
          is_xet_available() is False)
    check("...via the module constant, which is what actually takes effect",
          hf_constants.HF_HUB_DISABLE_XET is True)
    check("...and the environment variable agrees, for anything imported later",
          os.environ.get("HF_HUB_DISABLE_XET") == "1")

check("the constant is restored on exit",
      is_xet_available() is have_xet)
check("the environment is restored on exit",
      os.environ.get("HF_HUB_DISABLE_XET") == prev_env)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 5. Progress reporting ===")
# ════════════════════════════════════════════════════════════════════════════
tracker = amd._ByteTracker(total_bytes=1000, baseline=100)
tracker.add("reconstruct", 300)
tracker.add("reconstruct::transfer", 250)
check("the byte counter takes the larger of the two transports' counters",
      tracker.transferred() == 300, str(tracker.transferred()))
check("progress is absolute — the resume baseline is included",
      tracker.downloaded() == 400, str(tracker.downloaded()))
tracker.add("reconstruct", 5000)
check("progress never reports more than the total (a >100% bar looks broken)",
      tracker.downloaded() == 1000)

# Rate must be derived from what moved in THIS attempt. Counting the resume
# baseline made the first sample of a resumed download claim 10 MiB/s on a
# 0.6 MiB/s connection, and the ETA that followed was fiction.
resumed = amd._ByteTracker(total_bytes=1_000_000, baseline=900_000)
time.sleep(0.05)
done, rate, eta = resumed.sample()
check("a resumed download shows its true position immediately",
      done == 900_000, str(done))
check("...without claiming it downloaded the baseline at infinite speed",
      rate == 0.0, f"{rate:.0f} B/s")
check("...and offers no ETA until something has actually moved", eta is None)

# Xet writes to disk in ~64 MiB blocks, so bytes-on-disk sits flat for tens of
# seconds mid-transfer. The rate must not read as a stall during those pauses.
chunky = amd._ByteTracker(total_bytes=100 << 20)
chunky.add("reconstruct", 8 << 20)
time.sleep(0.3)
_, rate_moving, eta_moving = chunky.sample()
time.sleep(0.6)                      # a flat stretch: no new bytes reported
_, rate_flat, eta_flat = chunky.sample()
check("the rate survives a flat stretch instead of collapsing to zero",
      rate_flat > rate_moving * 0.2 and rate_flat > 0,
      f"{rate_moving / 2 ** 20:.1f} -> {rate_flat / 2 ** 20:.1f} MiB/s")
check("...so the ETA stays available rather than showing '?' mid-download",
      eta_moving is not None and eta_flat is not None)

# The tqdm stand-in is what huggingface_hub actually drives.
tracker = amd._ByteTracker(total_bytes=1000)
Sink = amd._make_tqdm_class(tracker)
byte_bar = Sink(total=1000, unit="B", name="huggingface_hub.snapshot_download")
file_bar = Sink(total=13, unit="it", name=None)
byte_bar.update(400)
file_bar.update(3)          # the "Fetching 13 files" bar must not count as bytes
byte_bar.update_transfer(120)
check("byte-unit bars are counted", tracker.transferred() == 400, str(tracker.transferred()))
check("the file-count bar is not mistaken for bytes",
      tracker.downloaded() == 400, str(tracker.downloaded()))
check("progress bars are silenced (no console output from a GUI app)",
      byte_bar.disable is True)

# Callbacks arrive on one thread, and a broken callback cannot kill a download.
seen = []
threads = set()
concurrent = []
in_flight = threading.Semaphore(1)


def noisy_callback(p):
    if not in_flight.acquire(blocking=False):
        concurrent.append(p)
    else:
        time.sleep(0.05)          # widen the window a real UI hook would occupy
        in_flight.release()
    threads.add(threading.current_thread().name)
    seen.append(p)
    raise RuntimeError("this UI hook is broken")


tracker = amd._ByteTracker(total_bytes=1000)
with amd._ProgressReporter(tracker, noisy_callback, "downloading") as reporter:
    tracker.add("reconstruct", 500)
    time.sleep(1.3)
reporter.emit_now()               # the final sample, as _download_snapshot sends it

check("progress was emitted", len(seen) >= 2, f"{len(seen)} samples")
check("...all from the reporter's own thread while the download runs",
      len(threads - {threading.current_thread().name}) == 1, str(threads))
check("...never two callbacks at once, including the final sample "
      "(a UI hook may assume it is not re-entered)",
      not concurrent, f"{len(concurrent)} overlapping calls")
check("...and an exception in the caller's callback was swallowed", True)
last = seen[-1]
check("the sample carries what a UI needs",
      last.phase == "downloading" and last.total_bytes == 1000
      and 0 < last.fraction <= 1.0 and last.downloaded_bytes == 500,
      f"{last}")
check("DownloadProgress is serialisable for logging", "phase" in last.as_dict())


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 6. Xet failure → automatic fallback → real successful download ===")
# ════════════════════════════════════════════════════════════════════════════
# Attempt 1 raises the real Xet error text. Everything after that is real:
# classification, the hub-reachability probe, the transport switch, and an
# actual network download on attempt 2.
reset_managed()
attempts = []
real_snapshot = amd._snapshot


def failing_then_real(repo_id, revision, target, tqdm_class, max_workers):
    attempts.append({
        "n": len(attempts) + 1,
        "xet_disabled": hf_constants.HF_HUB_DISABLE_XET,
        "xet_available": is_xet_available(),
    })
    if len(attempts) == 1:
        raise XET_CONNECTION_ERROR
    return real_snapshot(repo_id, revision, target, tqdm_class, max_workers)


amd._snapshot = failing_then_real
progress = []
try:
    t0 = time.time()
    path, transport = amd._download_snapshot(
        TINY_REPO, "main", amd.managed_model_dir(),
        total_bytes=8_462_000, progress_callback=progress.append, max_workers=4,
    )
    elapsed = time.time() - t0
    check("the download succeeded after the Xet failure", os.path.isdir(path), path)
    check("it took two attempts", len(attempts) == 2, str(len(attempts)))
    check("attempt 1 ran with Xet enabled", attempts[0]["xet_available"] is have_xet)
    check("attempt 2 ran with Xet disabled — the fallback actually flipped the switch",
          attempts[1]["xet_disabled"] is True and attempts[1]["xet_available"] is False)
    check("the reported transport is the HTTPS fallback", transport == "https", transport)
    check("real files landed on disk from the network",
          os.path.isfile(os.path.join(path, "config.json")))
    check("progress was reported during the fallback download", len(progress) >= 1,
          f"{len(progress)} samples in {elapsed:.1f}s")
finally:
    amd._snapshot = real_snapshot

check("Xet is left enabled again afterwards (the switch is scoped, not sticky)",
      is_xet_available() is have_xet)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 7. Non-retryable failures are NOT retried on another transport ===")
# ════════════════════════════════════════════════════════════════════════════
reset_managed()
calls = []


def always_disk_full(*a, **kw):
    calls.append(1)
    raise OSError(28, "No space left on device")


amd._snapshot = always_disk_full
try:
    amd._download_snapshot(TINY_REPO, "main", amd.managed_model_dir(),
                           total_bytes=1000, progress_callback=None, max_workers=4)
    check("a full disk stops the download", False, "no exception raised")
except amd.AccurateModelDownloadError as e:
    check("a full disk stops the download immediately", len(calls) == 1,
          f"{len(calls)} attempts")
    check("...with a message about free space, not about storage backends",
          "space" in str(e).lower() and "xet" not in str(e).lower(), str(e))
    check("...marked as not worth retrying", e.can_retry is False)
    check("...with the technical cause kept for the log, off the user's screen",
          "No space left" in (e.detail or ""), str(e.detail))
finally:
    amd._snapshot = real_snapshot


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 8. Resume: an interrupted download does not start over ===")
# ════════════════════════════════════════════════════════════════════════════
# Real download, real deletion of part of it, real second download. The
# assertion is on bytes actually transferred the second time.
reset_managed()
target = amd.managed_model_dir()
amd._download_snapshot(TINY_REPO, "main", target, total_bytes=8_462_000,
                       progress_callback=None, max_workers=4)
full_bytes = amd._bytes_on_disk(target)
check("first download completed", full_bytes > 1_000_000, f"{full_bytes} bytes")

# Simulate the interruption: drop one file, keep the rest.
victim = os.path.join(target, "pytorch_model.bin")
victim_size = os.path.getsize(victim)
os.remove(victim)
after_loss = amd._bytes_on_disk(target)
check("part of the download is now missing",
      after_loss == full_bytes - victim_size, f"{after_loss} vs {full_bytes}")

second = []


def record_snapshot(repo_id, revision, tgt, tqdm_class, max_workers):
    second.append(tqdm_class)
    return real_snapshot(repo_id, revision, tgt, tqdm_class, max_workers)


amd._snapshot = record_snapshot
resumed_progress = []
try:
    amd._download_snapshot(TINY_REPO, "main", target, total_bytes=8_462_000,
                           progress_callback=resumed_progress.append, max_workers=4)
finally:
    amd._snapshot = real_snapshot

check("the resumed download restored the missing file",
      amd._bytes_on_disk(target) == full_bytes, str(amd._bytes_on_disk(target)))

first_sample = resumed_progress[0] if resumed_progress else None
check("progress resumed from what was already on disk, not from zero",
      first_sample is not None and first_sample.downloaded_bytes >= after_loss,
      f"first sample: {first_sample.downloaded_bytes if first_sample else None} "
      f"(bytes already present: {after_loss})")

# The bytes that crossed the wire the second time should be about one file,
# not the whole repo. Generous bound: anything under half the repo proves the
# already-present files were not re-fetched.
transferred = max(p.downloaded_bytes for p in resumed_progress) - after_loss
check("only the missing file was re-downloaded",
      transferred < full_bytes / 2,
      f"{transferred} bytes transferred; the whole repo is {full_bytes}")

# The honest limit, asserted rather than assumed: a partly-transferred file
# leaves nothing reusable behind, so the baseline must never count it. See the
# module docstring for the measurement this comes from.
partial_marker = os.path.join(target, amd._HF_INTERNAL_DIRNAME, "huggingface",
                              "download", "fake.incomplete")
os.makedirs(os.path.dirname(partial_marker), exist_ok=True)
with open(partial_marker, "wb") as fh:
    fh.write(b"\0" * 4096)
check("the resume baseline ignores huggingface_hub's internal partial files "
      "(they are discarded on restart, so counting them would overstate progress)",
      amd._bytes_on_disk(target) == full_bytes, str(amd._bytes_on_disk(target)))

# ...and because they can never be resumed, they must not be left to pile up.
# huggingface_hub deletes them in a `finally`, which a SIGKILL or a power cut
# skips — observed for real while testing, three orphans in one directory.
reclaimed = amd._prune_stale_incomplete(target)
check("unresumable leftovers from a killed run are reclaimed",
      reclaimed == 4096 and not os.path.exists(partial_marker), f"{reclaimed} bytes")
check("...and pruning leaves the real files alone",
      amd._bytes_on_disk(target) == full_bytes)
check("pruning a directory with no leftovers is a no-op",
      amd._prune_stale_incomplete(target) == 0)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 9. Verification runs in FULL mode at this call site ===")
# ════════════════════════════════════════════════════════════════════════════
# Requirement: regardless of STILLSCRIPT_ACCURATE_FULL_VERIFY, a freshly
# downloaded model is verified in full. Probe-only mode is exactly what misses a
# truncated or corrupted download.
recorded = {}
real_verify = amd.verify_merged_model


def spy_verify(model_dir, fingerprint=None, full=None):
    recorded["full"] = full
    recorded["env"] = os.environ.get(accurate_guard.FULL_VERIFY_ENV_VAR)
    return {"ok": True, "full": True, "model_dir": model_dir}


for env_value, label in ((None, "unset"), ("0", "explicitly off"), ("1", "explicitly on")):
    if env_value is None:
        os.environ.pop(accurate_guard.FULL_VERIFY_ENV_VAR, None)
    else:
        os.environ[accurate_guard.FULL_VERIFY_ENV_VAR] = env_value

    # Each iteration must start from "nothing downloaded yet": a successful run
    # writes a completion stamp, and a stamped directory correctly skips both
    # the download and the verification.
    reset_managed()
    target = amd.managed_model_dir()
    os.makedirs(target, exist_ok=True)
    for name in amd.REQUIRED_MODEL_FILES:
        open(os.path.join(target, name), "w").write("{}")

    recorded.clear()
    amd.verify_merged_model = spy_verify
    amd._snapshot = lambda repo_id, revision, tgt, tqdm_class, mw: str(tgt)
    try:
        amd.ensure_accurate_model(repo_id=TINY_REPO, revision="main", force=False,
                                  layout="single")
    finally:
        amd.verify_merged_model = real_verify
        amd._snapshot = real_snapshot
    check(f"full=True is passed explicitly with the env var {label} "
          f"(it must not depend on the guard's global default — masterplan 2.2a)",
          recorded.get("full") is True, str(recorded))

os.environ.pop(accurate_guard.FULL_VERIFY_ENV_VAR, None)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 10. A corrupted download is deleted, not left half-verified ===")
# ════════════════════════════════════════════════════════════════════════════
# A synthetic model that fails the real guard, planted where a download would
# have landed. `_snapshot` is a no-op because the point under test is what
# happens AFTER the bytes arrive.
import base64  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def plant_corrupt_model(directory):
    """Probe tensors present and correctly shaped, but holding base large-v3's
    values — the signature of a model that is not the Afrikaans merge. Stands in
    for a download whose bytes did not survive the trip."""
    os.makedirs(directory, exist_ok=True)
    tensors = {}
    for p in ACCURATE_FINGERPRINT["probes"]:
        base = torch.frombuffer(
            bytearray(base64.b64decode(p["base_sample_b64"])), dtype=torch.float32)
        flat = torch.zeros(int(torch.tensor(p["shape"]).prod()), dtype=torch.float32)
        idx = accurate_guard._sample_indices(flat.numel(), base.numel())
        flat[idx] = base
        tensors[p["key"]] = flat.reshape(p["shape"]).contiguous()
    save_file(tensors, os.path.join(directory, "model.safetensors"))
    for name in ("config.json", "preprocessor_config.json"):
        open(os.path.join(directory, name), "w").write("{}")


test_fp = copy.deepcopy(ACCURATE_FINGERPRINT)
test_fp["tensor_count"] = len(test_fp["probes"])
test_fp["full_manifest"] = {p["key"]: "0" * 64 for p in test_fp["probes"]}


def verify_with_test_fingerprint(model_dir, fingerprint=None, full=None):
    return real_verify(model_dir, test_fp, full=full)


reset_managed()
target = amd.managed_model_dir()
plant_corrupt_model(target)

amd.verify_merged_model = verify_with_test_fingerprint
amd._snapshot = lambda repo_id, revision, tgt, tqdm_class, mw: str(tgt)
try:
    amd.ensure_accurate_model(repo_id=TINY_REPO, revision="main", layout="single")
    check("a corrupt download is rejected", False, "no exception raised")
except amd.AccurateModelDownloadError as e:
    check("a corrupt download is rejected", True)
    check("the directory is gone — nothing half-verified is left for a later run "
          "to pick up", not os.path.exists(target), str(target))
    check("the user is told to retry, in plain language",
          "try again" in str(e).lower() and "tensor" not in str(e).lower(), str(e))
    check("no traceback jargon reaches the user",
          not any(w in str(e).lower() for w in ("sha256", "safetensors", "xet", "traceback")),
          str(e))
    check("the guard's failed checks are kept for the log",
          len(e.guard_failures) > 0, f"{len(e.guard_failures)} failures recorded")
finally:
    amd.verify_merged_model = real_verify
    amd._snapshot = real_snapshot

check("no completion stamp survived the failure",
      amd.read_stamp(amd.managed_model_dir()) is None)
check("the directory is not treated as a verified download afterwards",
      amd.is_verified_download(amd.managed_model_dir()) is False)

# If the cleanup itself fails — a memory-mapped model.safetensors on Windows is
# the realistic case — the user must be told the truth, not "it has been
# removed". A message that claims a deletion that did not happen leaves them
# with a bad model and no reason to look for it.
reset_managed()
target = amd.managed_model_dir()
plant_corrupt_model(target)
real_rmtree = shutil.rmtree
amd.verify_merged_model = verify_with_test_fingerprint
amd._snapshot = lambda repo_id, revision, tgt, tqdm_class, mw: str(tgt)
amd.shutil.rmtree = lambda *a, **kw: None      # deletion silently does nothing
try:
    amd.ensure_accurate_model(repo_id=TINY_REPO, revision="main", layout="single")
    check("an undeletable corrupt download still raises", False, "no exception")
except amd.AccurateModelDownloadError as e:
    check("an undeletable corrupt download still raises", True)
    check("...and does NOT claim the copy was removed when it was not",
          "has been removed" not in str(e), str(e))
    check("...but tells the user where it is and what to do",
          str(target) in str(e) and "delete this folder" in str(e).lower(), str(e))
finally:
    amd.shutil.rmtree = real_rmtree
    amd.verify_merged_model = real_verify
    amd._snapshot = real_snapshot

check("remove_downloaded_model() reports failure rather than a false success",
      os.path.exists(target))
check("...and the leftover is still not trusted (no stamp)",
      amd.is_verified_download(target) is False)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 11. A directory the user supplied is never deleted ===")
# ════════════════════════════════════════════════════════════════════════════
# Same corrupt model, but somewhere the user pointed us at. Cleanup must not
# apply: we did not download it, so it is not ours to remove.
user_dir = os.path.join(TEST_ROOT, "user_supplied")
plant_corrupt_model(user_dir)
os.environ["STILLSCRIPT_ACCURATE_MODEL_DIR"] = user_dir
try:
    result = amd.ensure_accurate_model()
    check("a configured directory with model files is accepted as-is "
          "(accurate_engine's own guard still checks it at load time)",
          result == user_dir, result)
    check("...and is still on disk", os.path.isfile(os.path.join(user_dir, "model.safetensors")))
finally:
    del os.environ["STILLSCRIPT_ACCURATE_MODEL_DIR"]


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 12. The completion stamp gates re-downloading ===")
# ════════════════════════════════════════════════════════════════════════════
reset_managed()
target = amd.managed_model_dir()
os.makedirs(target, exist_ok=True)
for name in amd.REQUIRED_MODEL_FILES:
    open(os.path.join(target, name), "w").write("{}")

check("model files alone are not enough to call it finished — an interrupted "
      "download can leave a truncated model.safetensors",
      amd.is_verified_download(target) is False)

amd._write_stamp(target, amd.REPO_ID, amd.REVISION, {"ok": True}, 123)
check("a matching stamp marks it verified", amd.is_verified_download(target) is True)
check("the stamp records that full verification was used",
      amd.read_stamp(target).get("full_verification") is True)

check("a stamp from an unrelated revision does not count",
      amd.is_verified_download(target, revision="0" * 40) is False)

# A machine that finished a 2.1a.2 single-file download must NOT be made to
# re-fetch 5.75 GiB just because the app now pins the chunked commit. The two
# commits carry the same weights, and the stamp check has to know that.
amd._write_stamp(target, amd.REPO_ID, amd.REVISION, {"ok": True}, 123)
check("a model verified under the single-file pin is still accepted once the "
      "app moves to the chunked pin (same bytes, no 3-hour re-download)",
      amd.is_verified_download(target, revision=amd.CHUNKED_REVISION) is True)
amd._write_stamp(target, amd.REPO_ID, amd.CHUNKED_REVISION, {"ok": True}, 123)
check("...and the other way round", amd.is_verified_download(target) is True)


def explode(*a, **kw):
    raise AssertionError("the network must not be touched when a verified copy exists")


amd._snapshot = explode
try:
    done = []
    result = amd.ensure_accurate_model(progress_callback=done.append)
    check("a verified model short-circuits without any network access",
          result == str(target), result)
    check("...and still reports 'done' so a UI can move straight on",
          done and done[-1].phase == "done" and done[-1].fraction == 1.0)
finally:
    amd._snapshot = real_snapshot


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 13. The pinned revision is a full commit sha, not a branch ===")
# ════════════════════════════════════════════════════════════════════════════
check("REVISION is pinned to a 40-character commit sha",
      len(amd.REVISION) == 40 and amd.REVISION.isalnum(), amd.REVISION)
check("REVISION is not a branch name",
      amd.REVISION not in ("main", "master", "HEAD"))
check("REVISION matches the fingerprint's model (fedc252…)",
      amd.REVISION.startswith("fedc252"), amd.REVISION_SHORT)


# ════════════════════════════════════════════════════════════════════════════
print("\n=== 14. Pre-flight information for the consent dialog (2.3) ===")
# ════════════════════════════════════════════════════════════════════════════
info = amd.describe_download(layout="single", revision=amd.REVISION)
check("describe_download() resolves the pinned revision",
      info["revision"] == amd.REVISION, info["revision"])
check("...and reports the real size the user is being asked to accept",
      5.0 < info["total_bytes"] / 2 ** 30 < 6.5,
      f"{info['total_bytes'] / 2 ** 30:.2f} GiB")
check("...and the file count", info["file_count"] == 13, str(info["file_count"]))


# ════════════════════════════════════════════════════════════════════════════
#  CHUNKED LAYOUT (masterplan 2.1a.3)
# ════════════════════════════════════════════════════════════════════════════
# The mechanics below are exercised against a local stand-in for the Hub: a
# directory of real chunk files and a real manifest, reached through a stubbed
# `_hf_file`. That seam is deliberate. What huggingface_hub does over the wire
# — resume, the Xet fallback, error classification — is already covered for real
# against the network in sections 3-8 above and does not change because the
# files got smaller. What is new in 2.1a.3, and what these sections test, is
# everything AROUND the transfer: which chunks get skipped, what happens to a
# bad one, whether the bytes reassemble exactly, and what survives a failure.
#
# The real chunked download against the real repo is a separate end-to-end run;
# see the delivery notes.

print("\n=== 15. Manifest validation ===")

FAKE_HUB = os.path.join(TEST_ROOT, "fake_hub")
os.makedirs(FAKE_HUB, exist_ok=True)

# A small stand-in "model" split into uneven chunks, so an off-by-one in the
# ordering or the last-chunk size cannot pass unnoticed.
WHOLE = bytes((i * 7 + 13) % 251 for i in range(50_000)) * 3
CHUNK_SIZES = [40_000, 40_000, 40_000, 30_000]
assert sum(CHUNK_SIZES) == len(WHOLE)


def build_manifest(hub_dir, whole=WHOLE, sizes=CHUNK_SIZES, name="model.safetensors"):
    os.makedirs(os.path.join(hub_dir, amd.REPO_CHUNK_DIR), exist_ok=True)
    entries = []
    offset = 0
    for index, size in enumerate(sizes):
        data = whole[offset:offset + size]
        offset += size
        chunk = f"{name}.part-{index:05d}"
        with open(os.path.join(hub_dir, amd.REPO_CHUNK_DIR, chunk), "wb") as fh:
            fh.write(data)
        entries.append({"name": chunk, "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {
        "schema": amd.MANIFEST_SCHEMA,
        "original_filename": name,
        "total_bytes": len(whole),
        "sha256": hashlib.sha256(whole).hexdigest(),
        "chunk_size_bytes": sizes[0],
        "chunk_count": len(entries),
        "chunks": entries,
    }
    with open(os.path.join(hub_dir, amd.REPO_CHUNK_DIR, amd.MANIFEST_NAME), "w") as fh:
        json.dump(manifest, fh)
    return manifest


MANIFEST = build_manifest(FAKE_HUB)

real_hf_file = amd._hf_file
fetched = []


def fake_hf_file(repo_id, revision, filename, target_dir, tqdm_class=None):
    """Serve a file out of FAKE_HUB, the way hf_hub_download would place it."""
    fetched.append(filename)
    src = os.path.join(FAKE_HUB, filename)
    dst = os.path.join(str(target_dir), filename)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    if tqdm_class is not None:                # drive progress like the real one
        bar = tqdm_class(total=os.path.getsize(src), unit="B",
                         name="huggingface_hub.http_get")
        bar.update(os.path.getsize(src))
    return dst


amd._hf_file = fake_hf_file
try:
    scratch = os.path.join(TEST_ROOT, "manifest_scratch")
    got = amd.fetch_manifest("x/y", "rev", cache_dir=scratch)
    check("fetch_manifest() reads the manifest at the pinned revision",
          got["sha256"] == MANIFEST["sha256"] and len(got["chunks"]) == 4)

    bad_hub = os.path.join(TEST_ROOT, "bad_schema_hub")
    m = build_manifest(bad_hub)
    m["schema"] = 99
    with open(os.path.join(bad_hub, amd.REPO_CHUNK_DIR, amd.MANIFEST_NAME), "w") as fh:
        json.dump(m, fh)
    saved_hub = FAKE_HUB
    FAKE_HUB = bad_hub
    try:
        amd.fetch_manifest("x/y", "rev", cache_dir=os.path.join(TEST_ROOT, "s2"))
        check("an unreadable manifest schema is refused", False, "no exception")
    except amd.AccurateModelDownloadError as e:
        check("an unreadable manifest schema is refused", True)
        check("...telling the user to update rather than to retry forever",
              e.can_retry is False and "update" in str(e).lower(), str(e))
    finally:
        FAKE_HUB = saved_hub
finally:
    amd._hf_file = real_hf_file


print("\n=== 16. Chunk verification and skip logic ===")
chunk_stage = os.path.join(TEST_ROOT, "stage", amd.REPO_CHUNK_DIR)
os.makedirs(chunk_stage, exist_ok=True)
entry0 = MANIFEST["chunks"][0]
src0 = os.path.join(FAKE_HUB, amd.REPO_CHUNK_DIR, entry0["name"])
good = os.path.join(chunk_stage, entry0["name"])
shutil.copyfile(src0, good)

check("a correct chunk verifies", amd._chunk_ok(good, entry0) is True)
check("a missing chunk does not", amd._chunk_ok(good + ".nope", entry0) is False)

with open(good, "r+b") as fh:            # truncate
    fh.truncate(entry0["bytes"] - 100)
check("a truncated chunk is rejected on size", amd._chunk_ok(good, entry0) is False)

shutil.copyfile(src0, good)
with open(good, "r+b") as fh:            # right size, one wrong byte
    fh.seek(entry0["bytes"] // 2)
    fh.write(b"\xff")
check("a chunk with the right size but wrong bytes is rejected on sha256 "
      "(size alone would have accepted it)",
      amd._chunk_ok(good, entry0) is False)

# _existing_chunks: keeps good, discards bad, ignores absent.
shutil.rmtree(chunk_stage, ignore_errors=True)
os.makedirs(chunk_stage, exist_ok=True)
shutil.copyfile(src0, os.path.join(chunk_stage, entry0["name"]))           # good
entry1 = MANIFEST["chunks"][1]
with open(os.path.join(chunk_stage, entry1["name"]), "wb") as fh:          # bad
    fh.write(b"\0" * entry1["bytes"])
present = amd._existing_chunks(chunk_stage, MANIFEST)
check("already-downloaded good chunks are detected and will be skipped",
      present == {entry0["name"]}, str(present))
check("a corrupt leftover is deleted rather than trusted",
      not os.path.exists(os.path.join(chunk_stage, entry1["name"])))


print("\n=== 17. Only missing chunks are fetched ===")
stage = os.path.join(TEST_ROOT, "stage")
amd._hf_file = fake_hf_file
fetched.clear()
tracker = amd._ByteTracker(MANIFEST["total_bytes"], baseline=entry0["bytes"])
try:
    got = amd._fetch_chunks("x/y", "rev", MANIFEST, stage, chunk_stage,
                            {entry0["name"]}, None, tracker)
finally:
    amd._hf_file = real_hf_file

check("every chunk is present afterwards", len(got) == 4, str(len(got)))
check("the chunk already on disk was NOT re-fetched",
      all(entry0["name"] not in f for f in fetched), str(fetched))
check("exactly the three missing chunks were fetched", len(fetched) == 3, str(fetched))
check("progress accounts for all fetched chunks, not just the last one "
      "(the bar must not reset once per chunk)",
      tracker.downloaded() == MANIFEST["total_bytes"],
      f"{tracker.downloaded()} of {MANIFEST['total_bytes']}")


print("\n=== 18. Reassembly is byte-exact, and failure keeps the chunks ===")
dest = os.path.join(stage, "model.safetensors")
amd._reassemble(chunk_stage, MANIFEST, dest)
with open(dest, "rb") as fh:
    rebuilt = fh.read()
check("the reassembled file is byte-identical to the original",
      rebuilt == WHOLE, f"{len(rebuilt)} bytes")
check("...and matches the manifest sha256",
      hashlib.sha256(rebuilt).hexdigest() == MANIFEST["sha256"])

# Corrupt one chunk on disk AFTER verification and reassemble again: the
# whole-file hash must catch what the per-chunk check no longer can.
with open(os.path.join(chunk_stage, MANIFEST["chunks"][2]["name"]), "r+b") as fh:
    fh.seek(10)
    fh.write(b"\xde\xad\xbe\xef")
os.remove(dest)
try:
    amd._reassemble(chunk_stage, MANIFEST, dest)
    check("a corrupted chunk is caught when the whole file is hashed", False, "no error")
except amd.AccurateModelDownloadError as e:
    check("a corrupted chunk is caught when the whole file is hashed", True)
    check("...the half-written output is not left behind",
          not os.path.exists(dest) and not os.path.exists(dest + ".reassembling"))
    check("...but the chunks are kept — they are the cheap way back",
          len(os.listdir(chunk_stage)) == 4, str(os.listdir(chunk_stage)))
    check("...and the user is told to retry without jargon",
          "try again" in str(e).lower() and "sha256" not in str(e).lower(), str(e))


print("\n=== 19. Disk space is checked before, not during ===")
try:
    amd._check_disk_space(TEST_ROOT, 10 ** 15)      # a petabyte
    check("an impossible download is refused up front", False, "no error")
except amd.AccurateModelDownloadError as e:
    check("an impossible download is refused up front", True)
    check("...with free-space wording and no retry suggestion",
          "space" in str(e).lower() and e.can_retry is False, str(e))
check("a download that fits is allowed", amd._check_disk_space(TEST_ROOT, 1024) is None)


print("\n=== 20. Chunks survive a guard failure, and only then are discarded ===")
reset_managed()
target = amd.managed_model_dir()
os.makedirs(target, exist_ok=True)
plant_corrupt_model(target)                       # fails the real guard
guard_chunks = os.path.join(target, amd.REPO_CHUNK_DIR)
os.makedirs(guard_chunks, exist_ok=True)
for entry in MANIFEST["chunks"]:
    shutil.copyfile(os.path.join(FAKE_HUB, amd.REPO_CHUNK_DIR, entry["name"]),
                    os.path.join(guard_chunks, entry["name"]))

amd.verify_merged_model = verify_with_test_fingerprint
real_download_chunked = amd._download_chunked
amd._download_chunked = lambda repo, rev, tgt, cb, mw: (
    str(tgt / amd.ORIGINAL_WEIGHTS_FILENAME), "https")
try:
    amd.ensure_accurate_model(repo_id=TINY_REPO, revision="main", layout="chunked")
    check("a model that fails the guard is rejected", False, "no exception")
except amd.AccurateModelDownloadError as e:
    check("a model that fails the guard is rejected", True)
    check("the bad reassembled model is deleted",
          not os.path.exists(os.path.join(target, "model.safetensors")))
    check("the downloaded chunks are KEPT — re-fetching 5.75 GiB to fix an "
          "assembly problem would be the wrong reflex",
          os.path.isdir(guard_chunks) and len(os.listdir(guard_chunks)) == 4,
          str(os.listdir(guard_chunks)) if os.path.isdir(guard_chunks) else "gone")
    check("...and the message says the parts were kept",
          "parts have been kept" in str(e).lower() or "quicker" in str(e).lower(), str(e))
    check("no stamp was written", amd.read_stamp(target) is None)
finally:
    amd.verify_merged_model = real_verify
    amd._download_chunked = real_download_chunked

# The success path is the only place chunks are removed.
reclaimed = amd._discard_chunks(target)
check("_discard_chunks() reclaims the staging directory",
      reclaimed == MANIFEST["total_bytes"] and not os.path.exists(guard_chunks),
      f"{reclaimed} bytes")
check("discarding chunks twice is harmless", amd._discard_chunks(target) == 0)


print("\n=== 21. ensure_accurate_model() returns a DIRECTORY, both layouts ===")
# Wave 2.3 does `transcribe(audio, model_dir=ensure_accurate_model())`, and the
# guard and from_pretrained() both take a directory. The chunked path originally
# returned the reassembled FILE path — ".../accurate-af-large-v3/model.safetensors"
# — which the unit tests missed because the stub echoed the same wrong shape, and
# only the real end-to-end download exposed. Assert the contract itself, against
# the real functions, so a stub can never define it again.
for layout in ("chunked", "single"):
    reset_managed()
    target = amd.managed_model_dir()
    os.makedirs(target, exist_ok=True)
    for name in amd.REQUIRED_MODEL_FILES:
        open(os.path.join(target, name), "w").write("{}")
    # Pretend the transfer already happened; the point under test is the shape
    # of what comes back, not the transfer.
    amd._snapshot = lambda repo_id, revision, tgt, tqdm_class, mw: str(tgt)
    saved_reassemble, saved_fetch_manifest = amd._reassemble, amd.fetch_manifest
    saved_existing, saved_fetch_chunks = amd._existing_chunks, amd._fetch_chunks
    amd.fetch_manifest = lambda repo, rev, cache_dir=None: dict(
        MANIFEST, original_filename="model.safetensors", total_bytes=4)
    amd._existing_chunks = lambda cd, m, cb=None: {e["name"] for e in m["chunks"]}
    amd._fetch_chunks = lambda *a, **kw: None
    amd._reassemble = lambda cd, m, dest, cb=None: dest
    amd.verify_merged_model = spy_verify
    try:
        result = amd.ensure_accurate_model(repo_id=TINY_REPO, revision="main",
                                           layout=layout)
        check(f"[{layout}] returns a directory, not a file",
              os.path.isdir(result), result)
        check(f"[{layout}] ...and it is the model directory the engine expects",
              os.path.isfile(os.path.join(result, "model.safetensors")), result)
        check(f"[{layout}] ...which is exactly what the guard was handed",
              os.path.normpath(result) == os.path.normpath(str(target)), result)
    finally:
        amd._snapshot = real_snapshot
        amd._reassemble, amd.fetch_manifest = saved_reassemble, saved_fetch_manifest
        amd._existing_chunks, amd._fetch_chunks = saved_existing, saved_fetch_chunks
        amd.verify_merged_model = real_verify


print("\n=== 22. The two pinned revisions mean different things ===")
check("the single-file pin is still the 2.1a.2 commit",
      amd.REVISION.startswith("fedc252"), amd.REVISION)
check("the chunked pin is a separate constant",
      amd.CHUNKED_REVISION != amd.REVISION)
check("the chunked pin is a real 40-character commit sha, not a placeholder "
      "or a branch name",
      len(amd.CHUNKED_REVISION) == 40 and amd.CHUNKED_REVISION.isalnum()
      and amd.CHUNKED_REVISION not in ("main", "master", "HEAD"),
      amd.CHUNKED_REVISION)
check("chunked is the default layout", amd.DEFAULT_LAYOUT == "chunked")
try:
    amd.ensure_accurate_model(layout="nonsense")
    check("an unknown layout is refused", False, "no exception")
except ValueError:
    check("an unknown layout is refused", True)


# ════════════════════════════════════════════════════════════════════════════
shutil.rmtree(TEST_ROOT, ignore_errors=True)
print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
