"""Split the merged Accurate-mode model into resumable chunks and publish them.

WHY THIS EXISTS (masterplan 2.1a.3)
───────────────────────────────────
Wave 2.1a.2 measured something unwelcome: huggingface_hub 1.x resumes downloads
per FILE, never per byte. It writes each file to a process-unique
`<etag>.<uuid>.incomplete` and unlinks it in a `finally`, so if the process dies
the bytes are gone. For this repo that is close to worst-case — `model.safetensors`
is 5.75 GiB, about 99.9% of the payload, in one file. A connection drop or an
app close two hours into a ~2.7 hour download meant starting again at zero.

Downgrading huggingface_hub is not available (transformers 5.14.1 requires
>=1.5.0, and per-byte resume was removed before then), and hand-rolling ranged
HTTP against Xet's chunk-reconstruction behaviour would mean writing our own
transfer path for a model whose entire value rests on being verifiable.

So instead of fighting for byte-level resume, this makes the files small. Split
the weights into ~200 MiB pieces and file-level resume — the mechanism that
already works, and the one that got this model uploaded in the first place after
Xet kept failing — becomes good enough. The worst an interruption can now cost is
one partial chunk.

WHAT IT PRODUCES
────────────────
A plain byte-level split. Deliberately NOT tensor-aware: a chunk is a slice of
the file, nothing more. Tensor-aware sharding would mean re-serialising the
weights, which would change the bytes, which would invalidate the pinned
fingerprint in accurate_fingerprint.py and every guarantee built on it. Byte
slices concatenate back to a file that is bit-identical by construction, and the
manifest records the sha256 to prove it.

    chunks/model.safetensors.manifest.json
    chunks/model.safetensors.part-00000        (200 MiB)
    ...
    chunks/model.safetensors.part-00029        (remainder)

The manifest records the chunks in order with each one's size and sha256, plus
the sha256 and size of the reassembled whole. The download side verifies each
chunk on arrival and the reassembled file at the end, so a bad byte has two
places to be caught before the guard ever runs.

USAGE
─────
Dry run — split locally, write the manifest, verify the round trip, upload
nothing:

    ~/whisper_afrikaans_spike/venv/bin/python3 shard_accurate_model.py \
        --out /tmp/chunks

Publish to the Hub (requires an authenticated `hf auth login` session; the token
is read from the huggingface_hub credential store, never passed on the command
line):

    ~/whisper_afrikaans_spike/venv/bin/python3 shard_accurate_model.py \
        --out /tmp/chunks --upload

ON NOT DELETING THE OLD FILE
────────────────────────────
This adds the chunked layout ALONGSIDE the existing single-file
`model.safetensors`. It never removes it. Until the chunked path has been proven
in a real installation, the monolithic file is the fallback that keeps 2.1a.2's
code working unchanged, and deleting it would make a rollback impossible.
Removing it is a separate, later commit — see the masterplan follow-up.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ID = "DanieClar/stillscript-whisper-large-v3-afrikaans"
SOURCE_DEFAULT = Path.home() / "whisper_afrikaans_spike" / "merged_afrikaans_fp32"
ORIGINAL_FILENAME = "model.safetensors"

# 200 MiB. Small enough that losing one to an interruption is a ~6 minute
# setback on a 0.6 MiB/s line rather than a 2.7 hour one; large enough that 30
# files do not turn into 30 rounds of per-request overhead.
DEFAULT_CHUNK_SIZE = 200 * 1024 * 1024

# Where the chunked layout lives in the repo. A subdirectory keeps the repo root
# recognisable as a normal HuggingFace model and makes "delete the monolith
# later" a single obvious operation.
REPO_CHUNK_DIR = "chunks"
MANIFEST_NAME = f"{ORIGINAL_FILENAME}.manifest.json"

# Upload pacing. Tuned for a saturated, asymmetric home uplink (~0.26 MiB/s
# measured), where the first attempt at this died after 1h15m because five
# parallel S3 multipart streams kept getting 'Server disconnected without
# sending a response'. Small batches bank progress often; few threads stop the
# streams from starving each other.
UPLOAD_BATCH_SIZE = 3
UPLOAD_THREADS = 2
UPLOAD_BATCH_ATTEMPTS = 4

# Chunks are raw byte slices, not loadable safetensors files. The `.part-NNNNN`
# suffix says so, and keeps them from being picked up by anything scanning for
# `*.safetensors` — including HuggingFace's own model-page parser, which would
# otherwise try to read a header that is not there.
def chunk_name(index):
    return f"{ORIGINAL_FILENAME}.part-{index:05d}"


def sha256_of(path, buf=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(buf)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def split(source, out_dir, chunk_size):
    """Write the chunks and return the manifest dict."""
    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = source.stat().st_size
    expected_chunks = (total_bytes + chunk_size - 1) // chunk_size
    print(f"Source : {source}")
    print(f"Size   : {total_bytes:,} bytes ({total_bytes / 2 ** 30:.2f} GiB)")
    print(f"Chunks : {expected_chunks} x {chunk_size / 2 ** 20:.0f} MiB")

    whole = hashlib.sha256()
    chunks = []
    started = time.time()

    with open(source, "rb") as src:
        index = 0
        while True:
            data = src.read(chunk_size)
            if not data:
                break
            whole.update(data)
            name = chunk_name(index)
            path = out_dir / name
            with open(path, "wb") as fh:
                fh.write(data)
            digest = hashlib.sha256(data).hexdigest()
            chunks.append({"name": name, "bytes": len(data), "sha256": digest})
            print(f"  [{index + 1:>2}/{expected_chunks}] {name}  "
                  f"{len(data):>12,} bytes  {digest[:16]}…")
            index += 1

    manifest = {
        "schema": 1,
        "original_filename": ORIGINAL_FILENAME,
        "total_bytes": total_bytes,
        "sha256": whole.hexdigest(),
        "chunk_size_bytes": chunk_size,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": (
            "Byte-level split of the merged Whisper large-v3 + Afrikaans model. "
            "Concatenate the chunks in listed order to reproduce "
            f"{ORIGINAL_FILENAME} exactly; the sha256 above is of the whole "
            "reassembled file. Chunks are raw slices and are not individually "
            "loadable."
        ),
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    print(f"\nWhole-file sha256: {manifest['sha256']}")
    print(f"Split in {time.time() - started:.1f}s -> {out_dir}")
    return manifest


def verify_roundtrip(out_dir, manifest):
    """Re-read the chunks from disk and prove they reassemble to the original.

    Hashing the source as we wrote it only proves we read it correctly. This
    reads back what actually landed on disk, in the order the manifest lists,
    which is the operation the download side will perform. If this does not
    match, nothing gets uploaded.
    """
    out_dir = Path(out_dir)
    print("\nVerifying round trip from the chunks on disk…")
    h = hashlib.sha256()
    total = 0
    for entry in manifest["chunks"]:
        path = out_dir / entry["name"]
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise SystemExit(f"FAIL {entry['name']}: {size} bytes, manifest says {entry['bytes']}")
        digest = sha256_of(path)
        if digest != entry["sha256"]:
            raise SystemExit(f"FAIL {entry['name']}: sha256 mismatch")
        with open(path, "rb") as fh:
            while True:
                block = fh.read(8 * 1024 * 1024)
                if not block:
                    break
                h.update(block)
        total += size

    if total != manifest["total_bytes"]:
        raise SystemExit(f"FAIL total {total} != {manifest['total_bytes']}")
    if h.hexdigest() != manifest["sha256"]:
        raise SystemExit(f"FAIL reassembled sha256 {h.hexdigest()} != {manifest['sha256']}")
    print(f"OK — {len(manifest['chunks'])} chunks, {total:,} bytes, "
          f"sha256 {h.hexdigest()} matches.")


def upload(out_dir, repo_id, message, disable_xet=False):
    """Publish the chunk directory as one new commit. Returns the commit sha.

    XET IS LEFT ENABLED BY DEFAULT HERE, which is the opposite of what the
    download side does, and for a specific reason worth writing down.

    These chunks are byte slices of a file that is ALREADY stored in the Hub's
    Xet CAS. Xet is content-addressed with content-defined chunk boundaries, so
    the interior of every slice hashes to chunks the server already has —
    boundaries are chosen by content, not by offset, so cutting the file at
    200 MiB marks does not shift them. Uploading through Xet therefore
    deduplicates against the existing model.safetensors and moves almost no
    bytes. Plain LFS cannot do this: it dedupes on whole-file oid, and a 200 MiB
    slice has an oid the server has never seen, so every byte goes up the wire.

    Measured on Danie's line: the plain-LFS path ran at ~0.2 MiB/s aggregate,
    i.e. ~7.6 hours for 5.75 GiB. That is the cost of not trying Xet first.

    If Xet fails the way it did when this model was first published
    (huggingface/xet-core #311/#407/#592), pass disable_xet=True and take the
    slow, reliable path.

    Nothing is deleted: `delete_patterns` is not passed, so the existing
    single-file model.safetensors stays exactly where it is.
    """
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    who = api.whoami()
    print(f"\nUploading to {repo_id} as {who['name']} "
          f"(Xet {'disabled' if disable_xet else 'enabled — expecting dedup'})…")

    files = sorted(p for p in Path(out_dir).iterdir() if p.is_file())
    total = sum(p.stat().st_size for p in files)
    additions = [
        CommitOperationAdd(path_in_repo=f"{REPO_CHUNK_DIR}/{p.name}",
                           path_or_fileobj=str(p))
        for p in files
    ]

    # Upload the bytes FIRST, in small batches, and only commit at the very end.
    #
    # `upload_folder()` does this in one shot, and on a slow uplink that is a
    # trap: the first attempt here ran 1h15m and then died on a single S3
    # `400 Bad Request` after five 'Server disconnected without sending a
    # response' retries, losing the whole commit. Splitting the work makes the
    # failure cheap instead of total:
    #
    #   * LFS objects are content-addressed and live in the store independently
    #     of any commit. Every batch that lands is banked permanently — a later
    #     run asks the LFS batch API which oids exist and is told to skip them.
    #     So a crash costs only the files that were mid-flight.
    #   * Small batches and few threads suit a saturated ~0.26 MiB/s uplink.
    #     Five parallel multipart streams sharing that link is what produced the
    #     disconnects; there is no throughput to win by running more of them.
    #   * A failed batch is retried on its own rather than restarting everything.
    #
    # The commit at the end is then almost instantaneous: every object it
    # references is already uploaded.
    started = time.time()
    batches = [additions[i:i + UPLOAD_BATCH_SIZE]
               for i in range(0, len(additions), UPLOAD_BATCH_SIZE)]

    for index, batch in enumerate(batches, start=1):
        names = ", ".join(op.path_in_repo.rsplit("/", 1)[-1] for op in batch)
        for attempt in range(1, UPLOAD_BATCH_ATTEMPTS + 1):
            try:
                print(f"\n[batch {index}/{len(batches)}] {names} "
                      f"(attempt {attempt}/{UPLOAD_BATCH_ATTEMPTS})", flush=True)
                api.preupload_lfs_files(repo_id=repo_id, additions=batch,
                                        num_threads=UPLOAD_THREADS)
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                if attempt == UPLOAD_BATCH_ATTEMPTS:
                    raise
                delay = 30 * attempt
                print(f"  batch failed ({type(exc).__name__}: {exc}); "
                      f"retrying in {delay}s — anything already uploaded is kept",
                      flush=True)
                time.sleep(delay)

    elapsed = time.time() - started
    print(f"\nAll {len(files)} files uploaded: {total / 2 ** 30:.2f} GiB in "
          f"{elapsed / 60:.1f} min ({total / elapsed / 2 ** 20:.2f} MiB/s). "
          f"Committing…")

    info = api.create_commit(
        repo_id=repo_id,
        operations=additions,
        commit_message=message,
        commit_description=(
            "Byte-level ~200 MiB split of model.safetensors, so an interrupted "
            "download resumes at chunk granularity instead of losing the whole "
            "5.75 GiB file (masterplan 2.1a.3).\n\n"
            "The original single-file model.safetensors is UNCHANGED and still "
            "present; this commit only adds the chunked layout alongside it."
        ),
        num_threads=UPLOAD_THREADS,
    )
    print(f"Commit: {info.oid}")
    print(f"URL   : {info.commit_url}")
    return info.oid


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=str(SOURCE_DEFAULT / ORIGINAL_FILENAME),
                    help="the merged model.safetensors to split")
    ap.add_argument("--out", required=True, help="local directory for the chunks")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--upload", action="store_true",
                    help="publish the chunks as a new commit (adds, never deletes)")
    ap.add_argument("--message", default="Add chunked model.safetensors for resumable download")
    ap.add_argument("--no-xet", action="store_true",
                    help="upload over plain LFS instead of Xet (slow, but immune to "
                         "the Xet connection failures — see upload() for the trade-off)")
    ap.add_argument("--skip-split", action="store_true",
                    help="chunks already exist in --out; verify and (optionally) upload only")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.skip_split:
        manifest = json.loads((out_dir / MANIFEST_NAME).read_text())
        print(f"Reusing existing chunks in {out_dir}")
    else:
        manifest = split(args.source, out_dir, args.chunk_size)

    verify_roundtrip(out_dir, manifest)

    if args.upload:
        commit = upload(out_dir, args.repo, args.message, disable_xet=args.no_xet)
        print("\n" + "=" * 70)
        print("Pin this revision in accurate_model_download.py:")
        print(f"  CHUNKED_REVISION = \"{commit}\"")
    else:
        print("\n(--upload not given; nothing was published)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
