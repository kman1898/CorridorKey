#!/usr/bin/env python3
"""CorridorKey multi-machine work distributor.

Splits clips across multiple machines using a file-based job queue on a
shared volume. No message broker or network daemon required — atomic file
rename is the only mutual exclusion mechanism needed.

ARCHITECTURE
------------
One coordinator (any machine) writes job files into jobs/pending/.
Each worker machine runs this script in --worker mode, atomically claiming
jobs by renaming them from jobs/pending/ to jobs/running/<hostname>_<job>.json,
processing the clip range, then moving the result to jobs/done/ or jobs/failed/.

Two machines cannot both rename the same file, so there is no double-processing.

COORDINATOR USAGE
-----------------
Split each clip in ClipsForInference into N equal ranges and write job files:

    python distribute_work.py --coordinator --workers 4

Or split specific clips with a custom range size:

    python distribute_work.py --coordinator --workers 2 --clip shot_01

WORKER USAGE
------------
On each machine (all mounting the same shared volume at the same path):

    python distribute_work.py --worker

The worker loops until no pending jobs remain, then exits. Run it in a loop
via a shell script if you want it to wait for new jobs:

    while true; do python distribute_work.py --worker; sleep 10; done

SHARED VOLUME REQUIREMENTS
--------------------------
All machines must mount the shared volume at the same absolute path.
The ClipsForInference/ and Output/ directories must be on that volume.
10GbE or faster is strongly recommended when running 2+ workers writing 4K EXRs.

NETWORK PATH EXAMPLE
--------------------
NAS mounted at /mnt/studio on all machines:
    /mnt/studio/CorridorKey/ClipsForInference/
    /mnt/studio/CorridorKey/Output/
    /mnt/studio/CorridorKey/jobs/         <- created by this script
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Paths ---
BASE_DIR = Path(__file__).parent.resolve()
CLIPS_DIR = BASE_DIR / "ClipsForInference"
JOBS_DIR = BASE_DIR / "jobs"
PENDING_DIR = JOBS_DIR / "pending"
RUNNING_DIR = JOBS_DIR / "running"
DONE_DIR = JOBS_DIR / "done"
FAILED_DIR = JOBS_DIR / "failed"

CORRIDORKEY_CLI = BASE_DIR / "corridorkey_cli.py"
HOSTNAME = socket.gethostname()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ensure_job_dirs() -> None:
    for d in [PENDING_DIR, RUNNING_DIR, DONE_DIR, FAILED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _write_job(job: dict) -> Path:
    """Write a job dict as JSON into pending/. Returns the file path."""
    job_id = uuid.uuid4().hex[:8]
    clip = job["clip_name"].replace(" ", "_")
    fname = f"{clip}_{job_id}.json"
    path = PENDING_DIR / fname
    path.write_text(json.dumps(job, indent=2))
    return path


def _claim_job() -> tuple[Path, dict] | tuple[None, None]:
    """Atomically claim one pending job. Returns (running_path, job) or (None, None)."""
    candidates = sorted(PENDING_DIR.glob("*.json"))
    for src in candidates:
        dst = RUNNING_DIR / f"{HOSTNAME}_{src.name}"
        try:
            src.rename(dst)
            job = json.loads(dst.read_text())
            return dst, job
        except (FileNotFoundError, OSError):
            # Another worker claimed it first — try next
            continue
    return None, None


def _complete_job(running_path: Path) -> None:
    dst = DONE_DIR / running_path.name
    shutil.move(str(running_path), str(dst))


def _fail_job(running_path: Path, error: str) -> None:
    job = json.loads(running_path.read_text())
    job["error"] = error
    running_path.write_text(json.dumps(job, indent=2))
    dst = FAILED_DIR / running_path.name
    shutil.move(str(running_path), str(dst))


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

def _scan_ready_clips() -> list[str]:
    """Return clip names that have both Input and AlphaHint assets."""
    if not CLIPS_DIR.exists():
        return []
    ready = []
    for d in sorted(CLIPS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith((".","_")) or d.name == "IgnoredClips":
            continue
        has_input = (d / "Input").exists() or any(
            f.suffix.lower() in (".mp4", ".mov", ".mxf", ".avi")
            for f in d.iterdir()
            if f.stem.lower() == "input"
        )
        has_alpha = (d / "AlphaHint").exists()
        if has_input and has_alpha:
            ready.append(d.name)
    return ready


def _count_frames(clip_name: str) -> int:
    """Best-effort frame count from AlphaHint or Input sequence."""
    clip_dir = CLIPS_DIR / clip_name
    alpha_dir = clip_dir / "AlphaHint"
    if alpha_dir.exists():
        frames = [
            f for f in alpha_dir.iterdir()
            if f.suffix.lower() in (".exr", ".png", ".tif", ".tiff", ".jpg", ".jpeg")
        ]
        if frames:
            return len(frames)
    input_dir = clip_dir / "Input"
    if input_dir.exists():
        frames = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in (".exr", ".png", ".tif", ".tiff", ".jpg", ".jpeg")
        ]
        return len(frames)
    return 0


def run_coordinator(num_workers: int, clip_filter: str | None, device: str, backend: str) -> None:
    """Scan clips and write range-split job files into jobs/pending/."""
    _ensure_job_dirs()

    clips = _scan_ready_clips()
    if clip_filter:
        clips = [c for c in clips if clip_filter in c]

    if not clips:
        print("No ready clips found (need both Input and AlphaHint).")
        return

    print(f"Found {len(clips)} ready clip(s). Splitting across {num_workers} worker(s).\n")

    total_jobs = 0
    for clip_name in clips:
        num_frames = _count_frames(clip_name)
        if num_frames == 0:
            print(f"  SKIP {clip_name}: could not determine frame count")
            continue

        if num_workers == 1:
            # Single worker — process full clip, no splitting needed
            ranges = [(0, num_frames - 1)]
        else:
            chunk = max(1, num_frames // num_workers)
            ranges = []
            for w in range(num_workers):
                start = w * chunk
                end = (start + chunk - 1) if w < num_workers - 1 else num_frames - 1
                if start <= num_frames - 1:
                    ranges.append((start, end))

        for start, end in ranges:
            job = {
                "clip_name": clip_name,
                "frame_start": start,
                "frame_end": end,
                "device": device,
                "backend": backend,
            }
            path = _write_job(job)
            print(f"  {clip_name}: frames {start}-{end} -> {path.name}")
            total_jobs += 1

    print(f"\n{total_jobs} job file(s) written to {PENDING_DIR}")
    print("Start workers on each machine with:  python distribute_work.py --worker")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _build_command(job: dict) -> list[str]:
    cmd = [
        sys.executable, str(CORRIDORKEY_CLI),
        "--action", "run_inference",
        "--frame-start", str(job["frame_start"]),
        "--frame-end", str(job["frame_end"]),
    ]
    if job.get("device"):
        cmd += ["--device", job["device"]]
    return cmd


def run_worker(poll_interval: int) -> None:
    """Claim and process jobs until none remain."""
    _ensure_job_dirs()
    print(f"[{HOSTNAME}] Worker started. Polling {PENDING_DIR} ...")

    processed = 0
    while True:
        running_path, job = _claim_job()
        if running_path is None:
            if processed == 0:
                print(f"[{HOSTNAME}] No pending jobs found.")
            else:
                print(f"[{HOSTNAME}] No more pending jobs. Processed {processed} job(s).")
            break

        clip = job["clip_name"]
        start = job["frame_start"]
        end = job["frame_end"]
        print(f"[{HOSTNAME}] Claimed: {clip} frames {start}-{end}")

        cmd = _build_command(job)
        t0 = time.monotonic()
        try:
            result = subprocess.run(cmd, check=True, text=True)
            elapsed = time.monotonic() - t0
            _complete_job(running_path)
            processed += 1
            print(f"[{HOSTNAME}] Done: {clip} frames {start}-{end} in {elapsed:.1f}s")
        except subprocess.CalledProcessError as e:
            error = f"Exit code {e.returncode}"
            _fail_job(running_path, error)
            print(f"[{HOSTNAME}] FAILED: {clip} frames {start}-{end} — {error}")
            print(f"[{HOSTNAME}] Job moved to {FAILED_DIR}. Fix and re-queue manually if needed.")

        if poll_interval > 0:
            time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def run_status() -> None:
    """Print a summary of job queue state."""
    _ensure_job_dirs()
    def count_and_list(d: Path) -> list[str]:
        return sorted(f.name for f in d.glob("*.json"))

    pending = count_and_list(PENDING_DIR)
    running = count_and_list(RUNNING_DIR)
    done = count_and_list(DONE_DIR)
    failed = count_and_list(FAILED_DIR)

    print(f"Pending:  {len(pending)}")
    for f in pending:
        print(f"  {f}")
    print(f"Running:  {len(running)}")
    for f in running:
        print(f"  {f}")
    print(f"Done:     {len(done)}")
    print(f"Failed:   {len(failed)}")
    for f in failed:
        print(f"  {f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="CorridorKey multi-machine work distributor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--coordinator", action="store_true", help="Scan clips and write job files")
    mode.add_argument("--worker", action="store_true", help="Claim and process jobs")
    mode.add_argument("--status", action="store_true", help="Show queue state")

    parser.add_argument(
        "--workers", type=int, default=2,
        help="Number of workers to split each clip across (coordinator only, default: 2)",
    )
    parser.add_argument(
        "--clip", default=None,
        help="Process only clips whose name contains this string (coordinator only)",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
        help="Device to record in job files (default: auto)",
    )
    parser.add_argument(
        "--backend", choices=["auto", "torch", "mlx"], default="auto",
        help="Backend to record in job files (default: auto)",
    )
    parser.add_argument(
        "--poll", type=int, default=0,
        help="Seconds to pause between jobs (worker only, default: 0)",
    )

    args = parser.parse_args()

    if args.coordinator:
        run_coordinator(args.workers, args.clip, args.device, args.backend)
    elif args.worker:
        run_worker(args.poll)
    elif args.status:
        run_status()


if __name__ == "__main__":
    main()
