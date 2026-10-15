#!/usr/bin/env python3
"""
Resilient Hugging Face dataset downloader.

Works around the Xet-backend rate-limit crash:

    RuntimeError: Task error: File reconstruction error: CAS Client Error:
    Request error: HTTP status client error (429 Too Many Requests),
    domain: https://transfer.xethub-eu.hf.co/...

Strategy:
  1. Wrap `snapshot_download()` in an infinite retry loop with exponential
     backoff + jitter. snapshot_download() is resumable: files already fully
     downloaded in `--out` are skipped on each retry, so progress accumulates.
  2. Throttle concurrency on two levels:
       - HF's own thread pool  -> `--workers` (snapshot_download max_workers)
       - The Xet client's parallel range-GETs -> `--xet-concurrency`
  3. Make the Xet client itself retry transient errors -> `--xet-retry-attempts`.
  4. Optionally bypass Xet entirely and use the classic CDN HTTP path
     (`--disable-xet`), which honors Retry-After and handles 429s better.

Usage (on the GPU machine, inside tmux so it survives disconnects):

    tmux new -s hfdownload
    python scripts/download_hf_robust.py --out /data/weather_mtg
    # detach with Ctrl-b d ; reattach with: tmux attach -t hfdownload

Other useful invocations:

    # gentler on the rate limiter (slower but far fewer 429s)
    python scripts/download_hf_robust.py --workers 2 --xet-concurrency 4

    # last resort if Xet keeps failing
    python scripts/download_hf_robust.py --disable-xet

    # smoke test: grab only the README
    python scripts/download_hf_robust.py --out /tmp/smoke --include 'README.md' --max-retries 2
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

log = logging.getLogger("hf-dl")

DEFAULT_REPO = "meteolibre-dev/weather_mtg_europe_lightning_radar_128_0dot012_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO, help="repo id (default: %(default)s)")
    p.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    p.add_argument("--out", default="./data/weather_mtg", help="local directory to download into")
    p.add_argument("--workers", type=int, default=4,
                   help="snapshot_download max_workers, i.e. parallel files (default: %(default)s; HF default is 8)")
    p.add_argument("--max-retries", type=int, default=0,
                   help="give up after N failed attempts (0 = never give up)")
    p.add_argument("--backoff-base", type=float, default=60.0,
                   help="first wait after a failure, seconds (default: %(default)s)")
    p.add_argument("--backoff-cap", type=float, default=900.0,
                   help="max wait between retries, seconds (default: %(default)s)")
    p.add_argument("--disable-xet", action="store_true",
                   help="set HF_HUB_DISABLE_XET=1: classic HTTP CDN path instead of the Xet backend")
    p.add_argument("--xet-retry-attempts", type=int, default=20,
                   help="HF_XET_CLIENT_RETRY_MAX_ATTEMPTS (0 = leave library default)")
    p.add_argument("--xet-concurrency", type=int, default=0,
                   help="HF_XET_FIXED_DOWNLOAD_CONCURRENCY: parallel range-GETs inside the "
                        "Xet client (0 = leave library default; try 4 if you keep getting 429s)")
    p.add_argument("--token", default=None, help="HF token (default: $HF_TOKEN / $HF_TOKEN_PATH)")
    p.add_argument("--include", default=None,
                   help="comma-separated regex(es) of files to download (allow_regex)")
    p.add_argument("--exclude", default=None,
                   help="comma-separated regex(es) of files to skip (ignore_regex)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def apply_env(args: argparse.Namespace) -> None:
    """Must run BEFORE `import huggingface_hub` — the library reads env vars at import time."""
    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        log.info("HF_HUB_DISABLE_XET=1 -> using classic HTTP CDN downloads")
    if args.xet_retry_attempts > 0:
        os.environ["HF_XET_CLIENT_RETRY_MAX_ATTEMPTS"] = str(args.xet_retry_attempts)
        log.info("HF_XET_CLIENT_RETRY_MAX_ATTEMPTS=%d", args.xet_retry_attempts)
    if args.xet_concurrency > 0:
        os.environ["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] = str(args.xet_concurrency)
        log.info("HF_XET_FIXED_DOWNLOAD_CONCURRENCY=%d", args.xet_concurrency)
    # Per-request socket timeout so a stalled connection can't hang an attempt forever.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")


def dir_bytes(path: Path) -> int:
    """Total bytes on disk under `path`, including .incomplete temp files (progress detection)."""
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path, onerror=lambda e: None):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    apply_env(args)

    # Imported late on purpose: env vars above are read at import time.
    import inspect

    from huggingface_hub import snapshot_download

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # huggingface_hub 1.x uses allow_patterns/ignore_patterns,
    # older versions used allow_regex/ignore_regex.
    supported = set(inspect.signature(snapshot_download).parameters)
    allow_kw = "allow_patterns" if "allow_patterns" in supported else "allow_regex"
    ignore_kw = "ignore_patterns" if "ignore_patterns" in supported else "ignore_regex"

    kwargs: dict = dict(
        repo_id=args.repo,
        repo_type=args.repo_type,
        local_dir=str(out_dir),
        max_workers=args.workers,
    )
    if args.token:
        kwargs["token"] = args.token
    if args.include:
        kwargs[allow_kw] = [r.strip() for r in args.include.split(",") if r.strip()]
    if args.exclude:
        kwargs[ignore_kw] = [r.strip() for r in args.exclude.split(",") if r.strip()]

    log.info("repo      : %s (%s)", args.repo, args.repo_type)
    log.info("local dir : %s", out_dir)
    log.info("workers   : %d | xet retries: %d | xet concurrency: %s",
             args.workers, args.xet_retry_attempts, args.xet_concurrency or "default")

    attempt = 0
    backoff = args.backoff_base
    last_bytes = dir_bytes(out_dir)
    t0 = time.monotonic()

    while True:
        attempt += 1
        try:
            log.info("attempt #%d — downloading (already on disk: %s) ...",
                     attempt, fmt_bytes(last_bytes))
            path = snapshot_download(**kwargs)
            done_bytes = dir_bytes(out_dir)
            log.info("SUCCESS after %s — %s in %s — path: %s",
                     fmt_duration(time.monotonic() - t0),
                     fmt_bytes(done_bytes), out_dir, path)
            return 0
        except KeyboardInterrupt:
            log.warning("interrupted by user — already-downloaded files are kept; "
                        "rerun the same command to resume.")
            return 130
        except Exception as e:  # noqa: BLE001 — we intentionally retry on anything
            cur_bytes = dir_bytes(out_dir)
            gained = cur_bytes - last_bytes
            last_bytes = cur_bytes

            if args.max_retries > 0 and attempt >= args.max_retries:
                log.error("attempt #%d failed and --max-retries=%d reached. Last error: %s",
                          attempt, args.max_retries, e)
                return 1

            if gained > 0:
                # We are being served, just throttled: cool down briefly, don't escalate.
                log.info("attempt #%d still gained %s before failing — progress OK, "
                         "resetting backoff to %.0fs", attempt, fmt_bytes(gained), args.backoff_base)
                backoff = args.backoff_base
            else:
                # No progress at all: probably hard rate-limited, back off longer.
                backoff = min(backoff * 2, args.backoff_cap)

            wait = backoff + random.uniform(0, backoff * 0.25)  # jitter avoids thundering herd
            log.warning("attempt #%d failed: %s", attempt, str(e).splitlines()[0][:300])
            log.warning("gained this attempt: %s | total on disk: %s | retrying in %.0fs",
                        fmt_bytes(max(gained, 0)), fmt_bytes(cur_bytes), wait)
            time.sleep(wait)


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


if __name__ == "__main__":
    sys.exit(main())
