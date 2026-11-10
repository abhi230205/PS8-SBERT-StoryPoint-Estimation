"""Download the TAWOS MySQL dump (UCL RDR / figshare, DOI 10.5522/04/21308124).

Uses parallel HTTP range requests because single-stream downloads from the figshare
S3 bucket are often throttled.  Every chunk request goes through the figshare
redirect, which issues a fresh short-lived S3 URL.

Usage:  python scripts/download_tawos.py [--out data/raw/TAWOS.sql.zip] [--workers 16]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

ARTICLE_API = "https://api.figshare.com/v2/articles/21308124"
FILE_NAME = "TAWOS.sql.zip"
CHUNK = 8 * 1024 * 1024
UA = {"User-Agent": "tawos-downloader/1.0"}


def file_info():
    with urllib.request.urlopen(urllib.request.Request(ARTICLE_API, headers=UA), timeout=60) as r:
        meta = json.load(r)
    for f in meta["files"]:
        if f["name"] == FILE_NAME:
            return f["download_url"], int(f["size"]), f.get("computed_md5")
    raise SystemExit(f"{FILE_NAME} not found in figshare article")


def fetch(url: str, start: int, end: int, part: Path, retries: int = 8) -> int:
    want = end - start + 1
    if part.exists() and part.stat().st_size == want:
        return want
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != want:
                raise IOError(f"short read {len(data)} != {want}")
            part.write_bytes(data)
            return want
        except Exception as exc:  # noqa: BLE001
            time.sleep(2 + 2 * attempt)
            last = exc
    raise RuntimeError(f"chunk {start}-{end} failed: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/TAWOS.sql.zip")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    url, size, md5 = file_info()
    print(f"{FILE_NAME}: {size/1e6:.1f} MB (md5 {md5})")
    parts_dir = out.parent / (out.name + ".parts")
    parts_dir.mkdir(exist_ok=True)
    ranges = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, url, s, e, parts_dir / f"{i:05d}"): i for i, (s, e) in enumerate(ranges)}
        for fut in cf.as_completed(futs):
            done += fut.result()
            print(f"\r{done/1e6:8.1f} / {size/1e6:.1f} MB  ({done/1e6/(time.time()-t0):.1f} MB/s)", end="", flush=True)
    print()
    with open(out, "wb") as fh:
        for i in range(len(ranges)):
            fh.write((parts_dir / f"{i:05d}").read_bytes())
    for p in parts_dir.iterdir():
        p.unlink()
    parts_dir.rmdir()
    assert out.stat().st_size == size, "size mismatch"
    if md5:
        import hashlib

        h = hashlib.md5()
        with open(out, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""):
                h.update(blk)
        assert h.hexdigest() == md5, f"md5 mismatch {h.hexdigest()} != {md5}"
        print("md5 OK")
    with zipfile.ZipFile(out) as z:
        print("zip contents:", [(i.filename, i.file_size) for i in z.infolist()])


if __name__ == "__main__":
    main()
