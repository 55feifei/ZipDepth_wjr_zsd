"""Download the explicitly selected official TartanAir V1 archives safely.

No authentication or snapshot download is used. The published LFS SHA256 and
ZIP CRC protect downloads. Re-run the same command to resume .partial files.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import zipfile
import requests

HF = "https://huggingface.co"
REPO = "theairlabcmu/tartanair"
AIRLAB = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/tartanair"
KINDS = ("image_left.zip", "depth_left.zip", "flow_mask.zip", "flow_flow.zip")

class RangeReader(io.RawIOBase):
    """Seekable official remote archive. Cache adjacent reads to reduce round trips."""
    def __init__(self, item, ledger, limit):
        self.url = f"{HF}/datasets/{REPO}/resolve/main/{item['path']}"
        self.size = item["size"]
        self.position = 0
        self.cache_start, self.cache = 0, b""
        self.ledger, self.limit = ledger, limit
    def seekable(self):
        return True
    def readable(self):
        return True
    def tell(self):
        return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError("Negative range seek")
        return self.position
    def read(self, n=-1):
        n = min(self.size - self.position, n if n >= 0 else self.size)
        if n <= 0:
            return b""
        if not (self.cache_start <= self.position and self.position + n <= self.cache_start + len(self.cache)):
            start = self.position
            end = min(self.size - 1, start + max(n, 256 * 1024) - 1)
            if self.ledger["bytes"] + end - start + 1 > self.limit:
                raise RuntimeError("Bootstrap Range budget exhausted")
            response = requests.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=(30, 90))
            response.raise_for_status()
            if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                raise RuntimeError("Remote server did not return requested ZIP range")
            self.cache_start, self.cache = start, response.content
            self.ledger["bytes"] += len(self.cache)
        begin = self.position - self.cache_start
        result = self.cache[begin:begin + n]
        self.position += len(result)
        return result

def bootstrap(items, root, pairs_per_trajectory, budget):
    """Extract small genuine trajectories directly from complete remote ZIPs.

    A member's CRC is verified by ZipFile.read before atomically writing it.
    This is independent of incomplete local .partial files and is recorded as
    a bootstrap dataset rather than a full-dataset experiment.
    """
    root.mkdir(parents=True, exist_ok=True)
    provenance = root / "bootstrap_provenance.json"
    ledger = json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else {"bytes": 0, "source": "official remote ZIP HTTP Range", "bootstrap": True, "files": []}
    ledger["archives"] = items
    ledger["integrity"] = "ZIP CRC-32 verified per member; published whole-archive SHA256 recorded, not checked for Range subset"
    selected = list(items)
    # Flow archive directories define genuine frame pairs; process it first.
    selected.sort(key=lambda x: (x["path"].split("/")[0], 0 if x["path"].endswith("flow_flow.zip") else 1, x["path"]))
    wanted_frames, wanted_pairs = {}, {}
    ready_scenes = []
    for item in selected:
        with zipfile.ZipFile(RangeReader(item, ledger, budget)) as z:
            members = sorted(z.namelist())
            if item["path"].endswith("flow_flow.zip"):
                for name in members:
                    p = PurePosixPath(name)
                    if not p.name.endswith("_flow.npy"):
                        continue
                    trajectory = p.parent.parent.as_posix()
                    existing_scene = [t for t in wanted_pairs if t.split("/")[0] == trajectory.split("/")[0]]
                    if trajectory not in wanted_pairs and len(existing_scene) >= 3:
                        continue
                    frame, nxt, _ = p.stem.split("_")
                    pairs = wanted_pairs.setdefault(trajectory, [])
                    if len(pairs) < pairs_per_trajectory:
                        pairs.append(f"{frame}_{nxt}")
                        wanted_frames.setdefault(trajectory, set()).update((frame, nxt))
            for name in members:
                p = PurePosixPath(name)
                trajectory = p.parent.parent.as_posix()
                if trajectory not in wanted_pairs:
                    continue
                keep = (p.parent.name == "flow" and any(p.name.startswith(pair + "_") for pair in wanted_pairs[trajectory])) or (p.parent.name in ("image_left", "depth_left") and p.name[:6] in wanted_frames[trajectory])
                if not keep:
                    continue
                destination = root.joinpath(*p.parts)
                if not destination.resolve().is_relative_to(root.resolve()) or ".." in p.parts:
                    raise ValueError("Unsafe bootstrap member")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    continue
                content = z.read(name)  # zipfile validates this member's CRC.
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                temporary.write_bytes(content)
                os.replace(temporary, destination)
                ledger["files"].append(name)
                provenance.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
                print(f"BOOTSTRAP {name} ({ledger['bytes']} transferred bytes)", flush=True)
        if item["path"].endswith("image_left.zip"):
            from prepare_tartanair import build_manifest
            ready_scenes.append(item["path"].split("/")[0])
            manifest = build_manifest(root, scenes=ready_scenes)
            manifest["bootstrap"] = True
            manifest["purpose"] = "real-data pipeline validation; not the complete formal training set"
            manifest_path = root.parent / (root.name + "_manifest.json")
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"BOOTSTRAP_MANIFEST {manifest_path} scenes={ready_scenes}", flush=True)
    provenance.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"BOOTSTRAP_DONE {len(wanted_pairs)} trajectories, {ledger['bytes']} Range bytes", flush=True)

def selected_manifest(scenes):
    result = []
    for scene in scenes:
        r = requests.get(f"{HF}/api/datasets/{REPO}/tree/main/{scene}/Easy?expand=true", timeout=60)
        r.raise_for_status()
        items = {Path(x["path"]).name: x for x in r.json()}
        for kind in KINDS:
            x = items[kind]
            result.append({"path": x["path"], "size": int(x["size"]), "sha256": x["lfs"]["oid"]})
    return result

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_archive(path, item):
    if path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
        raise RuntimeError(f"Size/SHA256 mismatch: {path}")
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC mismatch: {bad}")

def safe_extract(path, root):
    """Check all members before extracting; reject traversal, drives and links."""
    root = Path(root).resolve()
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            p = PurePosixPath(info.filename.replace("\\", "/"))
            if p.is_absolute() or ".." in p.parts or any(":" in x for x in p.parts):
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"ZIP symbolic link rejected: {info.filename}")
            target = root.joinpath(*p.parts).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe ZIP destination: {target}")
        total = sum(i.file_size for i in z.infolist())
        if shutil.disk_usage(root).free < total:
            raise RuntimeError(f"Insufficient space for extraction: need {total} bytes")
        z.extractall(root)

def discard_covered_chunks(chunk_root, covered_bytes):
    """Delete range files already contained in the partial, then remove the directory.

    A finished byte range can leave a chunk behind when the partial already
    covers it. Removing that directory must not fail after the bytes are safe.
    """
    if not chunk_root.exists():
        return
    for child in list(chunk_root.iterdir()):
        if child.name == "ranges.json":
            continue
        stem = child.name.split(".", 1)[0]
        if not stem.isdigit():
            raise RuntimeError(f"Unexpected range file: {child}")
        if covered_bytes < int(stem) + child.stat().st_size:
            raise RuntimeError(f"Unmerged range file remains: {child}")
        child.unlink()
    metadata = chunk_root / "ranges.json"
    if metadata.exists():
        metadata.unlink()
    chunk_root.rmdir()


def parallel_ranges(item, partial, workers=8):
    """Persist independent 16 MiB chunks then append them in exact byte order."""
    url = f"{HF}/datasets/{REPO}/resolve/main/{item['path']}"
    head = requests.head(url, allow_redirects=True, timeout=(30, 90))
    head.raise_for_status()
    if int(head.headers.get("Content-Length", item["size"])) != item["size"]:
        raise RuntimeError("HEAD length differs from official LFS manifest")
    base = partial.stat().st_size if partial.exists() else 0
    if base > item["size"]:
        raise RuntimeError("Oversized partial")
    chunk_root = partial.with_suffix(partial.suffix + ".chunks")
    chunk_root.mkdir(exist_ok=True)
    metadata = chunk_root / "ranges.json"
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved["sha256"] != item["sha256"]:
            raise RuntimeError("Range metadata SHA changed")
        base = saved["base"]
    else:
        metadata.write_text(json.dumps({"base": base, "sha256": item["sha256"], "etag": head.headers.get("ETag")}), encoding="utf-8")
    ranges = [(start, min(item["size"] - 1, start + 16 * 1024 * 1024 - 1)) for start in range(base, item["size"], 16 * 1024 * 1024)]
    def fetch(bounds):
        start, end = bounds
        if partial.exists() and partial.stat().st_size >= end + 1:
            target = chunk_root / f"{start:012}.chunk"
            if target.exists():
                target.unlink()
            temporary = target.with_suffix(".part")
            if temporary.exists():
                temporary.unlink()
            return None
        target = chunk_root / f"{start:012}.chunk"
        expected = end - start + 1
        if target.exists() and target.stat().st_size == expected:
            return target
        temporary = target.with_suffix(".part")
        for attempt in range(6):
            existing = temporary.stat().st_size if temporary.exists() else 0
            try:
                with requests.get(url, headers={"Range": f"bytes={start + existing}-{end}"}, stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start + existing}-{end}/"):
                        raise RuntimeError("Range response mismatch")
                    with temporary.open("ab") as f:
                        for content in response.iter_content(1024 * 1024):
                            if existing + len(content) > expected:
                                raise RuntimeError("Range chunk oversized")
                            f.write(content)
                            existing += len(content)
                if existing != expected:
                    raise RuntimeError("Range chunk truncated")
                os.replace(temporary, target)
                print(f"RANGE {item['path']} {start}-{end}", flush=True)
                return target
            except (requests.RequestException, RuntimeError) as e:
                print(f"RANGE_RETRY {item['path']} {start} {attempt + 1}: {e}", flush=True)
                if attempt == 5:
                    raise
                time.sleep(min(10, 2 ** attempt))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(fetch, ranges):
            # map preserves range order and lets us reclaim disk immediately.
            if result is None:
                continue
            start = int(result.stem)
            offset = partial.stat().st_size if partial.exists() else 0
            if start <= offset < start + result.stat().st_size:
                with partial.open("ab") as destination, result.open("rb") as source:
                    source.seek(offset - start)
                    shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
            elif offset < start + result.stat().st_size:
                raise RuntimeError("Partial offset inconsistent with range ledger")
            result.unlink()
    discard_covered_chunks(chunk_root, partial.stat().st_size if partial.exists() else 0)

def download_one(item, archive_root, data_root, extract, attempts=6, range_workers=0):
    path = archive_root / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    marker = path.with_suffix(".verified.json")
    try:
        cached = json.loads(marker.read_text()) if marker.exists() else {}
    except (ValueError, OSError):
        cached = {}
    verified = path.exists() and cached.get("sha256") == item["sha256"] and path.stat().st_size == item["size"] and cached.get("mtime_ns") == path.stat().st_mtime_ns
    if not verified:
        if path.exists():
            validate_archive(path, item)
        else:
            urls = [f"{HF}/datasets/{REPO}/resolve/main/{item['path']}", f"{AIRLAB}/{item['path']}"]
            covered = partial.stat().st_size if partial.exists() else 0
            chunk_root = partial.with_suffix(partial.suffix + ".chunks")
            if covered == item["size"]:
                discard_covered_chunks(chunk_root, covered)
            elif range_workers > 1:
                try:
                    parallel_ranges(item, partial, range_workers)
                except (requests.RequestException, RuntimeError) as e:
                    print(f"RANGE_FALLBACK {item['path']}: {e}", flush=True)
            for attempt in range(attempts):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == item["size"]:
                    break
                if offset > item["size"]:
                    raise RuntimeError(f"Partial exceeds official size: {partial}")
                url = urls[0 if attempt < attempts // 2 else 1]
                try:
                    headers = {"Range": f"bytes={offset}-"} if offset else {}
                    with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as r:
                        r.raise_for_status()
                        if offset and r.status_code != 206:
                            raise RuntimeError("Server did not honor resume Range; retaining partial")
                        if r.status_code == 206 and not r.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise RuntimeError("Incorrect Content-Range")
                        length = r.headers.get("Content-Length")
                        if length and int(length) + offset != item["size"]:
                            raise RuntimeError("Content-Length differs from official manifest")
                        with partial.open("ab" if offset else "wb") as f:
                            last = time.monotonic()
                            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                                if not chunk:
                                    continue
                                if offset + len(chunk) > item["size"]:
                                    raise RuntimeError("Received bytes exceed manifest budget")
                                f.write(chunk)
                                offset += len(chunk)
                                if time.monotonic() - last >= 15:
                                    f.flush()
                                    print(f"PROGRESS {item['path']} {offset}/{item['size']}", flush=True)
                                    last = time.monotonic()
                    if offset != item["size"]:
                        raise RuntimeError("Truncated response")
                    break
                except (requests.RequestException, RuntimeError) as e:
                    print(f"RETRY {item['path']} {attempt + 1}/{attempts}: {e}", flush=True)
                    if attempt + 1 == attempts:
                        raise
                    time.sleep(min(10, 2 ** attempt))
            validate_archive(partial, item)
            os.replace(partial, path)
        marker.write_text(json.dumps({**item, "mtime_ns": path.stat().st_mtime_ns}, indent=2), encoding="utf-8")
    if extract and not path.with_suffix(".extracted").exists():
        safe_extract(path, data_root)
        path.with_suffix(".extracted").write_text("ZIP SHA256 and CRC verified before extraction\n", encoding="utf-8")
    print(f"DONE {item['path']}", flush=True)
    return item["path"]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", nargs="+", default=["carwelding", "japanesealley", "office"])
    p.add_argument("--archives", type=Path, default=Path("data/archives"))
    p.add_argument("--root", type=Path, default=Path("data/tartanair"))
    p.add_argument("--budget-bytes", type=int, default=50_000_000_000)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--range-workers", type=int, default=8, help="Parallel resumable 16 MiB ranges per archive; 0 uses single stream")
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--no-extract", action="store_true")
    p.add_argument("--bootstrap-pairs", type=int, default=0, help="Separate real small sample via remote ZIP Range, not full training data")
    p.add_argument("--bootstrap-root", type=Path, default=Path("data/tartanair_bootstrap"))
    args = p.parse_args()
    args.archives.mkdir(parents=True, exist_ok=True)
    args.root.mkdir(parents=True, exist_ok=True)
    items = selected_manifest(args.scenes)
    total = sum(x["size"] for x in items)
    if total > args.budget_bytes:
        raise RuntimeError(f"Official selected bytes {total} exceed budget {args.budget_bytes}")
    selected_paths = {x["path"] for x in items}
    unrelated = sum(x.stat().st_size for x in args.archives.rglob("*.zip*") if x.is_file() and str(x.relative_to(args.archives)).replace("\\", "/").removesuffix(".partial") not in selected_paths)
    if total + unrelated > args.budget_bytes:
        raise RuntimeError("Existing unrelated archives plus selection exceed download budget")
    manifest = {"source": f"{HF}/datasets/{REPO}", "budget_bytes": args.budget_bytes, "selected_bytes": total, "files": items}
    (args.archives / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"SELECTED {len(items)} files, {total} bytes / {args.budget_bytes} budget", flush=True)
    if args.manifest_only:
        return
    if args.bootstrap_pairs:
        bootstrap(items, args.bootstrap_root, args.bootstrap_pairs, min(200_000_000, args.budget_bytes - total))
        return
    remaining = sum(x["size"] - min(x["size"], (args.archives / x["path"]).stat().st_size if (args.archives / x["path"]).exists() else (args.archives / (x["path"] + ".partial")).stat().st_size if (args.archives / (x["path"] + ".partial")).exists() else 0) for x in items)
    if shutil.disk_usage(args.archives).free < remaining:
        raise RuntimeError("Insufficient space for remaining archives")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Complete a whole scene first so training can start while others download.
        futures = [executor.submit(download_one, x, args.archives, args.root, not args.no_extract, 6, args.range_workers) for x in items]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    if not args.no_extract:
        from prepare_tartanair import build_manifest
        manifest_path = args.root.parent / "tartanair_manifest.json"
        manifest = build_manifest(args.root, scenes=args.scenes)
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, manifest_path)
        print(f"FORMAL_MANIFEST {manifest_path}", flush=True)

if __name__ == "__main__":
    main()
