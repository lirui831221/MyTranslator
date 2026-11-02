#!/usr/bin/env python3
"""Prepare a deterministic 100 Chinese + 100 English FLEURS benchmark.

The script uses only Python's standard library.  It downloads the official
test metadata/archives when requested, extracts just one Chinese and one
English recording for 100 shared sentence IDs, and writes a JSONL suite with
parallel translation references.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import tarfile
import wave
from collections import defaultdict
from pathlib import Path
from typing import Iterable


BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
LANGUAGES = {
    "zh": {
        "config": "cmn_hans_cn",
        "direction": "zh2en",
        "prefix": "zh-real",
        "archive_size": 525_346_466,
        "archive_sha256": "09d19ad18f5d7e91076880807e866cd16abd924c7052b55f71cdae91714fc166",
    },
    "en": {
        "config": "en_us",
        "direction": "en2zh",
        "prefix": "en-real",
        "archive_size": 289_851_356,
        "archive_sha256": "d9c2e37b41aacd41bc283554a0a82b5476b36887049774ecb2819dcaaa55a356",
    },
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="shared sentence IDs per language")
    parser.add_argument("--cache-dir", type=Path, default=root / "benchmarks/cache/fleurs")
    parser.add_argument("--audio-dir", type=Path, default=root / "benchmarks/audio/fleurs200")
    parser.add_argument("--suite", type=Path, default=root / "benchmarks/suites/fleurs200.jsonl")
    parser.add_argument("--download", action="store_true", help="download missing official files with curl")
    parser.add_argument("--remove-archives", action="store_true", help="delete large archives after extraction")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be >= 1")
    return args


def file_paths(cache_dir: Path, language: str) -> tuple[Path, Path]:
    config = LANGUAGES[language]["config"]
    return cache_dir / f"{config}-test.tsv", cache_dir / f"{config}-test.tar.gz"


def curl(url: str, destination: Path, byte_range: tuple[int, int] | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl", "--http1.1", "-L", "--fail", "--retry", "10", "--retry-all-errors",
        "--retry-delay", "2", "--silent", "--show-error",
    ]
    if byte_range is not None:
        command.extend(["--range", f"{byte_range[0]}-{byte_range[1]}"])
    command.extend([url, "-o", str(destination)])
    subprocess.run(command, check=True)


def download_archive(url: str, destination: Path, expected_size: int, expected_sha256: str) -> None:
    if destination.exists() and destination.stat().st_size == expected_size:
        if sha256(destination) == expected_sha256:
            return
        destination.unlink()

    chunk_size = 16 * 1024 * 1024
    existing_size = destination.stat().st_size if destination.exists() else 0
    resume_at = min(existing_size // chunk_size * chunk_size, expected_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a+b") as handle:
        handle.truncate(resume_at)

    parts = destination.parent / f".{destination.name}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    ranges = [
        (start, min(start + chunk_size - 1, expected_size - 1))
        for start in range(resume_at, expected_size, chunk_size)
    ]

    def fetch(bounds: tuple[int, int]) -> tuple[int, Path]:
        start, end = bounds
        part = parts / f"{start:012d}-{end:012d}.part"
        expected = end - start + 1
        if not part.exists() or part.stat().st_size != expected:
            part.unlink(missing_ok=True)
            curl(url, part, bounds)
        if part.stat().st_size != expected:
            raise IOError(f"bad range size for {part}: {part.stat().st_size} != {expected}")
        return start, part

    print(
        f"download: {destination.name} ({resume_at / expected_size:.0%} complete, "
        f"{len(ranges)} ranges)",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        completed = list(pool.map(fetch, ranges))
    with destination.open("r+b") as output:
        for start, part in sorted(completed):
            output.seek(start)
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        output.truncate(expected_size)
    if sha256(destination) != expected_sha256:
        if resume_at > 0:
            # The new ranges are length-checked, so an invalid result after a
            # resumed download means the old prefix was corrupt. Re-fetch only
            # that prefix while retaining all verified-size range parts.
            destination.unlink()
            return download_archive(url, destination, expected_size, expected_sha256)
        raise IOError(f"SHA-256 mismatch after clean download: {destination}")
    shutil.rmtree(parts)


def ensure_inputs(cache_dir: Path, allow_download: bool) -> None:
    for language, info in LANGUAGES.items():
        tsv_path, archive_path = file_paths(cache_dir, language)
        config = info["config"]
        tsv_valid = False
        if tsv_path.exists():
            try:
                tsv_valid = bool(load_tsv(tsv_path))
            except (OSError, UnicodeError, ValueError):
                tsv_valid = False
        if not tsv_valid:
            if not allow_download:
                raise FileNotFoundError(f"missing or incomplete {tsv_path}; rerun with --download")
            print(f"download: {tsv_path.name}", flush=True)
            temporary = tsv_path.with_suffix(tsv_path.suffix + ".download")
            temporary.unlink(missing_ok=True)
            curl(f"{BASE_URL}/{config}/test.tsv", temporary)
            load_tsv(temporary)
            temporary.replace(tsv_path)
        archive_valid = archive_path.exists() and archive_path.stat().st_size == info["archive_size"]
        if not archive_valid and not allow_download:
            raise FileNotFoundError(f"missing or incomplete {archive_path}; rerun with --download")
        if allow_download:
            download_archive(
                f"{BASE_URL}/{config}/audio/test.tar.gz",
                archive_path,
                int(info["archive_size"]),
                str(info["archive_sha256"]),
            )


def load_tsv(path: Path) -> dict[int, list[dict[str, str]]]:
    rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    # FLEURS TSV is tab-delimited but raw sentences may contain unescaped ASCII
    # quotation marks, so CSV quote semantics would incorrectly merge rows.
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 7:
                raise ValueError(f"unexpected FLEURS row in {path}: {len(fields)} columns")
            sentence_id, filename, raw, normalized, _phones, samples, gender = fields
            rows[int(sentence_id)].append({
                "filename": filename,
                "raw": raw,
                "normalized": normalized,
                "samples": samples,
                "gender": gender.lower(),
            })
    return rows


def select_rows(
    zh_rows: dict[int, list[dict[str, str]]],
    en_rows: dict[int, list[dict[str, str]]],
    count: int,
) -> list[tuple[int, dict[str, str], dict[str, str]]]:
    shared = sorted(zh_rows.keys() & en_rows.keys())
    if len(shared) < count:
        raise ValueError(f"only {len(shared)} shared test IDs, requested {count}")
    # Alternate speaker gender where possible while retaining deterministic IDs.
    selected: list[tuple[int, dict[str, str], dict[str, str]]] = []
    for index, sentence_id in enumerate(shared[:count]):
        preferred = "female" if index % 2 == 0 else "male"
        choose = lambda rows: next((row for row in rows if row["gender"] == preferred), rows[0])
        selected.append((sentence_id, choose(zh_rows[sentence_id]), choose(en_rows[sentence_id])))
    return selected


def extract_selected(archive: Path, filenames: Iterable[str], destination: Path) -> None:
    wanted = set(filenames)
    found: set[str] = set()
    destination.mkdir(parents=True, exist_ok=True)
    print(f"extract: {archive.name} ({len(wanted)} files)", flush=True)
    with tarfile.open(archive, mode="r|gz") as bundle:
        for member in bundle:
            basename = Path(member.name).name
            if not member.isfile() or basename not in wanted:
                continue
            source = bundle.extractfile(member)
            if source is None:
                continue
            with (destination / basename).open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            found.add(basename)
    missing = wanted - found
    if missing:
        raise FileNotFoundError(f"{archive}: missing {len(missing)} selected files")


def validate_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as audio:
        details = (audio.getframerate(), audio.getnchannels(), audio.getsampwidth())
    if details != (16000, 1, 2):
        raise ValueError(f"{path}: expected 16 kHz mono 16-bit WAV, got {details}")


def normalize_selected(
    source_dir: Path, destination: Path, filenames: Iterable[str]
) -> dict[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required to convert FLEURS float WAV to PCM16")
    destination.mkdir(parents=True, exist_ok=True)
    source_hashes: dict[str, str] = {}
    for filename in filenames:
        source = source_dir / filename
        target = destination / filename
        temporary = target.with_suffix(".pcm.wav")
        source_hashes[filename] = sha256(source)
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(temporary),
            ],
            check=True,
        )
        temporary.replace(target)
        validate_wav(target)
    return source_hashes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_suite(
    root: Path,
    suite: Path,
    audio_dir: Path,
    selected: list[tuple[int, dict[str, str], dict[str, str]]],
    source_hashes: dict[tuple[str, str], str],
) -> None:
    suite.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, (sentence_id, zh_row, en_row) in enumerate(selected, 1):
        for language, source, reference in (("zh", zh_row, en_row), ("en", en_row, zh_row)):
            info = LANGUAGES[language]
            path = audio_dir / language / source["filename"]
            validate_wav(path)
            try:
                audio_value = str(path.relative_to(root))
            except ValueError:
                audio_value = str(path)
            case = {
                "id": f"{info['prefix']}-{index:03d}",
                "audio": audio_value,
                "source_lang": language,
                "direction": info["direction"],
                "categories": ["real", "fleurs", source["gender"]],
                "transcript": source["raw"],
                "translation_reference": reference["raw"],
                "target_terms": [],
                "dataset": {
                    "name": "google/fleurs",
                    "config": info["config"],
                    "split": "test",
                    "sentence_id": sentence_id,
                    "source_audio_sha256": source_hashes[(language, source["filename"])],
                    "benchmark_audio_sha256": sha256(path),
                },
            }
            lines.append(json.dumps(case, ensure_ascii=False, separators=(",", ":")))
    suite.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cache_dir = args.cache_dir.resolve()
    audio_dir = args.audio_dir.resolve()
    suite = args.suite.resolve()
    ensure_inputs(cache_dir, args.download)
    zh_tsv, zh_archive = file_paths(cache_dir, "zh")
    en_tsv, en_archive = file_paths(cache_dir, "en")
    zh_rows = load_tsv(zh_tsv)
    en_rows = load_tsv(en_tsv)
    selected = select_rows(zh_rows, en_rows, args.count)
    staging = audio_dir.parent / ".fleurs200-source"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        zh_filenames = [row[1]["filename"] for row in selected]
        en_filenames = [row[2]["filename"] for row in selected]
        extract_selected(zh_archive, zh_filenames, staging / "zh")
        extract_selected(en_archive, en_filenames, staging / "en")
        zh_hashes = normalize_selected(staging / "zh", audio_dir / "zh", zh_filenames)
        en_hashes = normalize_selected(staging / "en", audio_dir / "en", en_filenames)
        source_hashes = {
            **{("zh", name): value for name, value in zh_hashes.items()},
            **{("en", name): value for name, value in en_hashes.items()},
        }
        write_suite(root, suite, audio_dir, selected, source_hashes)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"suite: {suite}")
    print(f"audio: {audio_dir} ({args.count * 2} files)")
    if args.remove_archives:
        zh_archive.unlink()
        en_archive.unlink()
        print("removed downloaded archives; metadata and selected WAV files were retained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
