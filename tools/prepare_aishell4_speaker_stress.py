#!/usr/bin/env python3
"""Extract a compact speaker-change stress suite from the AISHELL-4 test set.

The script scans official TextGrid annotations inside test.tar.gz, chooses a
chronological window with frequent speaker changes, and keeps only short mono
clips from one meeting. Prefer --download-selected to fetch the official
AISHELL Hugging Face mirror's annotations plus only the selected meeting.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DATASET_URL = "https://openslr.org/111/"
ARCHIVE_URL = "https://www.openslr.org/resources/111/test.tar.gz"
ARCHIVE_SIZE = 5_241_010_904
HF_TREE_URL = "https://huggingface.co/api/datasets/AISHELL/AISHELL-4/tree/main/test/TextGrid"
HF_WAV_TREE_URL = "https://huggingface.co/api/datasets/AISHELL/AISHELL-4/tree/main/test/wav"
HF_RESOLVE_URL = "https://huggingface.co/datasets/AISHELL/AISHELL-4/resolve/main/test"


@dataclass(frozen=True)
class Interval:
    session: str
    speaker: str
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive",
        type=Path,
        nargs="?",
        default=root / "benchmarks/cache/aishell4/test.tar.gz",
    )
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--min-duration", type=float, default=0.8)
    parser.add_argument("--max-duration", type=float, default=5.0)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--download", action="store_true", help="resume/download the official test archive")
    parser.add_argument(
        "--download-selected",
        action="store_true",
        help="download all small TextGrids and only the selected meeting FLAC from AISHELL's mirror",
    )
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument(
        "--remove-archive",
        action="store_true",
        help="remove the 5.2 GB source archive after a successful extraction",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=root / "benchmarks/audio/aishell4-speaker-stress",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=root / "benchmarks/suites/aishell4-speaker-stress.jsonl",
    )
    parser.add_argument(
        "--annotation-copy",
        type=Path,
        default=root / "benchmarks/manifests/aishell4-selected.TextGrid",
    )
    parser.add_argument(
        "--textgrid-cache",
        type=Path,
        default=root / "benchmarks/cache/aishell4/textgrids",
    )
    parser.add_argument(
        "--selected-source-cache",
        type=Path,
        default=root / "benchmarks/cache/aishell4/selected-source",
        help="persistent download/part cache for the one selected meeting",
    )
    args = parser.parse_args()
    if args.count < 4:
        parser.error("--count must be >= 4")
    if args.channel < 0:
        parser.error("--channel must be >= 0")
    if args.download_workers < 1 or args.download_workers > 16:
        parser.error("--download-workers must be in 1..16")
    if args.download and args.download_selected:
        parser.error("choose either --download or --download-selected")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def curl_range(url: str, destination: Path, start: int, end: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "10",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--silent",
            "--show-error",
            "--range",
            f"{start}-{end}",
            url,
            "-o",
            str(destination),
        ],
        check=True,
    )


def curl_file(url: str, destination: Path, resume: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "10",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--silent",
        "--show-error",
    ]
    if resume and destination.exists():
        command.extend(["-C", "-"])
    command.extend([url, "-o", str(destination)])
    subprocess.run(command, check=True)


def download_sized_file(
    url: str, destination: Path, size: int, workers: int, chunk_size: int = 16 * 1024 * 1024
) -> None:
    """Download a known-size file in resumable parallel ranges."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == size:
        return
    parts = destination.parent / f".{destination.name}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    ranges = [
        (start, min(start + chunk_size - 1, size - 1))
        for start in range(0, size, chunk_size)
    ]

    def fetch(bounds: tuple[int, int]) -> tuple[int, Path]:
        start, end = bounds
        part = parts / f"{start:012d}-{end:012d}.part"
        expected = end - start + 1
        if not part.exists() or part.stat().st_size != expected:
            part.unlink(missing_ok=True)
            curl_range(url, part, start, end)
        if part.stat().st_size != expected:
            raise IOError(f"bad range size for {part}: {part.stat().st_size} != {expected}")
        return start, part

    print(
        f"download: {destination.name} ({len(ranges)} resumable ranges, {workers} workers)",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        completed = list(pool.map(fetch, ranges))
    temporary = destination.with_suffix(destination.suffix + ".assembling")
    with temporary.open("wb") as output:
        for _, part in sorted(completed):
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if temporary.stat().st_size != size:
        raise IOError(f"incomplete assembled file: {temporary}")
    temporary.replace(destination)
    shutil.rmtree(parts)


def download_archive(destination: Path, workers: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == ARCHIVE_SIZE:
        return
    chunk_size = 64 * 1024 * 1024
    existing = destination.stat().st_size if destination.exists() else 0
    resume_at = min(existing // chunk_size * chunk_size, ARCHIVE_SIZE)
    with destination.open("a+b") as handle:
        handle.truncate(resume_at)
    parts = destination.parent / ".test.tar.gz.parts"
    parts.mkdir(parents=True, exist_ok=True)
    ranges = [
        (start, min(start + chunk_size - 1, ARCHIVE_SIZE - 1))
        for start in range(resume_at, ARCHIVE_SIZE, chunk_size)
    ]

    def fetch(bounds: tuple[int, int]) -> tuple[int, Path]:
        start, end = bounds
        part = parts / f"{start:012d}-{end:012d}.part"
        expected = end - start + 1
        if not part.exists() or part.stat().st_size != expected:
            part.unlink(missing_ok=True)
            curl_range(ARCHIVE_URL, part, start, end)
        if part.stat().st_size != expected:
            raise IOError(f"bad range size for {part}: {part.stat().st_size} != {expected}")
        return start, part

    print(
        f"download: AISHELL-4 test.tar.gz ({resume_at / ARCHIVE_SIZE:.0%} complete, "
        f"{len(ranges)} ranges, {workers} workers)",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        completed = list(pool.map(fetch, ranges))
    with destination.open("r+b") as output:
        for start, part in sorted(completed):
            output.seek(start)
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        output.truncate(ARCHIVE_SIZE)
    shutil.rmtree(parts)
    if destination.stat().st_size != ARCHIVE_SIZE:
        raise IOError(f"incomplete archive: {destination}")


def clean_text(value: str) -> str:
    value = value.strip().strip('"').strip()
    if value in {"", " ", "<%>", "<$>"}:
        return ""
    value = re.sub(r"<(?:sil|\$|%)>", "", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", "", value)


def parse_textgrid(raw: str, session: str) -> list[Interval]:
    speaker = "unknown"
    current: dict[str, float | str] | None = None
    intervals: list[Interval] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("name ="):
            speaker = clean_text(stripped.split("=", 1)[1]) or speaker
        elif re.match(r"intervals \[\d+\]:", stripped):
            if current and {"start", "end", "text"} <= current.keys():
                text = clean_text(str(current["text"]))
                if text:
                    intervals.append(
                        Interval(session, speaker, float(current["start"]), float(current["end"]), text)
                    )
            current = {}
        elif current is not None and stripped.startswith("xmin ="):
            current["start"] = float(stripped.split("=", 1)[1])
        elif current is not None and stripped.startswith("xmax ="):
            current["end"] = float(stripped.split("=", 1)[1])
        elif current is not None and stripped.startswith("text ="):
            current["text"] = stripped.split("=", 1)[1]
    if current and {"start", "end", "text"} <= current.keys():
        text = clean_text(str(current["text"]))
        if text:
            intervals.append(
                Interval(session, speaker, float(current["start"]), float(current["end"]), text)
            )
    return intervals


def usable(interval: Interval, minimum: float, maximum: float) -> bool:
    return minimum <= interval.duration <= maximum and len(interval.text) >= 3


def window_score(window: list[Interval]) -> tuple[int, int, float]:
    changes = sum(left.speaker != right.speaker for left, right in zip(window, window[1:]))
    short_gaps = sum(
        -0.1 <= right.start - left.end <= 0.65
        for left, right in zip(window, window[1:])
    )
    total_duration = sum(interval.duration for interval in window)
    return changes, short_gaps, -total_duration


def choose_window(
    sessions: dict[str, list[Interval]], count: int, minimum: float, maximum: float
) -> list[Interval]:
    best: list[Interval] = []
    best_score = (-1, -1, float("-inf"))
    for intervals in sessions.values():
        ordered = sorted(
            (interval for interval in intervals if usable(interval, minimum, maximum)),
            key=lambda interval: (interval.start, interval.end, interval.speaker),
        )
        # Keep one primary turn at a time. Heavily overlapping speech is useful
        # for a separate overlap suite but makes a deterministic change boundary
        # impossible to grade here.
        non_overlapping: list[Interval] = []
        for interval in ordered:
            if non_overlapping and interval.start < non_overlapping[-1].end - 0.08:
                continue
            non_overlapping.append(interval)
        if len(non_overlapping) < count:
            continue
        for start in range(len(non_overlapping) - count + 1):
            candidate = non_overlapping[start : start + count]
            score = window_score(candidate)
            if score > best_score:
                best, best_score = candidate, score
    if not best:
        raise ValueError(f"no AISHELL-4 session contained {count} usable chronological turns")
    return best


def find_member_by_stem(
    members: Iterable[tarfile.TarInfo], stem: str, suffixes: set[str]
) -> tarfile.TarInfo:
    matches = [
        member for member in members
        if member.isfile() and Path(member.name).stem == stem
        and Path(member.name).suffix.lower() in suffixes
    ]
    if not matches:
        raise FileNotFoundError(f"archive has no audio for session {stem}")
    return sorted(matches, key=lambda member: len(member.name))[0]


def extract_member(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    source = archive.extractfile(member)
    if source is None:
        raise OSError(f"could not read archive member: {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)


def cut_clip(
    source: Path, destination: Path, start: float, duration: float, channel: int
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required to cut AISHELL-4 clips")
    temporary = destination.with_suffix(".tmp.wav")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source),
            "-af",
            f"pan=mono|c0=c{channel}",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(destination)
    with wave.open(str(destination), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"failed to normalize {destination}")


def build_rows(
    selected: list[Interval], source_audio: Path, source_hash: str,
    audio_dir: Path, root: Path, channel: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, interval in enumerate(selected, 1):
        wav = audio_dir / f"aishell4-turn-{index:03d}.wav"
        cut_clip(source_audio, wav, interval.start, interval.duration, channel)
        next_interval = selected[index] if index < len(selected) else None
        natural_gap_ms = (
            round((next_interval.start - interval.end) * 1000)
            if next_interval is not None else 1200
        )
        gap_ms = max(0, min(1200, natural_gap_ms))
        if next_interval is None:
            gap_ms = 1200
        try:
            audio_value = str(wav.relative_to(root))
        except ValueError:
            audio_value = str(wav)
        previous = selected[index - 2] if index > 1 else None
        rows.append(
            {
                "id": f"aishell4-speaker-{index:03d}",
                "audio": audio_value,
                "source_lang": "zh",
                "direction": "zh2en",
                "categories": ["real", "aishell4", "speaker-stress", "far-field"],
                "transcript": interval.text,
                "translation_reference": "",
                "speaker_id": interval.speaker,
                "must_break_before": bool(previous and previous.speaker != interval.speaker),
                "gap_after_ms": gap_ms,
                "target_terms": [],
                "dataset": {
                    "name": "AISHELL-4",
                    "url": DATASET_URL,
                    "split": "test",
                    "session": interval.session,
                    "speaker": interval.speaker,
                    "source_start": interval.start,
                    "source_end": interval.end,
                    "source_channel": channel,
                    "source_audio_sha256": source_hash,
                    "benchmark_audio_sha256": sha256(wav),
                    "license": "CC BY-SA 4.0",
                },
            }
        )
    return rows


def prepare_selected_from_hf(
    args: argparse.Namespace, root: Path, audio_dir: Path, annotation_copy: Path
) -> list[dict[str, object]]:
    cache = args.textgrid_cache.resolve()
    index_path = cache / "index.json"
    curl_file(HF_TREE_URL, index_path)
    listing = json.loads(index_path.read_text(encoding="utf-8"))
    textgrid_names = sorted(
        Path(item["path"]).name for item in listing
        if str(item.get("path", "")).lower().endswith(".textgrid")
    )
    if not textgrid_names:
        raise ValueError("AISHELL mirror returned no test TextGrid files")
    sessions: dict[str, list[Interval]] = {}
    raw_annotations: dict[str, str] = {}
    for name in textgrid_names:
        target = cache / name
        if not target.exists():
            curl_file(f"{HF_RESOLVE_URL}/TextGrid/{name}", target)
        raw = target.read_text(encoding="utf-8-sig", errors="replace")
        stem = target.stem
        parsed = parse_textgrid(raw, stem)
        if parsed:
            sessions[stem] = parsed
            raw_annotations[stem] = raw
    selected = choose_window(sessions, args.count, args.min_duration, args.max_duration)
    session = selected[0].session
    annotation_copy.write_text(raw_annotations[session], encoding="utf-8")
    wav_index = cache / "wav-index.json"
    curl_file(HF_WAV_TREE_URL, wav_index)
    wav_listing = json.loads(wav_index.read_text(encoding="utf-8"))
    wav_path = f"test/wav/{session}.flac"
    wav_item = next((item for item in wav_listing if item.get("path") == wav_path), None)
    if wav_item is None or not isinstance(wav_item.get("size"), int):
        raise ValueError(f"AISHELL mirror did not report the size of {wav_path}")
    source_cache = args.selected_source_cache.resolve()
    source_audio = source_cache / f"{session}.flac"
    print(f"download: selected AISHELL-4 meeting {session}.flac", flush=True)
    download_sized_file(
        f"{HF_RESOLVE_URL}/wav/{session}.flac",
        source_audio,
        int(wav_item["size"]),
        args.download_workers,
    )
    source_hash = sha256(source_audio)
    rows = build_rows(selected, source_audio, source_hash, audio_dir, root, args.channel)
    source_audio.unlink()
    print(f"removed temporary selected meeting: {source_audio}")
    return rows


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    audio_dir = args.audio_dir.resolve()
    suite_path = args.suite.resolve()
    annotation_copy = args.annotation_copy.resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_copy.parent.mkdir(parents=True, exist_ok=True)
    archive_path = args.archive.resolve()
    if args.download_selected:
        rows = prepare_selected_from_hf(args, root, audio_dir, annotation_copy)
    else:
        if args.download:
            download_archive(archive_path, args.download_workers)
        if not archive_path.exists():
            raise FileNotFoundError(
                f"{archive_path}; use --download-selected for the compact official mirror path"
            )
        with tarfile.open(archive_path, "r:*") as archive:
            members = archive.getmembers()
            annotations: dict[str, tuple[tarfile.TarInfo, str]] = {}
            sessions: dict[str, list[Interval]] = {}
            for member in members:
                if not member.isfile() or Path(member.name).suffix.lower() != ".textgrid":
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                raw = extracted.read().decode("utf-8-sig", errors="replace")
                stem = Path(member.name).stem
                parsed = parse_textgrid(raw, stem)
                if parsed:
                    annotations[stem] = (member, raw)
                    sessions[stem] = parsed
            selected = choose_window(sessions, args.count, args.min_duration, args.max_duration)
            session = selected[0].session
            audio_member = find_member_by_stem(members, session, {".wav", ".flac"})
            annotation_copy.write_text(annotations[session][1], encoding="utf-8")
            with tempfile.TemporaryDirectory(prefix="myt-aishell4-") as temporary_dir:
                source_audio = Path(temporary_dir) / Path(audio_member.name).name
                extract_member(archive, audio_member, source_audio)
                rows = build_rows(
                    selected, source_audio, sha256(source_audio), audio_dir, root, args.channel
                )

    suite_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
        + "\n",
        encoding="utf-8",
    )
    changes = sum(bool(row["must_break_before"]) for row in rows)
    print(f"suite: {suite_path}")
    print(f"audio: {audio_dir} ({len(rows)} turns, {changes} required speaker changes)")
    print(f"annotation: {annotation_copy}")
    if args.remove_archive and not args.download_selected:
        archive_path.unlink()
        parts = archive_path.parent / ".test.tar.gz.parts"
        if parts.exists():
            shutil.rmtree(parts)
        print(f"removed source archive: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
