#!/usr/bin/env python3
"""Build a small, deterministic CoVoST 2 zh-CN -> en test suite.

Use --download-preview for the public 100-row Hugging Face mirror preview, or
after accepting Mozilla Data Collective's terms point the tool at its archive
or an extracted directory. Only the selected clips are decoded and retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import wave
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DATASET_ID = "cmp5kv8j101hvo007hjamlqet"
DATASET_URL = f"https://mozilladatacollective.com/datasets/{DATASET_ID}"
HF_MIRROR_URL = "https://huggingface.co/datasets/fixie-ai/covost2"
HF_PREVIEW_URL = (
    "https://datasets-server.huggingface.co/first-rows?"
    "dataset=fixie-ai%2Fcovost2&config=zh-CN_en&split=test"
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        help="MDC CoVoST 2 archive (.tar.gz/.tgz) or extracted dataset directory",
    )
    parser.add_argument(
        "--download-preview",
        action="store_true",
        help="build from the public fixie-ai/covost2 test preview (up to 100 rows)",
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--meta-tsv",
        type=Path,
        default=root / "benchmarks/cache/covost2/covost_v2.zh-CN_en.tsv",
        help="official Meta translation TSV, used when the bundle separates Common Voice metadata",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=root / "benchmarks/audio/covost2-small",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=root / "benchmarks/suites/covost2-zh-en-small.jsonl",
    )
    parser.add_argument(
        "--preview-cache",
        type=Path,
        default=root / "benchmarks/cache/covost2/fixie-first-rows.json",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be >= 1")
    if args.download_preview and args.source is not None:
        parser.error("pass either SOURCE or --download-preview, not both")
    if not args.download_preview and args.source is None:
        parser.error("SOURCE is required unless --download-preview is used")
    if args.download_preview and args.count > 100:
        parser.error("--download-preview supports at most 100 rows")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv_bytes(raw: bytes) -> list[dict[str, str]]:
    text = raw.decode("utf-8-sig")
    return [dict(row) for row in csv.DictReader(io.StringIO(text), delimiter="\t")]


def read_tsv(path: Path) -> list[dict[str, str]]:
    return read_tsv_bytes(path.read_bytes())


class DatasetSource:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.archive: tarfile.TarFile | None = None
        self.audio: dict[str, Path | tarfile.TarInfo] = {}
        self.tables: list[tuple[str, list[dict[str, str]]]] = []

    def __enter__(self) -> "DatasetSource":
        if self.path.is_dir():
            for candidate in self.path.rglob("*.tsv"):
                try:
                    rows = read_tsv(candidate)
                except (OSError, UnicodeError, csv.Error):
                    continue
                if rows:
                    self.tables.append((str(candidate), rows))
            for suffix in ("*.mp3", "*.wav", "*.flac", "*.m4a"):
                for candidate in self.path.rglob(suffix):
                    self.audio.setdefault(candidate.name, candidate)
            return self

        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.archive = tarfile.open(self.path, "r:*")
        for member in self.archive.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if member.name.lower().endswith(".tsv"):
                extracted = self.archive.extractfile(member)
                if extracted is None:
                    continue
                try:
                    rows = read_tsv_bytes(extracted.read())
                except (UnicodeError, csv.Error):
                    continue
                if rows:
                    self.tables.append((member.name, rows))
            elif Path(name).suffix.lower() in {".mp3", ".wav", ".flac", ".m4a"}:
                self.audio.setdefault(name, member)
        return self

    def __exit__(self, *_: object) -> None:
        if self.archive is not None:
            self.archive.close()

    def copy_audio(self, name: str, destination: Path) -> None:
        item = self.audio.get(Path(name).name)
        if item is None:
            raise FileNotFoundError(f"audio referenced by metadata was not found: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(item, Path):
            shutil.copyfile(item, destination)
            return
        assert self.archive is not None
        extracted = self.archive.extractfile(item)
        if extracted is None:
            raise OSError(f"could not read archive member: {item.name}")
        with destination.open("wb") as output:
            shutil.copyfileobj(extracted, output, length=1024 * 1024)


class PreviewDatasetSource:
    """Expose the public Hugging Face dataset preview through DatasetSource's interface."""

    def __init__(self, cache: Path) -> None:
        self.cache = cache.resolve()
        self.audio: dict[str, str] = {}
        self.tables: list[tuple[str, list[dict[str, str]]]] = []

    def __enter__(self) -> "PreviewDatasetSource":
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "curl", "-L", "--fail", "--retry", "5", "--retry-all-errors",
                "--silent", "--show-error", HF_PREVIEW_URL, "-o", str(self.cache),
            ],
            check=True,
        )
        payload = json.loads(self.cache.read_text(encoding="utf-8"))
        rows: list[dict[str, str]] = []
        for item in payload.get("rows", []):
            row = item.get("row", {})
            audio = row.get("audio", [])
            if not isinstance(audio, list) or not audio or not audio[0].get("src"):
                continue
            original_name = Path(str(row.get("file", row.get("id", "")))).name
            if not original_name:
                continue
            source_url = str(audio[0]["src"])
            self.audio[original_name] = source_url
            rows.append(
                {
                    "path": original_name,
                    "sentence": str(row.get("sentence", "")),
                    "translation": str(row.get("translation", "")),
                    "client_id": str(row.get("client_id", "")),
                    "split": "test",
                    "mirror_revision": source_url.split("/--/")[1].split("/", 1)[0]
                    if "/--/" in source_url else "unknown",
                }
            )
        if not rows:
            raise ValueError("the Hugging Face CoVoST 2 preview returned no usable rows")
        self.tables = [(HF_PREVIEW_URL, rows)]
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def copy_audio(self, name: str, destination: Path) -> None:
        source_url = self.audio.get(Path(name).name)
        if source_url is None:
            raise FileNotFoundError(f"preview audio was not found: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "curl", "-L", "--fail", "--retry", "5", "--retry-all-errors",
                "--silent", "--show-error", source_url, "-o", str(destination),
            ],
            check=True,
        )


def load_meta_translations(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {Path(row.get("path", "")).name: row for row in read_tsv(path) if row.get("path")}


def resolve_rows(
    source: DatasetSource, meta_path: Path, split: str
) -> list[dict[str, str]]:
    meta = load_meta_translations(meta_path)
    complete: list[dict[str, str]] = []
    validated: list[dict[str, str]] = []
    for _name, rows in source.tables:
        columns = set(rows[0])
        if {"path", "sentence", "translation"} <= columns:
            complete.extend(rows)
        elif {"path", "sentence"} <= columns:
            validated.extend(rows)

    candidates: list[dict[str, str]] = []
    if complete:
        candidates = complete
    elif validated and meta:
        for row in validated:
            translation = meta.get(Path(row.get("path", "")).name)
            if translation is None:
                continue
            candidates.append({**row, **translation})
    else:
        found = ", ".join(name for name, _ in source.tables) or "none"
        raise ValueError(
            "could not find path/sentence/translation metadata. "
            f"TSV files found: {found}. Pass --meta-tsv if needed."
        )

    result = []
    for row in candidates:
        row_split = row.get("split", "").strip()
        if row_split and row_split != split:
            continue
        if not row.get("path") or not row.get("sentence") or not row.get("translation"):
            continue
        if Path(row["path"]).name not in source.audio:
            continue
        result.append(row)
    if not result:
        raise ValueError(f"no usable CoVoST rows for split={split!r}")
    return result


def length_band(text: str) -> str:
    count = len("".join(text.split()))
    if count <= 12:
        return "short"
    if count <= 30:
        return "medium"
    return "long"


def select_rows(rows: Iterable[dict[str, str]], count: int) -> list[dict[str, str]]:
    bands: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: Path(item["path"]).name):
        bands[length_band(row["sentence"])].append(row)

    selected: list[dict[str, str]] = []
    speaker_counts: dict[str, int] = defaultdict(int)
    order = ("short", "medium", "long")
    while len(selected) < count and any(bands.values()):
        progressed = False
        for band in order:
            while bands[band]:
                row = bands[band].pop(0)
                speaker = row.get("client_id", "")
                if speaker and speaker_counts[speaker] >= 2:
                    continue
                selected.append(row)
                speaker_counts[speaker] += 1
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} diverse rows available; requested {count}")
    return selected


def convert_to_wav(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required to decode the CoVoST audio")
    temporary = destination.with_suffix(".tmp.wav")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ar",
            "16000",
            "-ac",
            "1",
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


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    audio_dir = args.audio_dir.resolve()
    suite_path = args.suite.resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    suite_path.parent.mkdir(parents=True, exist_ok=True)

    source_context = (
        PreviewDatasetSource(args.preview_cache)
        if args.download_preview else DatasetSource(args.source)
    )
    with source_context as source:
        rows = resolve_rows(source, args.meta_tsv.resolve(), args.split)
        selected = select_rows(rows, args.count)
        output_rows: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="myt-covost2-") as temporary_dir:
            temporary = Path(temporary_dir)
            for index, row in enumerate(selected, 1):
                original_name = Path(row["path"]).name
                extracted = temporary / original_name
                source.copy_audio(original_name, extracted)
                wav = audio_dir / f"covost2-zh-en-{index:03d}.wav"
                convert_to_wav(extracted, wav)
                try:
                    audio_value = str(wav.relative_to(root))
                except ValueError:
                    audio_value = str(wav)
                output_rows.append(
                    {
                        "id": f"covost2-zh-en-{index:03d}",
                        "audio": audio_value,
                        "source_lang": "zh",
                        "direction": "zh2en",
                        "categories": ["real", "covost2", length_band(row["sentence"])],
                        "transcript": row["sentence"],
                        "translation_reference": row["translation"],
                        "speaker_id": row.get("client_id", "") or f"unknown-{index:03d}",
                        "gap_after_ms": 1200,
                        "target_terms": [],
                        "dataset": {
                            "name": "CoVoST 2 Chinese (China) - English",
                            "dataset_id": DATASET_ID,
                            "url": DATASET_URL,
                            "split": args.split,
                            "source_path": original_name,
                            "source_audio_sha256": sha256(extracted),
                            "benchmark_audio_sha256": sha256(wav),
                            "license": "CC-BY-NC-4.0; MDC dataset-specific restrictions also apply",
                            **(
                                {
                                    "mirror": HF_MIRROR_URL,
                                    "mirror_revision": row.get("mirror_revision", "unknown"),
                                    "retrieval": "Hugging Face datasets-server first-rows preview",
                                }
                                if args.download_preview else {}
                            ),
                        },
                    }
                )

    suite_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in output_rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"suite: {suite_path}")
    print(f"audio: {audio_dir} ({len(output_rows)} selected test clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
