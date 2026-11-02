#!/usr/bin/env python3
"""Compare MyTranslator's fast, balanced, and quality model profiles.

This is a component benchmark: every profile receives the same completed WAV
utterances.  The reported pipeline latency adds the profile's configured VAD
tail budget, but does not include playback time or semantic buffering.  Use
``benchmark_live_pipeline.py`` separately for end-to-end segmentation tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from benchmark_realtime import (
    ServerManager,
    ZhNormalizer,
    chrf,
    edit_distance,
    load_suite,
    normalize_chars,
    normalize_words,
    percentile,
    transcribe,
    translate,
)


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    asr: str
    translation: str
    endpoint_ms: float
    segmentation: str


PROFILES = (
    Profile("fast", "极速", "whisper-medium-q5", "hy-mt2-1.8b-q4", 440.0, "fast"),
    Profile("balanced", "均衡", "qwen3-asr-1.7b-mlx-8bit", "hy-mt2-1.8b-q4", 540.0, "fast"),
    Profile("quality", "高质量", "qwen3-asr-1.7b-mlx-8bit", "hy-mt2-1.8b-q4", 800.0, "steady"),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=root / "benchmarks/suites/fleurs200.jsonl")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--profiles", nargs="+", choices=[p.key for p in PROFILES],
                        default=["fast", "balanced", "quality"])
    parser.add_argument("--output-dir", type=Path, default=root / "benchmarks/results/profiles")
    parser.add_argument("--asr-url", default="http://127.0.0.1:8342")
    parser.add_argument("--translation-url", default="http://127.0.0.1:8341")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0 or args.offset < 0:
        parser.error("--limit must be positive and --offset non-negative")
    # ServerManager expects these shared options.
    args.no_start_servers = False
    args.keep_servers = False
    return args


class JsonlWorker:
    def __init__(self, command: list[str], env: dict[str, str] | None = None) -> None:
        worker_env = os.environ.copy()
        worker_env.update(env or {})
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=worker_env,
        )
        ready_line = self.process.stdout.readline() if self.process.stdout else ""
        if not ready_line:
            error = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"worker failed before ready: {error.strip()}")
        ready = json.loads(ready_line)
        if ready.get("event") != "ready":
            raise RuntimeError(f"worker failed to load: {ready}")
        self.load_ms = float(ready.get("load_ms") or 0)
        self.sequence = 0

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.process.stdin or not self.process.stdout:
            raise RuntimeError("worker pipes are unavailable")
        self.sequence += 1
        payload = {"id": self.sequence, **payload}
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            error = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"worker exited: {error.strip()}")
        value = json.loads(line)
        if value.get("id") != self.sequence:
            raise RuntimeError(f"worker response mismatch: {value}")
        if not value.get("ok"):
            raise RuntimeError(str(value.get("error") or value))
        return value

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


def create_qwen_asr(root: Path) -> JsonlWorker:
    return JsonlWorker(
        [
            str(root / "bin/qwen-asr-dist/qwen-asr-worker"),
            "--model", str(root / "models/qwen3-asr-1.7b"),
            "--idle-seconds", "600",
        ],
        {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"},
    )


def create_qwen_translation(root: Path) -> JsonlWorker:
    return JsonlWorker(
        [
            str(root / "bin/qwen-translation-dist/qwen-translation-worker"),
            "--model", str(root / "models/qwen3.5-2b-mlx-4bit"),
            "--idle-seconds", "600",
        ],
        {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"},
    )


def stats(values: Iterable[float]) -> dict[str, float]:
    items = list(values)
    return {
        "mean": sum(items) / len(items),
        "p50": percentile(items, 0.50),
        "p95": percentile(items, 0.95),
        "max": max(items),
    }


def transcribe_case(
    root: Path,
    profile: Profile,
    case: dict[str, Any],
    qwen_asr: JsonlWorker | None,
    asr_url: str,
) -> tuple[str, float, bool]:
    if profile.asr.startswith("whisper"):
        text, elapsed_ms = transcribe(
            asr_url, Path(case["audio_path"]), case["source_lang"]
        )
        return text, elapsed_ms, False
    if qwen_asr is None:
        raise RuntimeError("Qwen ASR worker is not running")
    started = time.perf_counter()
    try:
        value = qwen_asr.request(
            {
                "wav_path": str(Path(case["audio_path"]).resolve()),
                "language": case["source_lang"],
                "context": "",
            }
        )
        return (
            str(value.get("text") or "").strip(),
            float(value.get("elapsed_ms") or 0),
            False,
        )
    except RuntimeError:
        # This mirrors MyTranslator's production Qwen-ASR -> Whisper Medium
        # failure path and includes the failed attempt in user-visible latency.
        text, _ = transcribe(asr_url, Path(case["audio_path"]), case["source_lang"])
        return text, (time.perf_counter() - started) * 1000.0, True


def translate_case(
    profile: Profile,
    text: str,
    direction: str,
    qwen_translation: JsonlWorker | None,
    translation_url: str,
) -> tuple[str, float, bool]:
    if profile.translation.startswith("hy-mt2"):
        result, elapsed_ms, _, _ = translate(translation_url, text, direction, [])
        return result, elapsed_ms, False
    if qwen_translation is None:
        raise RuntimeError("Qwen translation worker is not running")
    started = time.perf_counter()
    try:
        value = qwen_translation.request(
            {
                "mode": "translate",
                "source": text,
                "direction": direction,
                "context": [],
                "glossary": [],
            }
        )
        return (
            str(value.get("translation") or "").strip(),
            float(value.get("elapsed_ms") or 0),
            False,
        )
    except RuntimeError:
        # Production direct-Qwen mode falls back to Hy-MT2 on malformed or
        # failed responses. Preserve that behavior and charge both attempts.
        result, _, _, _ = translate(translation_url, text, direction, [])
        return result, (time.perf_counter() - started) * 1000.0, True


def run_profile(
    root: Path,
    profile: Profile,
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    normalizer: ZhNormalizer,
) -> dict[str, Any]:
    qwen_asr = create_qwen_asr(root) if profile.asr.startswith("qwen") else None
    qwen_translation = (
        create_qwen_translation(root) if profile.translation.startswith("qwen") else None
    )
    try:
        # Warm both paths with the first sample. Model loading is reported
        # separately and is intentionally excluded from steady-state latency.
        first = cases[0]
        warm_text, _, _ = transcribe_case(root, profile, first, qwen_asr, args.asr_url)
        if first["source_lang"] == "zh":
            warm_text = normalizer.convert(warm_text)
        translate_case(
            profile, warm_text, first["direction"], qwen_translation,
            args.translation_url,
        )

        rows: list[dict[str, Any]] = []
        for index, case in enumerate(cases, 1):
            if not args.quiet:
                print(f"[{profile.label}] [{index}/{len(cases)}] {case['id']}", flush=True)
            text, asr_ms, asr_fallback = transcribe_case(
                root, profile, case, qwen_asr, args.asr_url
            )
            if case["source_lang"] == "zh":
                text = normalizer.convert(text)
            translation_text, translation_ms, translation_fallback = translate_case(
                profile, text, case["direction"], qwen_translation, args.translation_url
            )
            ref_units = (
                normalize_chars(case["transcript"])
                if case["source_lang"] == "zh"
                else normalize_words(case["transcript"])
            )
            hyp_units = (
                normalize_chars(text)
                if case["source_lang"] == "zh"
                else normalize_words(text)
            )
            rows.append(
                {
                    "id": case["id"],
                    "source_lang": case["source_lang"],
                    "reference_transcript": case["transcript"],
                    "transcript": text,
                    "asr_distance": edit_distance(ref_units, hyp_units),
                    "asr_reference_units": len(ref_units),
                    "translation_reference": case.get("translation_reference", ""),
                    "translation": translation_text,
                    "translation_chrf": chrf(
                        case.get("translation_reference", ""), translation_text
                    ),
                    "asr_ms": round(asr_ms, 2),
                    "translation_ms": round(translation_ms, 2),
                    "asr_fallback": asr_fallback,
                    "translation_fallback": translation_fallback,
                    "estimated_pipeline_ms": round(
                        profile.endpoint_ms + asr_ms + translation_ms, 2
                    ),
                }
            )
        zh = [row for row in rows if row["source_lang"] == "zh"]
        en = [row for row in rows if row["source_lang"] == "en"]

        def error_rate(items: list[dict[str, Any]]) -> float | None:
            if not items:
                return None
            return sum(row["asr_distance"] for row in items) / max(
                1, sum(row["asr_reference_units"] for row in items)
            )

        summary = {
            "samples": len(rows),
            "zh_cer": error_rate(zh),
            "en_wer": error_rate(en),
            "translation_chrf": sum(row["translation_chrf"] for row in rows) / len(rows),
            "asr_fallbacks": sum(bool(row["asr_fallback"]) for row in rows),
            "translation_fallbacks": sum(
                bool(row["translation_fallback"]) for row in rows
            ),
            "asr_ms": stats(row["asr_ms"] for row in rows),
            "translation_ms": stats(row["translation_ms"] for row in rows),
            "estimated_pipeline_ms": stats(row["estimated_pipeline_ms"] for row in rows),
            "cold_load_ms": {
                "asr": qwen_asr.load_ms if qwen_asr else 0,
                "translation": qwen_translation.load_ms if qwen_translation else 0,
            },
        }
        return {"profile": profile.__dict__, "summary": summary, "results": rows}
    finally:
        if qwen_translation:
            qwen_translation.close()
        if qwen_asr:
            qwen_asr.close()


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# MyTranslator 三档能力对比",
        "",
        f"- 时间：{report['created_at']}",
        f"- 套件：`{report['suite']}`",
        f"- 同题样本：{report['sample_count']} 条（中英文交替）",
        "- 性质：完成音频的组件基准；总延迟为断句预算 + ASR + 翻译，不含音频播放时长和语义缓冲",
        "- 说明：三档都由 Hy-MT2 首译；本组件基准不触发均衡/高质量档的选择性 Qwen3.5 异步复核",
        "",
        "| 档位 | 识别 / 翻译 | 中文 CER ↓ | 英文 WER ↓ | 翻译 chrF ↑ | ASR P50/P95 | 翻译 P50/P95 | 估算总延迟 P50/P95 | 回退 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["profiles"]:
        profile, summary = item["profile"], item["summary"]
        lines.append(
            f"| {profile['label']} | {profile['asr']} / {profile['translation']} | "
            f"{pct(summary['zh_cer'])} | {pct(summary['en_wer'])} | "
            f"{pct(summary['translation_chrf'])} | "
            f"{summary['asr_ms']['p50']:.0f}/{summary['asr_ms']['p95']:.0f} ms | "
            f"{summary['translation_ms']['p50']:.0f}/{summary['translation_ms']['p95']:.0f} ms | "
            f"{summary['estimated_pipeline_ms']['p50']:.0f}/{summary['estimated_pipeline_ms']['p95']:.0f} ms | "
            f"ASR {summary['asr_fallbacks']} / 翻译 {summary['translation_fallbacks']} |"
        )
    lines.extend([
        "",
        "## 档位定义",
        "",
    ])
    for item in report["profiles"]:
        p = item["profile"]
        lines.append(
            f"- **{p['label']}**：{p['asr']} + {p['translation']}，"
            f"{p['segmentation']} 断句，计入 {p['endpoint_ms']:.0f}ms 句尾预算。"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cases = load_suite(args.suite.resolve(), root)[args.offset : args.offset + args.limit]
    if not cases:
        raise ValueError("selected suite range is empty")
    selected = [profile for profile in PROFILES if profile.key in args.profiles]
    manager = ServerManager(root, args)
    normalizer = ZhNormalizer(root)
    try:
        manager.ensure()
        profiles = [run_profile(root, profile, cases, args, normalizer) for profile in selected]
    finally:
        normalizer.close()
        manager.close()

    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "suite": str(args.suite.resolve()),
        "sample_count": len(cases),
        "profiles": profiles,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"profile-comparison-{stamp}.json"
    md_path = args.output_dir / f"profile-comparison-{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report))
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
