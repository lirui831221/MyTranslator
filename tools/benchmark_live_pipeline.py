#!/usr/bin/env python3
"""Replay a suite through macOS system audio and score the complete app pipeline.

Unlike benchmark_realtime.py, this exercises ScreenCaptureKit, VAD, rolling
ASR, speaker turns, semantic buffering, translation ordering, and UI events.
It launches an isolated hidden MyTranslator process and records its local-only
subtitle WebSocket stream.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import os
import re
import signal
import socket
import struct
import subprocess
import threading
import time
import wave
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from benchmark_realtime import chrf, edit_distance, normalize_chars


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--tag", default="live")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--profile", choices=("fast", "balanced", "quality"), default="balanced")
    parser.add_argument(
        "--vad", choices=("fast", "steady"),
        help="override the profile's VAD only for segmentation experiments",
    )
    parser.add_argument("--asr-lang", choices=("zh", "en", "auto"), default="zh")
    parser.add_argument("--ws-port", type=int, default=18080)
    parser.add_argument("--leading-silence-ms", type=int, default=2500)
    parser.add_argument("--trailing-silence-ms", type=int, default=4500)
    parser.add_argument("--default-gap-ms", type=int, default=1200)
    parser.add_argument("--drain-timeout", type=float, default=30.0)
    parser.add_argument("--offset", type=int, default=0, help="skip the first N suite rows")
    parser.add_argument("--limit", type=int, help="only replay the first N suite rows")
    parser.add_argument(
        "--app-binary",
        type=Path,
        default=root / "target/release/my-translator",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "benchmarks/results/live",
    )
    parser.add_argument("--keep-replay", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.ws_port <= 65535:
        parser.error("--ws-port must be in 1..65535")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if min(args.leading_silence_ms, args.trailing_silence_ms, args.default_gap_ms) < 0:
        parser.error("silence durations must be non-negative")
    return args


def load_suite(path: Path, root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            case = json.loads(line)
            missing = {"id", "audio", "transcript"} - case.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
            audio = Path(case["audio"])
            audio = audio if audio.is_absolute() else root / audio
            if not audio.exists():
                raise FileNotFoundError(audio)
            case["audio_path"] = str(audio.resolve())
            cases.append(case)
    if not cases:
        raise ValueError(f"empty suite: {path}")
    return cases


def read_pcm16(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as audio:
        details = (audio.getframerate(), audio.getnchannels(), audio.getsampwidth())
        if details != (16000, 1, 2):
            raise ValueError(f"{path}: expected 16 kHz mono PCM16 WAV, got {details}")
        frames = audio.readframes(audio.getnframes())
        return frames, audio.getnframes()


def make_replay(
    cases: list[dict[str, Any]], destination: Path, leading_ms: int,
    trailing_ms: int, default_gap_ms: int,
) -> list[dict[str, Any]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    cursor = int(leading_ms * 16)
    timeline: list[dict[str, Any]] = []
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * cursor)
        previous_speaker = None
        for case in cases:
            frames, frame_count = read_pcm16(Path(case["audio_path"]))
            start_frame = cursor
            output.writeframes(frames)
            cursor += frame_count
            speaker = str(case.get("speaker_id", ""))
            must_break_before = bool(
                case.get("must_break_before", False)
                or (
                    case.get("evaluate_speaker_change", False)
                    and previous_speaker is not None
                    and speaker
                    and speaker != previous_speaker
                )
            )
            timeline.append(
                {
                    **{key: value for key, value in case.items() if key != "audio_path"},
                    "start_ms": round(start_frame / 16.0, 2),
                    "end_ms": round(cursor / 16.0, 2),
                    "must_break_before": must_break_before,
                }
            )
            previous_speaker = speaker or previous_speaker
            gap_ms = int(case.get("gap_after_ms", default_gap_ms))
            gap_frames = int(gap_ms * 16)
            output.writeframes(b"\0\0" * gap_frames)
            cursor += gap_frames
        trailing_frames = int(trailing_ms * 16)
        output.writeframes(b"\0\0" * trailing_frames)
    return timeline


def recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        block = sock.recv(count - len(chunks))
        if not block:
            raise ConnectionError("WebSocket closed")
        chunks.extend(block)
    return bytes(chunks)


class EventCollector(threading.Thread):
    def __init__(self, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.events: list[dict[str, Any]] = []
        self.error: str | None = None
        self.connected = threading.Event()
        self.stopping = threading.Event()
        self.sock: socket.socket | None = None
        self.playback_start: float | None = None

    def stop(self) -> None:
        self.stopping.set()
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()

    def run(self) -> None:
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=10)
            self.sock.settimeout(1.0)
            key = base64.b64encode(os.urandom(16)).decode("ascii")
            request = (
                f"GET / HTTP/1.1\r\nHost: {self.host}:{self.port}\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            )
            self.sock.sendall(request.encode("ascii"))
            response = bytearray()
            while b"\r\n\r\n" not in response:
                response.extend(self.sock.recv(4096))
            if not response.startswith(b"HTTP/1.1 101"):
                raise ConnectionError(response.decode("utf-8", errors="replace"))
            self.connected.set()
            while not self.stopping.is_set():
                try:
                    first, second = recv_exact(self.sock, 2)
                except socket.timeout:
                    continue
                opcode = first & 0x0F
                length = second & 0x7F
                if length == 126:
                    length = struct.unpack("!H", recv_exact(self.sock, 2))[0]
                elif length == 127:
                    length = struct.unpack("!Q", recv_exact(self.sock, 8))[0]
                masked = bool(second & 0x80)
                mask = recv_exact(self.sock, 4) if masked else b""
                payload = bytearray(recv_exact(self.sock, length))
                if masked:
                    for index in range(length):
                        payload[index] ^= mask[index % 4]
                if opcode == 8:
                    break
                if opcode != 1:
                    continue
                now = time.monotonic()
                event = json.loads(payload.decode("utf-8"))
                event["monotonic"] = now
                if self.playback_start is not None:
                    event["playback_ms"] = round((now - self.playback_start) * 1000, 2)
                self.events.append(event)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            if not self.stopping.is_set():
                self.error = str(error)
                self.connected.set()


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def wait_for_port(port: int, process: subprocess.Popen[bytes], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_open(port):
            return
        if process.poll() is not None:
            raise RuntimeError(f"MyTranslator exited during startup ({process.returncode})")
        time.sleep(0.1)
    raise TimeoutError(f"subtitle WebSocket did not open on port {port}")


def wait_for_log(
    path: Path, offset: int, marker: str, process: subprocess.Popen[bytes], timeout: float
) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            with path.open("rb") as handle:
                handle.seek(offset)
                text = handle.read().decode("utf-8", errors="replace")
            if marker in text:
                return text
            if "system source unavailable; falling back" in text:
                raise RuntimeError("system audio capture fell back to microphone; check permissions")
        if process.poll() is not None:
            raise RuntimeError(f"MyTranslator exited during audio startup ({process.returncode})")
        time.sleep(0.15)
    raise TimeoutError(f"did not observe readiness marker: {marker}")


def terminate_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def units(text: str) -> list[str]:
    return normalize_chars(text or "")


def distance(reference: str, hypothesis: str) -> tuple[int, int, float]:
    ref = units(reference)
    hyp = units(hypothesis)
    edits = edit_distance(ref, hyp)
    return edits, len(ref), edits / max(1, len(ref))


def align(
    timeline: list[dict[str, Any]], outputs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    n, m = len(timeline), len(outputs)
    infinity = float("inf")
    dp = [[infinity] * (m + 1) for _ in range(n + 1)]
    parent: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if not math.isfinite(dp[i][j]):
                continue
            for ref_count in range(1, min(4, n - i) + 1):
                for out_count in range(1, min(4, m - j) + 1):
                    reference = "".join(row["transcript"] for row in timeline[i : i + ref_count])
                    hypothesis = "".join(row["data"].get("source", "") for row in outputs[j : j + out_count])
                    edits, ref_units, _ = distance(reference, hypothesis)
                    cost = edits / max(1, ref_units) + 0.025 * (ref_count + out_count - 2)
                    ni, nj = i + ref_count, j + out_count
                    if dp[i][j] + cost < dp[ni][nj]:
                        dp[ni][nj] = dp[i][j] + cost
                        parent[ni][nj] = (i, j)

    if parent[n][m] is None:
        return []
    groups: list[tuple[int, int, int, int]] = []
    i, j = n, m
    while i or j:
        previous = parent[i][j]
        if previous is None:
            break
        pi, pj = previous
        groups.append((pi, i, pj, j))
        i, j = pi, pj
    groups.reverse()

    aligned: list[dict[str, Any]] = []
    for ref_start, ref_end, out_start, out_end in groups:
        refs = timeline[ref_start:ref_end]
        hypotheses = outputs[out_start:out_end]
        ref_asr = "".join(row["transcript"] for row in refs)
        hyp_asr = "".join(row["data"].get("source", "") for row in hypotheses)
        ref_translation = " ".join(
            str(row.get("translation_reference", "")) for row in refs
            if row.get("translation_reference")
        )
        hyp_translation = " ".join(
            str(row["data"].get("translation", "")) for row in hypotheses
        )
        edits, ref_units, cer = distance(ref_asr, hyp_asr)
        final_event_ms = max(float(row.get("playback_ms", 0)) for row in hypotheses)
        reference_end_ms = max(float(row["end_ms"]) for row in refs)
        aligned.append(
            {
                "reference_ids": [row["id"] for row in refs],
                "output_sequences": [row["data"].get("sequence") for row in hypotheses],
                "reference_source": ref_asr,
                "recognized_source": hyp_asr,
                "reference_translation": ref_translation,
                "translation": hyp_translation,
                "asr_edits": edits,
                "asr_reference_units": ref_units,
                "cer": cer,
                "translation_chrf": chrf(ref_translation, hyp_translation) if ref_translation else None,
                "translation_latency_ms": round(final_event_ms - reference_end_ms, 2),
            }
        )
    return aligned


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse_log_metrics(log_text: str) -> dict[str, Any]:
    asr_depths = [int(value) for value in re.findall(r"ASR queue depth=(\d+)", log_text)]
    translation_depths = [int(value) for value in re.findall(r"queued translation sequence=\d+.*?depth=(\d+)", log_text)]
    selective = re.findall(r"selective_trigger/suppressed=(\d+)/(\d+)", log_text)
    background = re.findall(
        r"background_active/queued/applied/discarded=(\d+)/(\d+)/(\d+)/(\d+)",
        log_text,
    )
    qwen = re.findall(r"qwen_success/queued/fallback=(\d+)/(\d+)/(\d+)", log_text)
    selective_trigger_logs = len(
        re.findall(r"\[asr-selective\].*(?:high-accuracy trigger|suppressed optional review)", log_text)
    )
    selective_suppressed_logs = len(re.findall(r"\[asr-selective\] suppressed", log_text))
    background_review_logs = len(
        re.findall(r"\[asr-background\] primary accepted immediately", log_text)
    )
    background_revision_logs = len(
        re.findall(r"\[asr-background\] queued revision utterance=", log_text)
    )
    qwen_review_logs = len(re.findall(r"review engine=qwen3-asr-", log_text))
    qwen_fallback_logs = len(re.findall(r"\[asr-qwen\].*falling back", log_text))
    background_discard_logs = len(
        re.findall(
            r"\[asr-background\].*(?:skipped|failed|decision=(?:Primary|Reject))",
            log_text,
        )
    )
    natural_policy = re.findall(r"natural_recent_hit/shortened=(\d+)‰/(\d+)", log_text)
    counters = {
        "quality_retries": len(re.findall(r"\[asr-quality\].*review", log_text)),
        "quality_drops": len(re.findall(r"\[asr-quality\].*dropped", log_text)),
        "speaker_boundaries": len(re.findall(r"applying hard speaker boundary", log_text)),
        "natural_merges": len(re.findall(r"merged same-speaker natural continuation", log_text)),
        "context_retries": len(re.findall(r"structural leak detected", log_text)),
        "selective_high_accuracy_triggers": (
            max(int(selective[-1][0]), selective_trigger_logs)
            if selective
            else selective_trigger_logs
        ),
        "selective_high_accuracy_suppressed": (
            max(int(selective[-1][1]), selective_suppressed_logs)
            if selective
            else selective_suppressed_logs
        ),
        "background_active": int(background[-1][0]) if background else 0,
        "background_reviews": (
            max(int(background[-1][1]), background_review_logs)
            if background
            else background_review_logs
        ),
        "background_revisions": (
            max(int(background[-1][2]), background_revision_logs)
            if background
            else background_revision_logs
        ),
        "background_reviews_discarded": (
            max(int(background[-1][3]), background_discard_logs)
            if background
            else background_discard_logs
        ),
        "qwen_reviews": (
            max(int(qwen[-1][1]), qwen_review_logs) if qwen else qwen_review_logs
        ),
        "qwen_review_successes": (
            max(int(qwen[-1][0]), qwen_review_logs) if qwen else qwen_review_logs
        ),
        "qwen_review_fallbacks": (
            max(int(qwen[-1][2]), qwen_fallback_logs) if qwen else qwen_fallback_logs
        ),
        "natural_wait_recent_hit_rate": (
            int(natural_policy[-1][0]) / 1000.0 if natural_policy else None
        ),
        "natural_wait_shortened": (
            int(natural_policy[-1][1]) if natural_policy
            else len(re.findall(r"low-yield wait gate active", log_text))
        ),
    }
    return {
        "asr_queue_peak": max(asr_depths, default=1 if "[asr] completed" in log_text else 0),
        "translation_queue_peak": max(translation_depths, default=0),
        **counters,
    }


def summarize(
    timeline: list[dict[str, Any]], events: list[dict[str, Any]], log_text: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    heard = [event for event in events if event.get("type") == "heard"]
    raw_translated = [event for event in events if event.get("type") == "translated"]
    initial_by_batch: dict[int, dict[str, Any]] = {}
    latest_by_batch: dict[int, dict[str, Any]] = {}
    batch_order: list[int] = []
    for event in raw_translated:
        data = event.get("data", {})
        batch_id = int(data.get("batch_id") or data.get("sequence") or len(batch_order) + 1)
        if batch_id not in initial_by_batch:
            initial_by_batch[batch_id] = event
            batch_order.append(batch_id)
        previous = latest_by_batch.get(batch_id)
        if previous is None or int(data.get("revision", 0)) >= int(
            previous.get("data", {}).get("revision", 0)
        ):
            latest_by_batch[batch_id] = event
    initial_translated = [initial_by_batch[batch_id] for batch_id in batch_order]
    translated = [latest_by_batch[batch_id] for batch_id in batch_order]
    aligned = align(timeline, translated) if translated else []
    initial_aligned = align(timeline, initial_translated) if initial_translated else []
    ref_all = "".join(row["transcript"] for row in timeline)
    hyp_all = "".join(event.get("data", {}).get("source", "") for event in translated)
    edits, ref_units, cer = distance(ref_all, hyp_all)
    chrfs = [row["translation_chrf"] for row in aligned if row["translation_chrf"] is not None]
    latencies = [max(0.0, float(row["translation_latency_ms"])) for row in aligned]
    initial_latencies = [
        max(0.0, float(row["translation_latency_ms"])) for row in initial_aligned
    ]

    fragment_turns = {
        int(event["data"]["id"]): event["data"].get("speaker_turn_id")
        for event in heard if isinstance(event.get("data"), dict) and "id" in event["data"]
    }
    merge_violations = []
    for event in translated:
        fragment_ids = [int(value) for value in event.get("data", {}).get("fragment_ids", [])]
        turns = {fragment_turns.get(value) for value in fragment_ids if fragment_turns.get(value) is not None}
        if len(turns) > 1:
            merge_violations.append(
                {
                    "sequence": event.get("data", {}).get("sequence"),
                    "fragment_ids": fragment_ids,
                    "speaker_turn_ids": sorted(turns),
                    "source": event.get("data", {}).get("source", ""),
                }
            )

    expected_changes = sum(bool(row.get("must_break_before")) for row in timeline)
    heard_turns = [event.get("data", {}).get("speaker_turn_id") for event in heard]
    heard_turns = [value for value in heard_turns if value is not None]
    detected_changes = sum(left != right for left, right in zip(heard_turns, heard_turns[1:]))
    end_reasons = Counter(
        event.get("data", {}).get("end_reason", "unknown") for event in heard
    )
    summary = {
        "reference_segments": len(timeline),
        "heard_fragments": len(heard),
        "translated_batches": len(translated),
        "translation_revision_updates": len(raw_translated) - len(translated),
        "asr_edits": edits,
        "asr_reference_units": ref_units,
        "zh_cer": cer,
        "translation_chrf": sum(chrfs) / len(chrfs) if chrfs else None,
        "translation_latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "initial_translation_latency_ms": {
            "p50": percentile(initial_latencies, 0.50),
            "p95": percentile(initial_latencies, 0.95),
            "max": max(initial_latencies, default=None),
        },
        "expected_speaker_changes": expected_changes,
        "detected_speaker_changes": detected_changes,
        "speaker_change_recall": min(1.0, detected_changes / expected_changes) if expected_changes else None,
        "cross_speaker_merge_violations": len(merge_violations),
        "merge_violations": merge_violations,
        "end_reasons": dict(end_reasons),
        **parse_log_metrics(log_text),
    }
    return summary, aligned


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def baseline_delta(summary: dict[str, Any], path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    old = json.loads(path.read_text(encoding="utf-8"))["summary"]
    return {
        "path": str(path.resolve()),
        "zh_cer": summary["zh_cer"] - old["zh_cer"],
        "translation_chrf": (
            summary["translation_chrf"] - old["translation_chrf"]
            if summary["translation_chrf"] is not None and old.get("translation_chrf") is not None
            else None
        ),
        "cross_speaker_merge_violations": summary["cross_speaker_merge_violations"]
        - old.get("cross_speaker_merge_violations", 0),
    }


def html_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = []
    for row in report["alignments"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(', '.join(row['reference_ids']))}</td>"
            f"<td>{html.escape(row['reference_source'])}<small>{html.escape(row['recognized_source'])}</small></td>"
            f"<td>{html.escape(row['reference_translation'])}<small>{html.escape(row['translation'])}</small></td>"
            f"<td>{fmt_pct(row['cer'])}</td>"
            f"<td>{fmt_pct(row['translation_chrf'])}</td>"
            f"<td>{fmt_ms(max(0, row['translation_latency_ms']))}</td>"
            "</tr>"
        )
    violation = "通过" if summary["cross_speaker_merge_violations"] == 0 else "失败"
    delta = report.get("baseline_delta")
    delta_section = ""
    if delta:
        chrf_delta = delta.get("translation_chrf")
        delta_section = f"""
<h2>相对基准变化</h2>
<div class="muted">基准：{html.escape(delta['path'])}</div>
<table><thead><tr><th>指标</th><th>变化</th><th>判读</th></tr></thead><tbody>
<tr><td>中文 CER</td><td>{delta['zh_cer'] * 100:+.2f} 个百分点</td><td>负数更好</td></tr>
<tr><td>翻译 chrF</td><td>{'n/a' if chrf_delta is None else f'{chrf_delta * 100:+.2f} 个百分点'}</td><td>正数更好</td></tr>
<tr><td>跨说话人合句</td><td>{delta['cross_speaker_merge_violations']:+d}</td><td>不得增加</td></tr>
</tbody></table>"""
    return f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MyTranslator 实时全链路报告</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1115;color:#e8eaf0;margin:0;padding:32px}}
main{{max-width:1200px;margin:auto}}h1{{margin:0 0 6px}}.muted,small{{color:#969dab}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:24px 0}}
.card{{background:#191d24;border:1px solid #2b313c;border-radius:12px;padding:16px}}.value{{font-size:25px;font-weight:700;margin-top:7px}}table{{width:100%;border-collapse:collapse;background:#191d24;border-radius:12px;overflow:hidden}}
th,td{{padding:11px 12px;border-bottom:1px solid #2b313c;text-align:left;vertical-align:top}}th{{background:#222833}}small{{display:block;margin-top:6px}}.ok{{color:#6ee7a0}}.bad{{color:#ff7b86}}
</style><main>
<h1>MyTranslator 实时全链路报告</h1><div class="muted">{html.escape(report['created_at'])} · {html.escape(report['suite'])}</div>
<section class="cards">
<div class="card">中文 CER<div class="value">{fmt_pct(summary['zh_cer'])}</div></div>
<div class="card">翻译 chrF<div class="value">{fmt_pct(summary['translation_chrf'])}</div></div>
<div class="card">首版译文 P50 / P95<div class="value">{fmt_ms(summary['initial_translation_latency_ms']['p50'])} / {fmt_ms(summary['initial_translation_latency_ms']['p95'])}</div></div>
<div class="card">译文延迟 P50 / P95<div class="value">{fmt_ms(summary['translation_latency_ms']['p50'])} / {fmt_ms(summary['translation_latency_ms']['p95'])}</div></div>
<div class="card">后台修正更新<div class="value">{summary['translation_revision_updates']}</div></div>
<div class="card">Qwen 复核成功 / 回退<div class="value">{summary['qwen_review_successes']} / {summary['qwen_review_fallbacks']}</div></div>
<div class="card">跨说话人合句<div class="value {'ok' if violation == '通过' else 'bad'}">{summary['cross_speaker_merge_violations']} · {violation}</div></div>
<div class="card">换人检测<div class="value">{summary['detected_speaker_changes']} / {summary['expected_speaker_changes']}</div></div>
<div class="card">队列峰值 ASR / 翻译<div class="value">{summary['asr_queue_peak']} / {summary['translation_queue_peak']}</div></div>
</section>
{delta_section}
<h2>逐组对齐结果</h2><table><thead><tr><th>参考 ID</th><th>参考原文 / 实际识别</th><th>参考译文 / 实际译文</th><th>CER</th><th>chrF</th><th>延迟</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</main></html>"""


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    suite_path = args.suite.resolve()
    cases = load_suite(suite_path, root)
    cases = cases[args.offset :]
    if not cases:
        raise ValueError(f"--offset {args.offset} skips the entire suite")
    if args.limit is not None:
        cases = cases[: args.limit]
    if not args.app_binary.exists():
        raise FileNotFoundError(f"build the release app first: {args.app_binary}")
    if port_open(args.ws_port):
        raise RuntimeError(f"port {args.ws_port} is already in use; choose another --ws-port")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "-", args.tag).strip("-")
    run_dir = args.output_dir.resolve() / f"{timestamp}-{safe_tag or 'live'}"
    run_dir.mkdir(parents=True, exist_ok=False)
    replay_path = run_dir / "replay.wav"
    timeline = make_replay(
        cases, replay_path, args.leading_silence_ms, args.trailing_silence_ms,
        args.default_gap_ms,
    )
    (run_dir / "timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    log_path = Path("/tmp/myt-voice-test.log")
    log_offset = log_path.stat().st_size if log_path.exists() else 0
    effective_vad = args.vad or ("steady" if args.profile == "quality" else "fast")
    env = os.environ.copy()
    env.update(
        {
            "MYT_BENCHMARK_MODE": "1",
            "MYT_BENCHMARK_INPUT": "system",
            "MYT_BENCHMARK_PROFILE": args.profile,
            "MYT_ASR_LANG": args.asr_lang,
            "MYT_AUTO_VOICE": "1",
            "MYT_WS_BIND": f"127.0.0.1:{args.ws_port}",
            "MYT_TRANSCRIPT_DIR": str(run_dir / "transcripts"),
        }
    )
    if args.vad:
        env["MYT_BENCHMARK_VAD"] = args.vad
    process = subprocess.Popen(
        [str(args.app_binary.resolve()), "--benchmark"],
        cwd=root,
        env=env,
        stdout=(run_dir / "app.stdout.log").open("wb"),
        stderr=(run_dir / "app.stderr.log").open("wb"),
        start_new_session=True,
    )
    collector: EventCollector | None = None
    try:
        wait_for_port(args.ws_port, process)
        collector = EventCollector("127.0.0.1", args.ws_port)
        collector.start()
        if not collector.connected.wait(timeout=10) or collector.error:
            raise RuntimeError(collector.error or "could not connect to subtitle WebSocket")
        wait_for_log(log_path, log_offset, "[audio] source=system", process, 35)

        collector.playback_start = time.monotonic()
        player = subprocess.Popen(["afplay", str(replay_path)])
        if player.wait() != 0:
            raise RuntimeError("afplay failed")

        deadline = time.monotonic() + args.drain_timeout
        last_count = -1
        stable_since = time.monotonic()
        while time.monotonic() < deadline:
            count = sum(event.get("type") == "translated" for event in collector.events)
            if count != last_count:
                last_count = count
                stable_since = time.monotonic()
            if count > 0 and time.monotonic() - stable_since >= 5.0:
                break
            if process.poll() is not None:
                break
            time.sleep(0.2)
    finally:
        if collector is not None:
            collector.stop()
            collector.join(timeout=3)
        terminate_group(process)

    events = [] if collector is None else collector.events
    for event in events:
        event.pop("monotonic", None)
    with log_path.open("rb") as handle:
        handle.seek(log_offset)
        log_text = handle.read().decode("utf-8", errors="replace")
    (run_dir / "app.pipeline.log").write_text(log_text, encoding="utf-8")
    summary, alignments = summarize(timeline, events, log_text)
    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "suite": str(suite_path),
        "settings": {
            "input": "system",
            "profile": args.profile,
            "vad": effective_vad,
            "asr_lang": args.asr_lang,
            "leading_silence_ms": args.leading_silence_ms,
            "trailing_silence_ms": args.trailing_silence_ms,
            "default_gap_ms": args.default_gap_ms,
            "offset": args.offset,
            "limit": args.limit,
        },
        "summary": summary,
        "baseline_delta": baseline_delta(summary, args.baseline.resolve() if args.baseline else None),
        "timeline": timeline,
        "alignments": alignments,
        "events": events,
        "collector_error": None if collector is None else collector.error,
    }
    json_path = run_dir / "report.json"
    html_path = run_dir / "report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(html_report(report), encoding="utf-8")
    if not args.keep_replay:
        replay_path.unlink(missing_ok=True)
    return json_path, html_path


def main() -> int:
    args = parse_args()
    json_path, html_path = run(args)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    summary = report["summary"]
    print(f"CER: {fmt_pct(summary['zh_cer'])}")
    print(f"translation chrF: {fmt_pct(summary['translation_chrf'])}")
    print(f"cross-speaker merge violations: {summary['cross_speaker_merge_violations']}")
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")
    return 0 if summary["cross_speaker_merge_violations"] == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        print(f"live benchmark failed: {error}", file=os.sys.stderr)
        raise SystemExit(2)
