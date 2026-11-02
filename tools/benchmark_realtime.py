#!/usr/bin/env python3
"""Dependency-free benchmark for MyTranslator's resident ASR + MT pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


TRANSLATION_PROMPT = (
    "Translate the source text into natural {target}. Preserve names, numbers, "
    "dates, codes, and technical terms exactly; use established English spellings "
    "for well-known people and places. Output only the translation, with no labels "
    "or explanation."
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=root / "benchmarks/suites/smoke.jsonl")
    parser.add_argument("--runs", type=int, default=1, help="measured repetitions per sample")
    parser.add_argument("--warmups", type=int, default=1, help="unmeasured first-sample runs")
    parser.add_argument("--endpoint-ms", type=float, default=420.0, help="VAD tail budget")
    parser.add_argument("--asr-url", default="http://127.0.0.1:8342")
    parser.add_argument("--translation-url", default="http://127.0.0.1:8341")
    parser.add_argument("--output-dir", type=Path, default=root / "benchmarks/results")
    parser.add_argument("--glossary", type=Path, default=root / "config/default_glossary.json")
    parser.add_argument("--no-glossary", action="store_true", help="disable terminology protection")
    parser.add_argument("--tag", default="", help="report filename suffix")
    parser.add_argument("--no-start-servers", action="store_true")
    parser.add_argument("--keep-servers", action="store_true")
    parser.add_argument("--fail-on-target", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0:
        parser.error("--runs must be >= 1 and --warmups must be >= 0")
    return args


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> Any:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def healthy(base_url: str) -> bool:
    try:
        http_json(base_url.rstrip("/") + "/health", timeout=1.5)
        return True
    except Exception:
        return False


def wait_healthy(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if healthy(base_url):
            return
        time.sleep(0.1)
    raise RuntimeError(f"server did not become healthy: {base_url}")


class ServerManager:
    def __init__(self, root: Path, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.children: list[subprocess.Popen[bytes]] = []

    def _spawn(self, command: list[str], library_dir: Path) -> None:
        env = os.environ.copy()
        env["DYLD_LIBRARY_PATH"] = str(library_dir)
        child = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.children.append(child)

    def ensure(self) -> None:
        if not healthy(self.args.asr_url):
            if self.args.no_start_servers:
                raise RuntimeError(f"ASR server is unavailable: {self.args.asr_url}")
            binary = self.root / "bin/whisper-server-dist/whisper-server"
            model = self.root / "models/ggml-medium-q5_0.bin"
            self._require(binary, model)
            self._spawn(
                [str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", "8342",
                 "-t", "6", "-ac", "1280", "-nt", "-sns", "-nth", "0.65"],
                binary.parent,
            )
            wait_healthy(self.args.asr_url, 20.0)

        if not healthy(self.args.translation_url):
            if self.args.no_start_servers:
                raise RuntimeError(f"translation server is unavailable: {self.args.translation_url}")
            binary = self.root / "bin/llama-dist/llama-server"
            model = self.root / "models/Hy-MT2-1.8B-Q4_K_M.gguf"
            self._require(binary, model)
            self._spawn(
                [str(binary), "--model", str(model), "--host", "127.0.0.1", "--port", "8341",
                 "--no-webui", "--ctx-size", "4096"],
                binary.parent,
            )
            wait_healthy(self.args.translation_url, 75.0)

    @staticmethod
    def _require(*paths: Path) -> None:
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError("missing benchmark dependency: " + ", ".join(missing))

    def close(self) -> None:
        if self.args.keep_servers:
            return
        for child in reversed(self.children):
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)


class ZhNormalizer:
    """Reuse the same zhconv implementation as the application."""

    def __init__(self, root: Path) -> None:
        configured = os.environ.get("MYT_ZH_NORMALIZER")
        candidates = ([Path(configured)] if configured else []) + [
            root / "target/release/myt-zhconv",
            root / "target/debug/myt-zhconv",
        ]
        binary = next((path for path in candidates if path.exists()), None)
        self.process: subprocess.Popen[str] | None = None
        if binary is not None:
            self.process = subprocess.Popen(
                [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
            )

    def convert(self, text: str) -> str:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            return text
        self.process.stdin.write(text.replace("\n", " ") + "\n")
        self.process.stdin.flush()
        return self.process.stdout.readline().rstrip("\n")

    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)


def multipart_body(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
    boundary = "----mytranslator-benchmark-" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode("utf-8"),
            b"\r\n",
        ])
    chunks.extend([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode(),
        b"Content-Type: audio/wav\r\n\r\n",
        file_path.read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def transcribe(base_url: str, audio: Path, language: str) -> tuple[str, float]:
    body, content_type = multipart_body(
        {"language": language, "temperature": "0.0", "temperature_inc": "0.0", "response_format": "json"},
        audio,
    )
    req = urllib.request.Request(base_url.rstrip("/") + "/inference", data=body)
    req.add_header("Content-Type", content_type)
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=90.0) as response:
        value = json.loads(response.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return str(value.get("text", "")).strip(), elapsed_ms


def prepare_glossary(
    text: str, direction: str, glossary: list[dict[str, Any]]
) -> tuple[str, list[tuple[str, str]]]:
    source_key, target_key = ("zh", "en") if direction == "zh2en" else ("en", "zh")
    entries = [
        (str(entry.get(source_key, "")).strip(), str(entry.get(target_key, "")).strip())
        for entry in glossary if entry.get("enabled", True)
    ]
    entries = sorted((pair for pair in entries if all(pair)), key=lambda pair: len(pair[0]), reverse=True)
    prepared = text
    replacements: list[tuple[str, str]] = []
    for source, target in entries:
        marker = f"ZXQTERM{len(replacements):03d}QXZ"
        if direction == "en2zh" and source.isascii():
            replaced = re.sub(re.escape(source), marker, prepared, flags=re.IGNORECASE)
        else:
            replaced = prepared.replace(source, marker)
        if replaced != prepared:
            prepared = replaced
            replacements.append((marker, target))
    return prepared, replacements


def matching_glossary(
    text: str, direction: str, glossary: list[dict[str, Any]]
) -> list[tuple[str, str]]:
    source_key, target_key = ("zh", "en") if direction == "zh2en" else ("en", "zh")
    pairs = [
        (str(entry.get(source_key, "")).strip(), str(entry.get(target_key, "")).strip())
        for entry in glossary if entry.get("enabled", True)
    ]
    result = []
    for source, target in sorted((pair for pair in pairs if all(pair)), key=lambda pair: len(pair[0]), reverse=True):
        found = source.casefold() in text.casefold() if source.isascii() else source in text
        if found:
            result.append((source, target))
    return result


def restore_glossary(text: str, replacements: list[tuple[str, str]]) -> str:
    for marker, target in replacements:
        text = re.sub(re.escape(marker), lambda _: target, text, flags=re.IGNORECASE)
    return text


def translate(
    base_url: str, text: str, direction: str, glossary: list[dict[str, Any]]
) -> tuple[str, float, float | None, int]:
    target = "Chinese" if direction == "en2zh" else "English"
    terms = matching_glossary(text, direction, glossary)
    instruction = TRANSLATION_PROMPT.format(target=target)
    if terms:
        mappings = "; ".join(f"{source} = {target_term}" for source, target_term in terms)
        instruction += f" Required terminology: {mappings}. Use these exact target spellings."

    def request(system_prompt: str, user_text: str) -> tuple[dict[str, Any], float]:
        body = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": 256,
            "temperature": 0.1,
        }
        request_start = time.perf_counter()
        response = http_json(base_url.rstrip("/") + "/v1/chat/completions", body, timeout=90.0)
        return response, (time.perf_counter() - request_start) * 1000.0

    start = time.perf_counter()
    value, _ = request(instruction, text)
    output = str(value.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
    if not all(target_term.casefold() in output.casefold() for _, target_term in terms):
        request_text, replacements = prepare_glossary(text, direction, glossary)
        value, _ = request(TRANSLATION_PROMPT.format(target=target) + " Keep ZXQTERM tokens unchanged.", request_text)
        output = str(value.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        output = restore_glossary(output, replacements)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    timings = value.get("timings", {})
    predicted_n = float(timings.get("predicted_n", 0.0) or 0.0)
    predicted_ms = float(timings.get("predicted_ms", 0.0) or 0.0)
    tps = predicted_n / predicted_ms * 1000.0 if predicted_ms > 0 else None
    return output, elapsed_ms, tps, len(terms)


def normalize_chars(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [char for char in normalized if not char.isspace() and not unicodedata.category(char).startswith("P")]


def normalize_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return re.findall(r"[\w]+(?:'[\w]+)?", normalized, flags=re.UNICODE)


def edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row_index, right_item in enumerate(right, 1):
        current = [row_index]
        for col_index, left_item in enumerate(left, 1):
            current.append(min(
                current[-1] + 1,
                previous[col_index] + 1,
                previous[col_index - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def ngrams(chars: Sequence[str], size: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(chars[i:i + size]) for i in range(max(0, len(chars) - size + 1)))


def chrf(reference: str, hypothesis: str, beta: float = 2.0) -> float:
    ref = normalize_chars(reference)
    hyp = normalize_chars(hypothesis)
    if not ref or not hyp:
        return float(ref == hyp)
    precisions: list[float] = []
    recalls: list[float] = []
    for size in range(1, 7):
        ref_grams = ngrams(ref, size)
        hyp_grams = ngrams(hyp, size)
        overlap = sum((ref_grams & hyp_grams).values())
        precisions.append(overlap / max(1, sum(hyp_grams.values())))
        recalls.append(overlap / max(1, sum(ref_grams.values())))
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta * beta
    return (1.0 + beta2) * precision * recall / (beta2 * precision + recall)


def term_present(output: str, aliases: str | list[str]) -> bool:
    candidates = [aliases] if isinstance(aliases, str) else aliases
    haystack = "".join(normalize_chars(output))
    return any("".join(normalize_chars(candidate)) in haystack for candidate in candidates)


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_suite(path: Path, root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            case = json.loads(line)
            required = {"id", "audio", "source_lang", "direction", "transcript"}
            missing = required - case.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
            audio = Path(case["audio"])
            case["audio_path"] = str(audio if audio.is_absolute() else root / audio)
            if not Path(case["audio_path"]).exists():
                raise FileNotFoundError(case["audio_path"])
            cases.append(case)
    if not cases:
        raise ValueError(f"empty suite: {path}")
    return cases


def run_case(
    case: dict[str, Any], args: argparse.Namespace, glossary: list[dict[str, Any]],
    zh_normalizer: ZhNormalizer,
) -> dict[str, Any]:
    transcript, asr_ms = transcribe(args.asr_url, Path(case["audio_path"]), case["source_lang"])
    if case["source_lang"] == "zh":
        transcript = zh_normalizer.convert(transcript)
    translation, translation_ms, tps, glossary_terms = translate(
        args.translation_url, transcript, case["direction"], glossary
    )
    reference = case.get("translation_reference", "")
    terms = case.get("target_terms", [])
    term_results = [term_present(translation, term) for term in terms]
    ref_units = normalize_chars(case["transcript"]) if case["source_lang"] == "zh" else normalize_words(case["transcript"])
    hyp_units = normalize_chars(transcript) if case["source_lang"] == "zh" else normalize_words(transcript)
    return {
        "id": case["id"],
        "source_lang": case["source_lang"],
        "categories": case.get("categories", []),
        "audio": case["audio"],
        "reference_transcript": case["transcript"],
        "transcript": transcript,
        "asr_distance": edit_distance(ref_units, hyp_units),
        "asr_reference_units": len(ref_units),
        "asr_exact": ref_units == hyp_units,
        "translation_reference": reference,
        "translation": translation,
        "translation_chrf": chrf(reference, translation) if reference else None,
        "term_results": term_results,
        "term_hits": sum(term_results),
        "term_total": len(term_results),
        "glossary_terms": glossary_terms,
        "asr_ms": round(asr_ms, 2),
        "translation_ms": round(translation_ms, 2),
        "endpoint_ms": args.endpoint_ms,
        "pipeline_ms": round(args.endpoint_ms + asr_ms + translation_ms, 2),
        "tokens_per_second": round(tps, 2) if tps is not None else None,
    }


def aggregate(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    zh = [row for row in results if row["source_lang"] == "zh"]
    en = [row for row in results if row["source_lang"] == "en"]
    rate = lambda rows: sum(r["asr_distance"] for r in rows) / max(1, sum(r["asr_reference_units"] for r in rows))
    term_total = sum(r["term_total"] for r in results)
    summary = {
        "samples": len({row["id"] for row in results}),
        "measurements": len(results),
        "runs_per_sample": args.runs,
        "endpoint_ms": args.endpoint_ms,
        "asr_exact_rate": sum(r["asr_exact"] for r in results) / len(results),
        "zh_cer": rate(zh) if zh else None,
        "en_wer": rate(en) if en else None,
        "translation_chrf": sum(r["translation_chrf"] for r in results if r["translation_chrf"] is not None)
            / max(1, sum(r["translation_chrf"] is not None for r in results)),
        "term_accuracy": sum(r["term_hits"] for r in results) / term_total if term_total else None,
    }
    for name in ("asr_ms", "translation_ms", "pipeline_ms"):
        values = [float(row[name]) for row in results]
        summary[name] = {
            "mean": sum(values) / len(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }
    return summary


def target_status(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "zh_cer_lte_5pct": summary["zh_cer"] is None or summary["zh_cer"] <= 0.05,
        "en_wer_lte_6pct": summary["en_wer"] is None or summary["en_wer"] <= 0.06,
        "pipeline_p50_lte_1300ms": summary["pipeline_ms"]["p50"] <= 1300.0,
        "pipeline_p95_lte_1800ms": summary["pipeline_ms"]["p95"] <= 1800.0,
        "term_accuracy_gte_95pct": summary["term_accuracy"] is None or summary["term_accuracy"] >= 0.95,
    }


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    targets = report["targets"]
    lines = [
        "# MyTranslator 实时翻译基准报告",
        "",
        f"- 时间：{report['created_at']}",
        f"- 套件：`{report['suite']}`",
        f"- 样本：{summary['samples']} 条 × {summary['runs_per_sample']} 次",
        f"- VAD 尾静音预算：{summary['endpoint_ms']:.0f} ms",
        "",
        "## 总览",
        "",
        "| 指标 | 结果 | 目标 |",
        "|---|---:|---:|",
        f"| 中文 CER | {pct(summary['zh_cer'])} | ≤ 5% |",
        f"| 英文 WER | {pct(summary['en_wer'])} | ≤ 6% |",
        f"| ASR 完全一致率 | {pct(summary['asr_exact_rate'])} | — |",
        f"| 翻译 chrF | {pct(summary['translation_chrf'])} | 越高越好 |",
        f"| 关键术语保持率 | {pct(summary['term_accuracy'])} | ≥ 95% |",
        f"| ASR P50 / P95 | {summary['asr_ms']['p50']:.0f} / {summary['asr_ms']['p95']:.0f} ms | — |",
        f"| 翻译 P50 / P95 | {summary['translation_ms']['p50']:.0f} / {summary['translation_ms']['p95']:.0f} ms | — |",
        f"| 端到端 P50 / P95 | {summary['pipeline_ms']['p50']:.0f} / {summary['pipeline_ms']['p95']:.0f} ms | ≤ 1300 / 1800 ms |",
        "",
        "## 目标检查",
        "",
    ]
    lines.extend(f"- {'✅' if passed else '❌'} `{name}`" for name, passed in targets.items())
    lines.extend([
        "",
        "## 明细",
        "",
        "| ID | ASR | 译文 | ASR ms | 翻译 ms | 总计 ms | 术语 |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in report["results"]:
        asr = row["transcript"].replace("|", "\\|")
        translation = row["translation"].replace("|", "\\|")
        terms = f"{row['term_hits']}/{row['term_total']}" if row["term_total"] else "—"
        lines.append(
            f"| {row['id']} | {asr} | {translation} | {row['asr_ms']:.0f} | "
            f"{row['translation_ms']:.0f} | {row['pipeline_ms']:.0f} | {terms} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    suite_path = args.suite.resolve()
    cases = load_suite(suite_path, root)
    glossary_entries: list[dict[str, Any]] = []
    if not args.no_glossary:
        glossary_entries = json.loads(args.glossary.read_text(encoding="utf-8"))
        if not isinstance(glossary_entries, list):
            raise ValueError(f"glossary must be a JSON array: {args.glossary}")
    manager = ServerManager(root, args)
    zh_normalizer = ZhNormalizer(root)
    try:
        manager.ensure()

        def run_resilient(case: dict[str, Any]) -> dict[str, Any]:
            # A running desktop app may own the two ports when the benchmark
            # starts, then be closed midway. Recover by starting the missing
            # local engines and repeat only the interrupted measurement.
            try:
                return run_case(case, args, glossary_entries, zh_normalizer)
            except (OSError, urllib.error.URLError, urllib.error.HTTPError):
                if args.no_start_servers:
                    raise
                time.sleep(0.2)
                manager.ensure()
                return run_case(case, args, glossary_entries, zh_normalizer)

        if args.warmups:
            if not args.quiet:
                print(f"warming engines ({args.warmups} run)...", flush=True)
            for _ in range(args.warmups):
                run_resilient(cases[0])

        results: list[dict[str, Any]] = []
        for run_index in range(1, args.runs + 1):
            for case_index, case in enumerate(cases, 1):
                if not args.quiet:
                    print(f"[{run_index}/{args.runs}] [{case_index}/{len(cases)}] {case['id']}", flush=True)
                row = run_resilient(case)
                row["run"] = run_index
                results.append(row)

        summary = aggregate(results, args)
        report = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "suite": str(suite_path),
            "summary": summary,
            "targets": target_status(summary),
            "results": results,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = "-" + re.sub(r"[^a-zA-Z0-9_.-]+", "-", args.tag).strip("-") if args.tag else ""
        stem = f"benchmark-{timestamp}{suffix}"
        json_path = args.output_dir / f"{stem}.json"
        md_path = args.output_dir / f"{stem}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(markdown_report(report), encoding="utf-8")
        print(markdown_report(report))
        print(f"JSON: {json_path}")
        print(f"Markdown: {md_path}")
        return 1 if args.fail_on_target and not all(report["targets"].values()) else 0
    finally:
        manager.close()
        zh_normalizer.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
