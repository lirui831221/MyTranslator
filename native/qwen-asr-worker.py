#!/usr/bin/env python3
"""Persistent JSONL worker for MyTranslator's optional Qwen3-ASR review.

The process owns one MLX model instance, accepts one WAV request per input
line, and emits exactly one JSON response per request.  Keeping this outside
the Rust UI process prevents a background review from blocking the Whisper
caption path and lets macOS reclaim the model when the idle timer expires.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
import traceback


def emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def qwen_language(language: object) -> str | None:
    value = str(language or "").strip().lower()
    supported = {
        "zh": "Chinese",
        "en": "English",
        "yue": "Cantonese",
        "ar": "Arabic",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "pt": "Portuguese",
        "id": "Indonesian",
        "it": "Italian",
        "ko": "Korean",
        "ru": "Russian",
        "th": "Thai",
        "vi": "Vietnamese",
        "ja": "Japanese",
        "tr": "Turkish",
        "hi": "Hindi",
        "ms": "Malay",
        "nl": "Dutch",
        "sv": "Swedish",
        "da": "Danish",
        "fi": "Finnish",
        "pl": "Polish",
        "cs": "Czech",
        "tl": "Filipino",
        "fa": "Persian",
        "el": "Greek",
        "ro": "Romanian",
        "hu": "Hungarian",
        "mk": "Macedonian",
    }
    aliases = {
        "zh-cn": "zh",
        "zh-hans": "zh",
        "chinese": "zh",
        "en-us": "en",
        "english": "en",
    }
    value = aliases.get(value, value)
    if value in supported:
        return supported[value]
    # Qwen's own language detector is safer for auto and unknown locales.
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--idle-seconds", type=float, default=75.0)
    args = parser.parse_args()

    # Avoid online fallback: a release must work exclusively from the model
    # directory shipped inside MyTranslator.app.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    started = time.monotonic()
    try:
        import mlx.core as mx
        from mlx_qwen3_asr import Session

        # Bound reusable Metal buffers.  Active weights are unaffected, while
        # idle scratch memory is prevented from competing with translation.
        mx.set_cache_limit(768 * 1024 * 1024)
        session = Session(model=args.model)
    except Exception as exc:  # pragma: no cover - exercised by bundle smoke test
        emit(
            {
                "event": "fatal",
                "error": str(exc),
                "trace": traceback.format_exc(limit=8),
            }
        )
        return 2

    emit(
        {
            "event": "ready",
            "load_ms": round((time.monotonic() - started) * 1000),
            "model": args.model,
        }
    )

    while True:
        readable, _, _ = select.select([sys.stdin], [], [], max(1.0, args.idle_seconds))
        if not readable:
            emit({"event": "idle-exit"})
            return 0
        line = sys.stdin.readline()
        if not line:
            return 0

        request_id: object = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            wav_path = str(request["wav_path"])
            if not os.path.isfile(wav_path):
                raise FileNotFoundError(wav_path)

            inference_started = time.monotonic()
            result = session.transcribe(
                wav_path,
                language=qwen_language(request.get("language")),
                context=str(request.get("context") or "")[:512],
                max_new_tokens=256,
            )
            # Synchronize before reporting latency so the measurement includes
            # queued Metal work rather than only Python dispatch.
            mx.synchronize()
            emit(
                {
                    "id": request_id,
                    "ok": True,
                    "text": result.text,
                    "language": result.language,
                    "elapsed_ms": round((time.monotonic() - inference_started) * 1000),
                }
            )
            mx.clear_cache()
        except Exception as exc:
            emit(
                {
                    "id": request_id,
                    "ok": False,
                    "error": str(exc),
                    "trace": traceback.format_exc(limit=5),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
