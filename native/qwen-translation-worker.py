#!/usr/bin/env python3
"""Persistent JSONL worker for Qwen3.5 translation and optional review.

The worker accepts two request modes:

* ``translate`` performs the foreground translation directly.
* ``review`` conservatively checks an existing foreground draft.

Keeping both operations in one resident process avoids reloading the same MLX
weights when the user changes translation strategy.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import select
import sys
import time
import traceback
from typing import Any


def emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("reviewer did not return a JSON object")
    value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("reviewer JSON is not an object")
    return value


def context_text_from_request(request: dict[str, Any]) -> str:
    context = request.get("context") or []
    context_lines: list[str] = []
    if isinstance(context, list):
        for entry in context[-2:]:
            if not isinstance(entry, dict):
                continue
            previous_source = str(entry.get("source") or "").strip()
            previous_translation = str(entry.get("translation") or "").strip()
            if previous_source and previous_translation:
                context_lines.append(
                    f"- source: {previous_source}\n  translation: {previous_translation}"
                )
    return "\n".join(context_lines) if context_lines else "(none)"


def glossary_text_from_request(request: dict[str, Any]) -> str:
    glossary = request.get("glossary") or []
    mappings: list[str] = []
    if isinstance(glossary, list):
        for entry in glossary:
            if not isinstance(entry, dict):
                continue
            source = str(entry.get("source") or "").strip()
            target = str(entry.get("target") or "").strip()
            if source and target:
                mappings.append(f"- {source} = {target}")
    return "\n".join(mappings) if mappings else "(none)"


def build_translate_prompt(request: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    source = str(request.get("source") or "").strip()
    direction = str(request.get("direction") or "")
    source_code = str(request.get("source_code") or "").strip()
    target = str(request.get("target_language") or "").strip()
    if not target:
        target = "Chinese" if direction == "en2zh" else "English"
    context_text = context_text_from_request(request)
    glossary_text = glossary_text_from_request(request)
    system = (
        f"You are a precise real-time translation engine. Translate only the current source into {target}. "
        "Use previous completed translation pairs only to resolve pronouns, ellipsis, names, and consistent "
        "terminology. Never repeat, summarize, or add facts from previous context. Preserve meaning, negation, "
        "numbers, dates, units, names, codes, and register. Follow every required terminology mapping exactly. "
        "Tokens matching ZXQTERM followed by digits and QXZ are protected placeholders and must be copied "
        "unchanged. Return exactly one JSON object with one string key named translation. Do not explain."
    )
    user = (
        f"Detected source language code: {source_code or '(unknown)'}\n"
        f"Previous completed translation pairs:\n{context_text}\n\n"
        f"Required terminology:\n{glossary_text}\n\n"
        f"Current source:\n{source}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], user


def build_review_prompt(request: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    source = str(request.get("source") or "").strip()
    draft = str(request.get("draft") or "").strip()
    direction = str(request.get("direction") or "")
    context_text = context_text_from_request(request)
    target = str(request.get("target_language") or "").strip()
    source_code = str(request.get("source_code") or "").strip()
    if not target:
        target = "Chinese" if direction == "en2zh" else "English"
    risks = request.get("risks") or []
    risk_text = ", ".join(str(risk) for risk in risks) if isinstance(risks, list) else ""
    system = (
        "You are a conservative translation reviewer. The foreground draft is usually correct. "
        "Change it only when the source or supplied dialogue context proves a material error in "
        "meaning, pronoun reference, omission, negation, number, name, or terminology. Never add "
        "facts, explanations, style embellishment, or text from prior context. Preserve the source "
        "gender exactly: Chinese 他/他的 maps to he/his and 她/她的 maps to she/her; likewise "
        "preserve explicit English gender when translating into Chinese. "
        "Preserve the source register and translate only the current source. Return exactly one JSON object with keys "
        'translation, change_needed, confidence, and reason. reason must be one of "context", '
        '"number", "negation", "terminology", "omission", or "none". confidence is 0 to 1.'
        " If change_needed is true, reason must not be none. If there is no material error, copy "
        "the draft exactly, set change_needed to false, confidence to 1, and reason to none."
    )
    user = (
        f"Detected source language code: {source_code or '(unknown)'}\n"
        f"Target language: {target}\n"
        f"Automated risk signals: {risk_text or '(none)'}\n"
        "Check the signaled risks first, but still make no change unless the source or context "
        "proves that the draft is materially wrong.\n"
        f"Previous completed translation pairs:\n{context_text}\n\n"
        f"Current source:\n{source}\n\n"
        f"Foreground draft:\n{draft}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], user


def build_prompt(request: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    if str(request.get("mode") or "review") == "translate":
        return build_translate_prompt(request)
    return build_review_prompt(request)


def numeric_tokens(text: str) -> set[str]:
    return {token.lstrip("0") or "0" for token in re.findall(r"\d+(?:\.\d+)?", text)}


def explicit_gender(text: str, language: str) -> str | None:
    if language == "zh":
        if ("她" in text or "她的" in text) and "她们" not in text:
            return "female"
        if ("他" in text or "他的" in text) and "他们" not in text:
            return "male"
        return None
    words = {word.lower() for word in re.findall(r"[A-Za-z']+", text)}
    if words & {"she", "her", "hers"}:
        return "female"
    if words & {"he", "him", "his"}:
        return "male"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--idle-seconds", type=float, default=75.0)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    started = time.monotonic()
    try:
        import mlx.core as mx
        from mlx_vlm import generate, load

        mx.set_cache_limit(768 * 1024 * 1024)
        model, processor = load(args.model)
    except Exception as exc:  # pragma: no cover - bundle smoke test covers this path
        emit({"event": "fatal", "error": str(exc), "trace": traceback.format_exc(limit=8)})
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
            messages, fallback_prompt = build_prompt(request)
            if hasattr(processor, "apply_chat_template"):
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = fallback_prompt

            inference_started = time.monotonic()
            result = generate(
                model,
                processor,
                prompt,
                max_tokens=256,
                verbose=False,
            )
            mx.synchronize()
            raw = result if isinstance(result, str) else getattr(result, "text", str(result))
            reviewed = extract_json(raw)
            translation = str(reviewed.get("translation") or "").strip()
            if not translation:
                raise ValueError("Qwen3.5 returned an empty translation")
            if str(request.get("mode") or "review") == "translate":
                emit(
                    {
                        "id": request_id,
                        "ok": True,
                        "translation": translation,
                        "change_needed": False,
                        "confidence": 1.0,
                        "reason": "direct",
                        "elapsed_ms": round((time.monotonic() - inference_started) * 1000),
                    }
                )
                mx.clear_cache()
                continue
            reason = str(reviewed.get("reason") or "none").strip().lower()
            if reason not in {"context", "number", "negation", "terminology", "omission", "none"}:
                reason = "none"
            confidence = max(0.0, min(1.0, float(reviewed.get("confidence") or 0.0)))
            change_needed = bool(reviewed.get("change_needed", False))
            source_numbers = numeric_tokens(str(request.get("source") or ""))
            draft_numbers = numeric_tokens(str(request.get("draft") or ""))
            candidate_numbers = numeric_tokens(translation)
            if (
                not change_needed
                and source_numbers - draft_numbers
                and source_numbers <= candidate_numbers
            ):
                # The candidate demonstrably restores source digits omitted by
                # the draft. Rust performs the same preservation check before
                # it is allowed to replace visible text.
                change_needed = True
                reason = "number"
                confidence = max(confidence, 0.9)
            target_code = str(request.get("target_code") or "")
            direction = str(request.get("direction") or "")
            source_language = str(request.get("source_code") or "")
            if not source_language:
                source_language = (
                    "en" if direction == "en2zh" else "zh" if direction == "zh2en" else ""
                )
            target_language = target_code if target_code in {"zh", "en"} else ""
            source_gender = (
                explicit_gender(str(request.get("source") or ""), source_language)
                if source_language and target_language
                else None
            )
            draft_gender = (
                explicit_gender(str(request.get("draft") or ""), target_language)
                if target_language
                else None
            )
            candidate_gender = (
                explicit_gender(translation, target_language) if target_language else None
            )
            if (
                not change_needed
                and source_gender is not None
                and candidate_gender == source_gender
                and draft_gender != source_gender
            ):
                change_needed = True
                reason = "context"
                confidence = max(confidence, 0.95)
            # Make the contract deterministic even if the model returns a
            # rewritten string while claiming that no correction is needed.
            if not change_needed or reason == "none":
                translation = str(request.get("draft") or "").strip()
                change_needed = False
                reason = "none"
            emit(
                {
                    "id": request_id,
                    "ok": True,
                    "translation": translation,
                    "change_needed": change_needed,
                    "confidence": confidence,
                    "reason": reason,
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
    multiprocessing.freeze_support()
    raise SystemExit(main())
