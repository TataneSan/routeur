from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routeur.capabilities import CAPABILITIES, RISKS, normalize_capabilities
from routeur.tasks import TASKS


def _extract_json(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError(f"GLM returned no JSON array: {text[:300]}")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("GLM response is not an array")
    return [item for item in value if isinstance(item, dict)]


def annotate_batch(
    rows: list[dict[str, Any]],
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
) -> list[dict[str, Any]]:
    payload_rows = [
        {"id": idx, "prompt": str(row["prompt"])[:12000]}
        for idx, row in enumerate(rows)
    ]
    system = (
        "You are a strict, model-independent prompt-routing data annotator. Return ONLY valid JSON, "
        "an array with one object per input id. Use exactly these task labels: "
        + ", ".join(TASKS)
        + ". Choose one primary_task and zero or more genuinely relevant secondary_tasks. "
        "Use only these required_capabilities: "
        + ", ".join(CAPABILITIES)
        + ". Level is the minimum model "
        "power needed for a reliable answer: 1 trivial, 2 routine, 3 multi-step, "
        "4 hard technical/reasoning, 5 high-stakes or failure-sensitive. "
        "Judge the prompt itself, independently of any named vendor or model. Be conservative "
        "about under-routing. confidence must be 0..1."
    )
    user = (
        "Annotate these prompts. Output objects shaped as "
        "{id, primary_task, secondary_tasks, level, confidence, risk, required_capabilities}. "
        "risk is one of low, medium, high.\n"
        + json.dumps(payload_rows, ensure_ascii=False)
    )
    body = json.dumps(
        {
            "model": model,
            "temperature": 0.1,
            "max_tokens": max(1024, len(rows) * 220),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_body = json.loads(response.read().decode("utf-8"))
    content = response_body["choices"][0]["message"]["content"]
    annotations = _extract_json(content)
    by_id: dict[int, dict[str, Any]] = {}
    for annotation in annotations:
        try:
            idx = int(annotation["id"])
            task = str(annotation.get("primary_task", annotation.get("task", ""))).strip().lower()
            level = max(1, min(5, int(annotation["level"])))
            confidence = max(0.0, min(1.0, float(annotation.get("confidence", 0.7))))
            risk = str(annotation.get("risk", "medium")).lower()
            secondary = [str(value).strip().lower() for value in annotation.get("secondary_tasks", [])]
            secondary = sorted({value for value in secondary if value in TASKS and value != task})
            capabilities = normalize_capabilities(annotation.get("required_capabilities", []))
            if task not in TASKS or risk not in RISKS:
                continue
            by_id[idx] = {
                "task": task,
                "secondary_tasks": secondary,
                "level": level,
                "confidence": confidence,
                "risk": risk,
                "required_capabilities": capabilities,
            }
        except (KeyError, TypeError, ValueError):
            continue
    if len(by_id) < max(1, len(rows) * 0.8):
        raise ValueError(f"GLM annotated only {len(by_id)}/{len(rows)} rows")
    output: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        annotation = by_id.get(idx)
        if annotation is None:
            output.append(
                {
                    **row,
                    "source": f"annotation_missing:{row.get('source', 'unknown')}",
                    "metadata": {
                        **(row.get("metadata") or {}),
                        "annotation_missing": True,
                        "annotation_version": 2,
                    },
                }
            )
            continue
        output.append(
            {
                **row,
                "level": annotation["level"],
                "task": annotation["task"],
                "source": f"teacher:{model}:{row.get('source', 'unknown')}",
                "metadata": {
                    **(row.get("metadata") or {}),
                    "grader": model,
                    "grader_confidence": annotation["confidence"],
                    "weak_level": row.get("level"),
                    "weak_task": row.get("task"),
                    "risk": annotation["risk"],
                    "secondary_tasks": annotation["secondary_tasks"],
                    "required_capabilities": annotation["required_capabilities"],
                    "annotation_version": 2,
                },
            }
        )
    return output


def _is_quota_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    code = getattr(exc, "code", None)
    return code in {400, 402, 403, 409, 429} or any(
        marker in text
        for marker in ("quota", "rate limit", "token", "credit", "insufficient")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key", default=os.environ.get("EDELUX_API_KEY"))
    parser.add_argument("--base-url", default="https://edelux.contes.me/v1")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--fallback-model", default="Mistral:codestral-latest")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("EDELUX_API_KEY is required")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    already_done = 0
    if args.resume and args.output.exists():
        already_done = sum(1 for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip())
    rows = rows[already_done:]
    jobs = [(start, rows[start : start + args.batch_size]) for start in range(0, len(rows), args.batch_size)]

    def process(job: tuple[int, list[dict[str, Any]]]) -> tuple[int, list[dict[str, Any]], str]:
        start, batch = job
        active_model = args.model
        for attempt in range(args.retries):
            try:
                result = annotate_batch(
                    batch,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    model=active_model,
                    timeout=args.timeout,
                )
                return start, result, active_model
            except (OSError, ValueError, KeyError, urllib.error.HTTPError) as exc:
                if active_model != args.fallback_model and (_is_quota_error(exc) or isinstance(exc, ValueError)):
                    active_model = args.fallback_model
                    print(f"batch {start}: switching annotation model to {active_model}: {exc}", flush=True)
                    if attempt + 1 >= args.retries:
                        result = annotate_batch(
                            batch,
                            api_key=args.api_key,
                            base_url=args.base_url,
                            model=active_model,
                            timeout=args.timeout,
                        )
                        return start, result, active_model
                    continue
                if attempt + 1 >= args.retries:
                    if active_model != args.fallback_model:
                        active_model = args.fallback_model
                        print(f"batch {start}: final retry with {active_model}: {exc}", flush=True)
                        result = annotate_batch(
                            batch,
                            api_key=args.api_key,
                            base_url=args.base_url,
                            model=active_model,
                            timeout=args.timeout,
                        )
                        return start, result, active_model
                    raise
                print(f"retrying batch {start}: {exc}", flush=True)
                time.sleep(2**attempt)
        raise RuntimeError(f"unreachable annotation retry state for batch {start}")

    written = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for start, batch_rows, active_model in executor.map(process, jobs):
            with args.output.open("a", encoding="utf-8") as handle:
                for row in batch_rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            written += len(batch_rows)
            print(
                f"annotated {already_done + written}/{already_done + len(rows)} with {active_model}",
                flush=True,
            )
    print(f"wrote {already_done + written} annotated rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
