from __future__ import annotations

from typing import Any, Iterable


OMISSION_MARKER = "\n...[middle of long prompt omitted]...\n"


def _slice_sequence(value: Any, head: int, tail: int, marker: list[Any]) -> Any:
    """Slice a parallel sequence (input_ids, token_type_ids, etc.) consistently."""
    return list(value[:head]) + list(marker) + list(value[-tail:])


def encode_head_tail(
    tokenizer: Any,
    text: str,
    *,
    max_length: int,
    input_prefix: str = "",
) -> dict[str, list[int]]:
    """Tokenize a prompt while preserving both its request and trailing context."""
    encoded = tokenizer(
        input_prefix + str(text),
        add_special_tokens=True,
        truncation=False,
    )
    input_ids = list(encoded["input_ids"])
    budget = max(2, int(max_length))
    if len(input_ids) <= budget:
        result = {
            key: list(value)
            for key, value in encoded.items()
            if isinstance(value, (list, tuple)) and len(value) == len(input_ids)
        }
        result.setdefault("attention_mask", [1] * len(input_ids))
        return result
    marker_ids = list(
        tokenizer(OMISSION_MARKER, add_special_tokens=False, truncation=False)["input_ids"]
    )
    if len(marker_ids) >= budget:
        marker_ids = []
    remaining = budget - len(marker_ids)
    # User instructions and explicit output constraints often sit at the
    # end, so retain slightly more tail than head.
    head = remaining * 45 // 100
    tail = remaining - head
    result: dict[str, list[int]] = {}
    for key, value in encoded.items():
        if isinstance(value, (list, tuple)) and len(value) == len(input_ids):
            result[key] = _slice_sequence(value, head, tail, marker_ids)
    if "attention_mask" not in result:
        result["attention_mask"] = [1] * len(result.get("input_ids", []))
    return result


def encode_head_tail_batch(
    tokenizer: Any,
    texts: Iterable[str],
    *,
    max_length: int,
    input_prefix: str = "",
    padding: bool = False,
    return_tensors: str | None = None,
) -> Any:
    values = [input_prefix + str(text) for text in texts]
    encoded_batch = tokenizer(
        values,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        verbose=False,
    )
    marker_ids = list(
        tokenizer(
            OMISSION_MARKER,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
    )
    features: list[dict[str, list[int]]] = []
    for index, ids_value in enumerate(encoded_batch["input_ids"]):
        input_ids = list(ids_value)
        feature: dict[str, list[int]] = {}
        if len(input_ids) <= max_length:
            for key, batch_values in encoded_batch.items():
                if (
                    isinstance(batch_values, (list, tuple))
                    and index < len(batch_values)
                    and isinstance(batch_values[index], (list, tuple))
                    and len(batch_values[index]) == len(input_ids)
                ):
                    feature[key] = list(batch_values[index])
            feature.setdefault("attention_mask", [1] * len(input_ids))
        else:
            marker = marker_ids if len(marker_ids) < max_length else []
            remaining = max_length - len(marker)
            head = remaining * 45 // 100
            tail = remaining - head
            for key, batch_values in encoded_batch.items():
                if (
                    isinstance(batch_values, (list, tuple))
                    and index < len(batch_values)
                    and isinstance(batch_values[index], (list, tuple))
                    and len(batch_values[index]) == len(input_ids)
                ):
                    feature[key] = _slice_sequence(batch_values[index], head, tail, marker)
            feature.setdefault("attention_mask", [1] * len(feature.get("input_ids", [])))
        features.append(feature)
    return tokenizer.pad(features, padding=padding, return_tensors=return_tensors)
