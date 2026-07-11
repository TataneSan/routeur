from routeur.tokenization import encode_head_tail, encode_head_tail_batch


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        ids = [ord(character) for character in text]
        if kwargs.get("add_special_tokens", True):
            ids = [1, *ids, 2]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class BertLikeTokenizer:
    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            encoded = [self._encode(item, **kwargs) for item in text]
            return {
                "input_ids": [item["input_ids"] for item in encoded],
                "attention_mask": [item["attention_mask"] for item in encoded],
                "token_type_ids": [item["token_type_ids"] for item in encoded],
            }
        return self._encode(text, **kwargs)

    def _encode(self, text, **kwargs):
        ids = [ord(character) for character in text]
        if kwargs.get("add_special_tokens", True):
            ids = [1, *ids, 2]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "token_type_ids": [0] * len(ids),
        }

    def pad(self, features, **kwargs):
        return {
            "input_ids": [feature["input_ids"] for feature in features],
            "attention_mask": [feature["attention_mask"] for feature in features],
            "token_type_ids": [feature.get("token_type_ids", []) for feature in features],
        }


def test_head_tail_tokenization_preserves_both_ends():
    text = "BEGIN-" + "x" * 200 + "-FINAL-INSTRUCTION"
    encoded = encode_head_tail(CharacterTokenizer(), text, max_length=80)
    decoded = "".join(chr(value) for value in encoded["input_ids"][1:-1])
    assert decoded.startswith("BEGIN-")
    assert decoded.endswith("-FINAL-INSTRUCTION")
    assert len(encoded["input_ids"]) <= 80


def test_head_tail_preserves_all_tokenizer_fields():
    text = "BEGIN-" + "x" * 200 + "-FINAL-INSTRUCTION"
    encoded = encode_head_tail(BertLikeTokenizer(), text, max_length=80)
    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert "token_type_ids" in encoded
    assert len(encoded["input_ids"]) == len(encoded["attention_mask"]) == len(encoded["token_type_ids"])


def test_head_tail_batch_preserves_all_tokenizer_fields():
    texts = ["BEGIN-" + "x" * 200 + "-FINAL", "short"]
    encoded = encode_head_tail_batch(BertLikeTokenizer(), texts, max_length=80)
    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert "token_type_ids" in encoded
    assert len(encoded["input_ids"]) == len(texts)
    assert all(len(ids) == len(types) for ids, types in zip(encoded["input_ids"], encoded["token_type_ids"]))
