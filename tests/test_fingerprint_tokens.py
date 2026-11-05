"""Offline regression tests for trainable fingerprint-token selection."""
import unittest

from utils.fingerprint_tokens import collect_trainable_input_ids


class ListOnlyTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def __call__(self, texts):
        if not isinstance(texts, list):
            raise TypeError("Tokenizer requires a concrete list of strings")
        self.calls.append(texts)
        return {"input_ids": [self.mapping[text] for text in texts]}


class DatasetColumn:
    """An iterable column which is deliberately not a Python list."""
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values)


class FingerprintTokenSelectionTests(unittest.TestCase):
    def test_input_only_tokens_are_trainable(self):
        tokenizer = ListOnlyTokenizer({"instruction": [1, 2], "input": [3, 99]})
        result = collect_trainable_input_ids(tokenizer, ["instruction"], ["input"])
        self.assertEqual(result, {1, 2, 3, 99})

    def test_dataset_columns_are_materialized_before_tokenization(self):
        tokenizer = ListOnlyTokenizer({"instruction": [4], "input": [8]})
        result = collect_trainable_input_ids(
            tokenizer, DatasetColumn(["instruction"]), DatasetColumn(["input"])
        )
        self.assertEqual(tokenizer.calls, [["instruction"], ["input"]])
        self.assertEqual(result, {4, 8})

    def test_shared_and_special_tokens_are_kept_once(self):
        tokenizer = ListOnlyTokenizer({"a": [0, 2, 2], "b": [0, 3], "": [0]})
        result = collect_trainable_input_ids(tokenizer, ["a", "b"], [""])
        self.assertEqual(result, {0, 2, 3})


if __name__ == "__main__":
    unittest.main()
