"""Collect fingerprint token IDs without importing model-training dependencies."""


def collect_trainable_input_ids(tokenizer, instructions, inputs):
    """Return token IDs from both dataset columns using the tokenizer's defaults.

    Materialize each column as a list for Hugging Face Datasets compatibility.
    Input-only tokens must be retained: ``set.union`` alone does not mutate the
    instruction-token set.
    """
    token_ids = set()
    for column in (instructions, inputs):
        encoded = tokenizer(list(column))["input_ids"]
        token_ids.update(token_id for row in encoded for token_id in row)
    return token_ids
