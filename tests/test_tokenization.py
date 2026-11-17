from sentiment_lab.tokenization import (
    explain_tokenization,
    load_tokenizer,
    tokenize_texts,
)

MODEL_NAME = "hf-internal-testing/tiny-random-bert"


def test_tokenize_texts_returns_padded_torch_tensors() -> None:
    tokenizer = load_tokenizer(MODEL_NAME)
    texts = ["short", "a much longer piece of text"]

    batch = tokenize_texts(tokenizer, texts, max_length=32)

    lengths = [len(tokenizer(text)["input_ids"]) for text in texts]
    width = max(lengths)
    assert tuple(batch["input_ids"].shape) == (len(texts), width)
    assert batch["attention_mask"][0].tolist() == [1] * lengths[0] + [0] * (
        width - lengths[0]
    )
    assert batch["attention_mask"][1].tolist() == [1] * lengths[1]


def test_tokenize_texts_truncates_to_max_length() -> None:
    tokenizer = load_tokenizer(MODEL_NAME)

    batch = tokenize_texts(tokenizer, ["word " * 100], max_length=8)

    assert tuple(batch["input_ids"].shape) == (1, 8)
    assert batch["attention_mask"].tolist() == [[1] * 8]


def test_explain_tokenization_matches_tokenizer() -> None:
    tokenizer = load_tokenizer(MODEL_NAME)
    explanation = explain_tokenization("Transformers are amazing!", tokenizer)

    assert explanation.tokens == tokenizer.tokenize(explanation.text)
    assert explanation.token_ids == tokenizer.convert_tokens_to_ids(explanation.tokens)
    assert f"Количество: {len(explanation.tokens)}" in str(explanation)
    assert "transformers" in tokenizer.decode(explanation.token_ids).lower()
