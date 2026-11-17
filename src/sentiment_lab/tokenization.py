"""Переиспользуемая токенизация текстов для классификации тональности."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer, BatchEncoding

DEFAULT_MODEL_NAME = "distilbert-base-uncased"


@dataclass(frozen=True)
class TokenizationExplanation:
    text: str
    tokens: list[str]
    token_ids: list[int]

    def __str__(self) -> str:
        return (
            f"Исходный текст: {self.text}\n"
            f"Токены: {self.tokens}\n"
            f"IDs: {self.token_ids}\n"
            f"Количество: {len(self.tokens)}"
        )


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> AutoTokenizer:
    """Загрузить токенизатор из HF Hub или локального кэша."""
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_texts(
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int = 128,
) -> BatchEncoding:
    """Токенизировать батч с padding/truncation в torch-тензоры."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def explain_tokenization(
    text: str,
    tokenizer: AutoTokenizer,
) -> TokenizationExplanation:
    """Разобрать текст на токены и их ID."""
    tokens = tokenizer.tokenize(text)
    explanation = TokenizationExplanation(
        text=text,
        tokens=tokens,
        token_ids=tokenizer.convert_tokens_to_ids(tokens),
    )
    print(explanation)
    return explanation
