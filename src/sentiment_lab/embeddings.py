"""Получение эмбеддингов из скрытых состояний (hidden states) модели."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from .tokenization import DEFAULT_MODEL_NAME, tokenize_texts


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> AutoModel:
    """Загрузить модель и перевести её в режим оценки."""
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model


def get_embeddings(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = 32,
) -> np.ndarray:
    """Вернуть CLS-эмбеддинг для каждого текста."""
    if not texts:
        return np.empty((0, 0))

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokens = tokenize_texts(tokenizer, batch_texts)
        with torch.no_grad():
            outputs = model(**tokens)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def similarity(
    text1: str,
    text2: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
) -> float:
    """Косинусное сходство CLS-эмбеддингов двух текстов."""
    emb = get_embeddings([text1, text2], tokenizer, model)
    return float(cosine_similarity(emb[0:1], emb[1:2])[0][0])
