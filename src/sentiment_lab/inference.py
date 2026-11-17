"""Инференс baseline и fine-tuned моделей."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .embeddings import get_embeddings


@dataclass(frozen=True)
class Prediction:
    text: str
    prediction: int
    probabilities: np.ndarray | None


def predict_fine_tuned(
    texts: str | list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
) -> list[Prediction]:
    """Предсказания дообученной модели классификации."""
    if isinstance(texts, str):
        texts = [texts]

    predictions = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        predictions.append(
            Prediction(
                text=text,
                prediction=int(pred),
                probabilities=probs[0].cpu().numpy(),
            )
        )
    return predictions


def predict_baseline(
    texts: str | list[str],
    model: LogisticRegression,
    tokenizer: AutoTokenizer,
    encoder_model: AutoModel,
    clean_func=None,
) -> list[Prediction]:
    """Предсказания baseline на CLS-эмбеддингах."""
    if isinstance(texts, str):
        texts = [texts]
    if clean_func:
        texts = [clean_func(text) for text in texts]

    X = get_embeddings(texts, tokenizer, encoder_model)
    preds = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    return [
        Prediction(
            text=text,
            prediction=int(preds[i]),
            probabilities=probs[i] if probs is not None else None,
        )
        for i, text in enumerate(texts)
    ]
