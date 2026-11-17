"""FastAPI-демо для анализа тональности."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("models/fine_tuned_model")
LABEL_MAP = {0: "Negative", 1: "Positive"}

app = FastAPI(
    title="Sentiment Analysis с BERT",
    description="Введите текст, и модель определит его тональность",
)


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    prediction: str
    label: int
    probabilities: dict[str, float]


@lru_cache(maxsize=1)
def load_model() -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return model, tokenizer


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model, tokenizer = load_model()
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    label = int(torch.argmax(probabilities).item())
    return PredictionResponse(
        prediction=LABEL_MAP.get(label, str(label)),
        label=label,
        probabilities={
            LABEL_MAP.get(i, str(i)): float(probabilities[i])
            for i in range(len(probabilities))
        },
    )
