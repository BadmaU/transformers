"""Обучение и оценка модели классификации тональности."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_MODEL_DIR = Path("models/fine_tuned_model")
DEFAULT_RESULTS_PATH = Path("outputs/fine_tuned_results.txt")


@dataclass(frozen=True)
class TrainingResult:
    val_accuracy: float
    val_f1: float
    history: list[dict[str, float]]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    predictions: list[int] = []
    true_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["labels"].numpy())

    return (
        accuracy_score(true_labels, predictions),
        f1_score(true_labels, predictions, average="macro"),
    )


def train_model(
    model: AutoModelForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    optimizer: AdamW | None = None,
) -> TrainingResult:
    device = get_device()
    model.to(device)
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=learning_rate)

    history: list[dict[str, float]] = []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_accuracy, val_f1 = evaluate(model, val_loader, device)
        history.append(
            {
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1,
            }
        )
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        print(f"Val F1: {val_f1:.4f}")
        print("-" * 50)

    return TrainingResult(
        val_accuracy=history[-1]["val_accuracy"],
        val_f1=history[-1]["val_f1"],
        history=history,
    )


def save_finetuned(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    result: TrainingResult,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    results_path: Path | str = DEFAULT_RESULTS_PATH,
) -> tuple[Path, Path]:
    """Сохранить модель, токенизатор и итоговые метрики."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        f"Final Validation F1: {result.val_f1:.4f}\n"
        f"Final Validation Accuracy: {result.val_accuracy:.4f}\n",
        encoding="utf-8",
    )
    return model_dir, results_path
