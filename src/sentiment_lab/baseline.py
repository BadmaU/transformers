"""Baseline-классификатор: логистическая регрессия на CLS-эмбеддингах."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

DEFAULT_RESULTS_PATH = Path("outputs/baseline_results.txt")
DEFAULT_MODEL_PATH = Path("models/baseline_model.pkl")


@dataclass(frozen=True)
class BaselineResult:
    model: LogisticRegression
    y_test: np.ndarray
    y_pred: np.ndarray
    report: str
    macro_f1: float


def train_baseline(
    embeddings: np.ndarray,
    labels,
    test_size: float = 0.2,
    random_state: int = 42,
) -> BaselineResult:
    """Обучить baseline на эмбеддингах и вернуть предсказания с метриками."""
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return BaselineResult(
        model=model,
        y_test=y_test,
        y_pred=y_pred,
        report=classification_report(y_test, y_pred),
        macro_f1=float(f1_score(y_test, y_pred, average="macro")),
    )


def save_model(
    model: LogisticRegression,
    path: Path | str = DEFAULT_MODEL_PATH,
) -> Path:
    """Сохранить обученный baseline с помощью joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path | str = DEFAULT_MODEL_PATH) -> LogisticRegression:
    return joblib.load(path)


def save_results(
    result: BaselineResult,
    path: Path | str = DEFAULT_RESULTS_PATH,
) -> Path:
    """Сохранить macro F1 и отчёт о классификации."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"Macro F1: {result.macro_f1:.4f}\n\n{result.report}",
        encoding="utf-8",
    )
    return path
