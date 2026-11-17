"""Визуализация результатов классификации."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import seaborn as sns
from IPython import get_ipython
from IPython.display import Image, display
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_COMPARISON_PATH = Path("outputs/comparison_results.txt")


@dataclass(frozen=True)
class ModelMetrics:
    accuracy: float
    macro_f1: float


def compute_metrics(y_true, y_pred) -> ModelMetrics:
    return ModelMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro"),
    )


def save_comparison_results(
    fine_tuned: ModelMetrics,
    baseline: ModelMetrics | None,
    path: Path | str = DEFAULT_COMPARISON_PATH,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    improvement = None
    if baseline is not None and baseline.macro_f1:
        improvement = (fine_tuned.macro_f1 - baseline.macro_f1) / baseline.macro_f1 * 100

    with path.open("w", encoding="utf-8") as result:
        result.write("=== Сравнение моделей ===\n\n")
        result.write("Fine-tuned Model:\n")
        result.write(f"  F1 (macro): {fine_tuned.macro_f1:.4f}\n")
        result.write(f"  Accuracy: {fine_tuned.accuracy:.4f}\n")
        if baseline is not None:
            result.write("\nBaseline Model:\n")
            result.write(f"  F1 (macro): {baseline.macro_f1:.4f}\n")
            result.write(f"  Accuracy: {baseline.accuracy:.4f}\n")
            if improvement is not None:
                result.write(f"\nУлучшение F1: {improvement:.2f}%\n")
    return path


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[int],
    name: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Сохранить confusion matrix и показать её в Jupyter."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig = Figure(figsize=(6, 5))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(name)
    fig.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_confusion_matrix.png"
    fig.savefig(path)
    if get_ipython() is not None:
        display(Image(filename=str(path)))
    return path
