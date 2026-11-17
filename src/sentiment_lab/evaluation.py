"""Визуализация результатов классификации."""

from __future__ import annotations

from pathlib import Path

import seaborn as sns
from IPython import get_ipython
from IPython.display import Image, display
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

DEFAULT_OUTPUT_DIR = Path("outputs")


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
