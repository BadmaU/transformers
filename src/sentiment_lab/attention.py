"""Получение и визуализация attention-весов трансформера."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from IPython import get_ipython
from IPython.display import Image, display
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from .tokenization import DEFAULT_MODEL_NAME

DEFAULT_OUTPUT_DIR = Path("outputs")


def load_attention_model(model_name: str = DEFAULT_MODEL_NAME) -> AutoModel:
    """Загрузить модель с включённым выводом attention."""
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    return model


def get_attention(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
) -> tuple[BatchEncoding, tuple[torch.Tensor, ...]]:
    """Прогнать текст и вернуть токены с весами attention по слоям."""
    tokens = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    return tokens, outputs.attentions


def visualize_attention(
    tokens: BatchEncoding,
    attention: tuple[torch.Tensor, ...],
    tokenizer: AutoTokenizer,
    layer: int = 0,
    head: int = 0,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Сохранить heatmap attention для заданного слоя и головы."""
    attn = attention[layer][0, head]
    token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    fig = Figure(figsize=(10, 8))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    sns.heatmap(
        attn.cpu().numpy(),
        xticklabels=token_list,
        yticklabels=token_list,
        cmap="viridis",
        cbar=True,
        ax=ax,
    )
    ax.set_title(f"Attention — Layer {layer}, Head {head}")
    ax.set_xlabel("Keys")
    ax.set_ylabel("Queries")
    fig.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"attention_layer{layer}_head{head}.png"
    fig.savefig(path)
    if get_ipython() is not None:
        display(Image(filename=str(path)))
    return path


def attention_for_token(
    tokens: BatchEncoding,
    attention: tuple[torch.Tensor, ...],
    tokenizer: AutoTokenizer,
    query: str,
    layer: int,
    head: int,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Вернуть токены, на которые смотрит query, в порядке убывания веса."""
    token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    query_index = token_list.index(query)
    weights = attention[layer][0, head, query_index].cpu().numpy()
    top_indices = np.argsort(weights)[::-1][:top_k]
    return [(token_list[i], float(weights[i])) for i in top_indices]
