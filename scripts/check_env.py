"""Проверка окружения: версии библиотек + прогон крошечной модели.

Запуск:
    uv run python scripts/check_env.py
"""

from __future__ import annotations

import platform

import torch
import transformers
from transformers import AutoModel, AutoTokenizer


def print_versions() -> None:
    print(f"Python:       {platform.python_version()} ({platform.platform()})")
    print(f"torch:        {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"CUDA:         {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:          {torch.cuda.get_device_name(0)}")


def run_tiny_model(model_name: str = "prajjwal1/bert-tiny") -> None:
    print(f"\nЗагружаю {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    text = "This movie was absolutely amazing!"
    batch = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True)

    print(f"input_ids:     {tuple(batch['input_ids'].shape)}")
    print(f"last_hidden:   {tuple(outputs.last_hidden_state.shape)}")
    print(f"hidden_states: {len(outputs.hidden_states)} слоёв")
    print("\nОкружение готово.")


def main() -> None:
    print_versions()
    try:
        run_tiny_model()
    except Exception as exc:  # noqa: BLE001 - smoke-тест должен показать любую ошибку
        print(f"\nНе удалось прогнать модель: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
