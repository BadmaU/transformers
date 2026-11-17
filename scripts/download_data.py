"""Скачать датасет и сохранить его в CSV.

Примеры:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --max-rows 5000
    uv run python scripts/download_data.py --dataset tweet_eval --config sentiment
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# у разных датасетов текстовая колонка называется по-разному
TEXT_COLUMN_ALIASES = ("text", "sentence", "review", "content", "tweet")

# datasets 5.x требует полный id вида "namespace/name".
# Короткие имена оставляем для удобства и разворачиваем в канонические.
DATASET_ALIASES = {
    "imdb": "stanfordnlp/imdb",
    "sst2": "stanfordnlp/sst2",
    "tweet_eval": "cardiffnlp/tweet_eval",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="imdb",
        help=(
            "id датасета на Hugging Face, например stanfordnlp/imdb; "
            "короткие имена imdb, sst2, tweet_eval поддерживаются "
            "(по умолчанию: imdb)"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="конфигурация датасета, если нужна (например, sentiment у tweet_eval)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="ограничить число строк в каждом сплите (удобно для CPU)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR,
        help="куда сохранять CSV (по умолчанию: data/)",
    )
    return parser.parse_args()


def normalize_columns(split: Dataset) -> Dataset:
    """Привести датасет к формату text,label."""
    if "label" not in split.column_names:
        raise ValueError(
            f"В датасете нет колонки 'label': {split.column_names}"
        )

    if "text" not in split.column_names:
        for alias in TEXT_COLUMN_ALIASES:
            if alias in split.column_names:
                split = split.rename_column(alias, "text")
                break
        else:
            raise ValueError(
                "Не нашёл текстовую колонку. "
                f"Доступные колонки: {split.column_names}"
            )

    return split.select_columns(["text", "label"])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = DATASET_ALIASES.get(args.dataset, args.dataset)
    dataset = load_dataset(dataset_id, args.config)
    print(f"Загружен {dataset_id}: {dataset}")

    slug = args.dataset.replace("/", "_")
    if args.config:
        slug = f"{slug}_{args.config}"

    for split_name, split in dataset.items():
        if args.max_rows is not None:
            split = split.select(range(min(args.max_rows, len(split))))

        split = normalize_columns(split)
        out_path = args.out_dir / f"{slug}_{split_name}.csv"
        split.to_csv(out_path)
        print(f"{split_name}: {len(split):,} строк -> {out_path}")

    print("Готово.")


if __name__ == "__main__":
    main()
