"""Анализ ошибок бинарного классификатора тональности."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_REPORT_PATH = Path("outputs/error_analysis.txt")


@dataclass(frozen=True)
class ErrorAnalysis:
    frame: pd.DataFrame
    errors: pd.DataFrame
    false_positives: pd.DataFrame
    false_negatives: pd.DataFrame
    error_rate: float
    mean_error_length: float
    mean_all_length: float


def analyze_errors(texts, true_labels, pred_labels) -> ErrorAnalysis:
    frame = pd.DataFrame(
        {
            "text": list(texts),
            "true_label": list(true_labels),
            "pred_label": list(pred_labels),
        }
    )
    errors = frame.loc[frame["true_label"] != frame["pred_label"]].copy()
    errors["text_length"] = errors["text"].str.len()
    false_positives = errors.loc[
        (errors["pred_label"] == 1) & (errors["true_label"] == 0)
    ]
    false_negatives = errors.loc[
        (errors["pred_label"] == 0) & (errors["true_label"] == 1)
    ]

    return ErrorAnalysis(
        frame=frame,
        errors=errors,
        false_positives=false_positives,
        false_negatives=false_negatives,
        error_rate=len(errors) / len(frame),
        mean_error_length=float(errors["text_length"].mean())
        if not errors.empty
        else 0.0,
        mean_all_length=float(frame["text"].str.len().mean()),
    )


def save_error_report(
    analysis: ErrorAnalysis,
    path: Path | str = DEFAULT_REPORT_PATH,
    examples: int = 5,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fp_len = analysis.false_positives["text_length"].mean()
    fn_len = analysis.false_negatives["text_length"].mean()
    length_delta = analysis.mean_error_length - analysis.mean_all_length
    if analysis.mean_all_length:
        length_percent = length_delta / analysis.mean_all_length * 100
    else:
        length_percent = 0.0

    if len(analysis.false_positives) > len(analysis.false_negatives):
        bias = "Чаще модель ошибочно относят негатив к позитиву (FP)."
    elif len(analysis.false_negatives) > len(analysis.false_positives):
        bias = "Чаще модель пропускает позитив, называя его негативом (FN)."
    else:
        bias = "Ошибки FP и FN распределены поровну."

    if abs(length_percent) < 5:
        length_note = "Длина ошибочных текстов почти не отличается от средней."
    elif length_percent > 0:
        length_note = "Ошибочные тексты длиннее среднего."
    else:
        length_note = "Ошибочные тексты короче среднего."

    with path.open("w", encoding="utf-8") as report:
        report.write("=== АНАЛИЗ ОШИБОК ===\n\n")
        report.write(f"Всего ошибок: {len(analysis.errors)}\n")
        report.write(f"False Positives: {len(analysis.false_positives)}\n")
        report.write(f"False Negatives: {len(analysis.false_negatives)}\n")
        report.write(f"Доля ошибок: {analysis.error_rate:.2%}\n")
        report.write(f"\nСредняя длина ошибочных текстов: {analysis.mean_error_length:.0f}\n")
        report.write(f"Средняя длина всех текстов: {analysis.mean_all_length:.0f}\n\n")

        report.write("=== ПРИМЕРЫ FALSE POSITIVES ===\n")
        for _, row in analysis.false_positives.head(examples).iterrows():
            report.write(f"\nТекст: {row['text']}\n")
            report.write(
                f"Истинный: {row['true_label']}, Предсказан: {row['pred_label']}\n"
            )

        report.write("\n\n=== ПРИМЕРЫ FALSE NEGATIVES ===\n")
        for _, row in analysis.false_negatives.head(examples).iterrows():
            report.write(f"\nТекст: {row['text']}\n")
            report.write(
                f"Истинный: {row['true_label']}, Предсказан: {row['pred_label']}\n"
            )

        report.write("\n\n=== НАБЛЮДЕНИЯ ===\n")
        report.write(f"{bias}\n")
        report.write(f"{length_note} Отклонение: {length_percent:+.1f}%.\n")
        report.write(
            f"Средняя длина FP: {fp_len:.0f}, FN: {fn_len:.0f} "
            "(символов, до токенизации).\n"
        )
        report.write(
            "Часть ошибок может быть связана с иронией, смешанной тональностью "
            "и обрезанием текста до 128 токенов.\n"
        )
    return path
