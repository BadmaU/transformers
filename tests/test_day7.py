from pathlib import Path

from fastapi.testclient import TestClient
from transformers import AutoModelForSequenceClassification

import api
from sentiment_lab.error_analysis import analyze_errors, save_error_report
from sentiment_lab.evaluation import compute_metrics, save_comparison_results
from sentiment_lab.tokenization import load_tokenizer

MODEL_NAME = "hf-internal-testing/tiny-random-bert"


def test_error_analysis_reports_fp_fn(tmp_path: Path) -> None:
    analysis = analyze_errors(
        ["good", "bad", "great", "awful"],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
    )

    assert len(analysis.errors) == 2
    assert len(analysis.false_positives) == 1
    assert len(analysis.false_negatives) == 1

    path = save_error_report(analysis, tmp_path / "error_analysis.txt")
    content = path.read_text(encoding="utf-8")
    assert "False Positives: 1" in content
    assert "False Negatives: 1" in content
    assert "=== НАБЛЮДЕНИЯ ===" in content
    assert "Добавьте сюда" not in content


def test_comparison_results_file(tmp_path: Path) -> None:
    fine_tuned = compute_metrics([0, 1, 1, 0], [0, 1, 1, 0])
    baseline = compute_metrics([0, 1, 1, 0], [0, 1, 0, 0])

    assert fine_tuned.accuracy == 1.0
    assert fine_tuned.macro_f1 > baseline.macro_f1

    path = save_comparison_results(
        fine_tuned, baseline, tmp_path / "comparison_results.txt"
    )
    content = path.read_text(encoding="utf-8")
    assert "Fine-tuned Model:" in content
    assert "Baseline Model:" in content
    assert "Улучшение F1:" in content


def test_api_predict_endpoint(monkeypatch) -> None:
    tokenizer = load_tokenizer(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.eval()
    monkeypatch.setattr(api, "load_model", lambda: (model, tokenizer))

    with TestClient(api.app) as client:
        response = client.post("/predict", json={"text": "Great movie!"})
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in {"Negative", "Positive"}
        assert data["label"] in {0, 1}
        assert set(data["probabilities"]) == {"Negative", "Positive"}
        assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-5

        assert client.post("/predict", json={"text": ""}).status_code == 422
