import numpy as np

from sentiment_lab.baseline import save_results, train_baseline


def make_data(n_per_class: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.concatenate(
        [
            rng.normal(-1.0, 1.0, (n_per_class, 8)),
            rng.normal(1.0, 1.0, (n_per_class, 8)),
        ]
    )
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def test_train_baseline_returns_metrics() -> None:
    X, y = make_data()

    result = train_baseline(X, y)

    assert 0.0 <= result.macro_f1 <= 1.0
    assert result.macro_f1 > 0.8
    assert len(result.y_test) == len(result.y_pred) == 24
    assert "precision" in result.report


def test_save_results(tmp_path) -> None:
    X, y = make_data()
    result = train_baseline(X, y)

    path = save_results(result, tmp_path / "baseline_results.txt")

    text = path.read_text(encoding="utf-8")
    assert "Macro F1" in text
    assert "precision" in text
