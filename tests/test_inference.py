import numpy as np
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoModelForSequenceClassification

from sentiment_lab.evaluation import plot_confusion_matrix
from sentiment_lab.inference import predict_baseline, predict_fine_tuned
from sentiment_lab.tokenization import load_tokenizer

MODEL_NAME = "hf-internal-testing/tiny-random-bert"


def test_predict_fine_tuned_returns_probabilities() -> None:
    tokenizer = load_tokenizer(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.eval()

    single = predict_fine_tuned("Great movie!", model, tokenizer)
    many = predict_fine_tuned(["Great movie!", "Terrible film"], model, tokenizer)

    assert len(single) == 1 and len(many) == 2
    assert single[0].text == "Great movie!"
    assert single[0].prediction in (0, 1)
    assert single[0].probabilities.shape == (2,)
    assert np.isclose(single[0].probabilities.sum(), 1.0)


def test_predict_baseline_uses_cls_embeddings() -> None:
    tokenizer = load_tokenizer(MODEL_NAME)
    encoder_model = AutoModel.from_pretrained(MODEL_NAME)
    encoder_model.eval()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, encoder_model.config.hidden_size))
    y = (X[:, 0] > 0).astype(int)
    classifier = LogisticRegression(max_iter=100).fit(X, y)

    predictions = predict_baseline("Good movie", classifier, tokenizer, encoder_model)

    assert len(predictions) == 1
    assert predictions[0].prediction in (0, 1)
    assert predictions[0].probabilities.shape == (2,)


def test_plot_confusion_matrix_saves_png(tmp_path) -> None:
    path = plot_confusion_matrix(
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        labels=[0, 1],
        name="test",
        output_dir=tmp_path,
    )

    assert path.exists()
    assert path.stat().st_size > 0
