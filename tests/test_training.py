import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiment_lab.dataset import SentimentDataset
from sentiment_lab.tokenization import load_tokenizer
from sentiment_lab.training import evaluate, save_finetuned, train_epoch, train_model

MODEL_NAME = "hf-internal-testing/tiny-random-bert"
TEXTS = [
    "a great movie",
    "terrible film",
    "I loved it",
    "boring and long",
    "amazing acting",
    "waste of time",
    "pretty good",
    "awful script",
]
LABELS = [1, 0, 1, 0, 1, 0, 1, 0]


def make_loaders() -> tuple[AutoTokenizer, DataLoader, DataLoader]:
    tokenizer = load_tokenizer(MODEL_NAME)
    train = SentimentDataset(TEXTS, LABELS, tokenizer, max_length=16)
    val = SentimentDataset(TEXTS[:4], LABELS[:4], tokenizer, max_length=16)
    return tokenizer, DataLoader(train, batch_size=4), DataLoader(val, batch_size=4)


def make_model() -> AutoModelForSequenceClassification:
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


def test_dataset_returns_padded_tensors() -> None:
    tokenizer = load_tokenizer(MODEL_NAME)
    dataset = SentimentDataset(TEXTS, LABELS, tokenizer, max_length=16)

    item = dataset[0]

    assert len(dataset) == len(TEXTS)
    assert tuple(item["input_ids"].shape) == (16,)
    assert tuple(item["attention_mask"].shape) == (16,)
    assert item["labels"].dtype == torch.long


def test_train_epoch_and_evaluate() -> None:
    _, train_loader, val_loader = make_loaders()
    model = make_model()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device("cpu")

    loss = train_epoch(model, train_loader, optimizer, device)
    accuracy, f1 = evaluate(model, val_loader, device)

    assert loss > 0
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_train_model_and_save(tmp_path) -> None:
    tokenizer, train_loader, val_loader = make_loaders()
    model = make_model()

    result = train_model(model, train_loader, val_loader, num_epochs=1)
    model_dir, results_path = save_finetuned(
        model,
        tokenizer,
        result,
        tmp_path / "model",
        tmp_path / "metrics.txt",
    )

    assert len(result.history) == 1
    assert (model_dir / "config.json").exists()
    assert "Final Validation F1" in results_path.read_text(encoding="utf-8")
