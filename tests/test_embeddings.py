import numpy as np

from sentiment_lab.embeddings import get_embeddings, load_model, similarity
from sentiment_lab.tokenization import load_tokenizer

MODEL_NAME = "hf-internal-testing/tiny-random-bert"


def test_load_model_sets_eval_mode() -> None:
    model = load_model(MODEL_NAME)

    assert not model.training


def test_get_embeddings_shape_matches_hidden_size() -> None:
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME)
    texts = [
        "This movie was absolutely amazing!",
        "Terrible movie, waste of time.",
    ]

    embeddings = get_embeddings(texts, tokenizer, model, batch_size=1)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(texts), model.config.hidden_size)


def test_get_embeddings_is_independent_of_batch_size() -> None:
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME)
    texts = ["short text", "a considerably longer piece of text here"]

    one_by_one = get_embeddings(texts, tokenizer, model, batch_size=1)
    together = get_embeddings(texts, tokenizer, model, batch_size=32)

    assert np.allclose(one_by_one, together, atol=1e-5)


def test_similarity_of_identical_text_is_one() -> None:
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME)
    text = "Great movie!"

    sim = similarity(text, text, tokenizer, model)

    assert abs(sim - 1.0) < 1e-5
