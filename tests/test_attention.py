from sentiment_lab.attention import (
    attention_for_token,
    get_attention,
    load_attention_model,
    visualize_attention,
)
from sentiment_lab.tokenization import load_tokenizer

MODEL_NAME = "hf-internal-testing/tiny-random-bert"
TEXT = "The amazing movie won many awards"


def test_get_attention_shapes() -> None:
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_attention_model(MODEL_NAME)

    tokens, attentions = get_attention(TEXT, tokenizer, model)

    assert isinstance(attentions, tuple)
    assert len(attentions) == model.config.num_hidden_layers
    seq_len = tokens["input_ids"].shape[1]
    assert attentions[0].shape == (
        1,
        model.config.num_attention_heads,
        seq_len,
        seq_len,
    )


def test_visualize_attention_saves_png(tmp_path) -> None:
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_attention_model(MODEL_NAME)
    tokens, attentions = get_attention(TEXT, tokenizer, model)

    path = visualize_attention(
        tokens,
        attentions,
        tokenizer,
        layer=0,
        head=0,
        output_dir=tmp_path,
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_attention_for_token_returns_sorted_weights() -> None:
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_attention_model(MODEL_NAME)
    text = "This movie was absolutely terrible and I hated it"
    tokens, attentions = get_attention(text, tokenizer, model)
    token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    top = attention_for_token(
        tokens,
        attentions,
        tokenizer,
        query=token_list[1],
        layer=0,
        head=0,
        top_k=3,
    )

    assert len(top) == 3
    assert all(token in token_list for token, _ in top)
    weights = [weight for _, weight in top]
    assert weights == sorted(weights, reverse=True)
