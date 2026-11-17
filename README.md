# Sentiment Lab — трансформеры за 7 дней

Учебный проект: от токенизации до дообучения трансформера на задаче
классификации тональности (sentiment analysis).

## Требования

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- ~3 ГБ свободного места (torch + модели + датасет)
- GPU не обязателен: на CPU всё работает, если брать подвыборку данных

## Установка

```bash
# из корня проекта
uv sync
```

Опционально — демо на FastAPI (день 7):

```bash
uv sync --group demo
```

> `.venv` привязан к ОС и Python. Если проект открывается на другой машине
> (например, папка лежит на Windows-диске, а команды запускались в Linux),
> удалите `.venv` и выполните `uv sync` заново.

## Данные

Скачать IMDB и разложить в CSV:

```bash
uv run python scripts/download_data.py
# быстрый вариант для CPU-экспериментов:
uv run python scripts/download_data.py --max-rows 5000
```

Получится:

```
data/imdb_train.csv   # колонки: text, label
data/imdb_test.csv
```

Альтернативы (см. `datasets` на Hugging Face):

- `imdb` — английский, 0/1, 50k отзывов;
- `sst2` — английский, короткие фразы;
- `tweet_eval` с конфигом `sentiment` — английский, 0/1/2:
  `uv run python scripts/download_data.py --dataset tweet_eval --config sentiment`;
- русский: `blinoff/kinopoisk` или `sepidmnorozy/Russian_sentiment`.

Короткие имена `imdb`, `sst2`, `tweet_eval` поддерживаются; для остальных
датасетов указывайте полный id вида `namespace/name`.

Скрипт сам приводит колонки к формату `text,label`, даже если в исходном
датасете текст называется `sentence`, `review` или `content`.

## Проверка окружения

```bash
uv run python scripts/check_env.py
```

Скрипт печатает версии и прогоняет крошечную модель
(`hf-internal-testing/tiny-random-bert`), чтобы убедиться, что
torch + transformers работают.

## Токенизация

Пошаговое выполнение задач Дня 1 — в `notebooks/01_tokenization.ipynb`.
Переиспользуемый код живёт в `src/sentiment_lab/tokenization.py`:

```python
from sentiment_lab.tokenization import (
    explain_tokenization,
    load_tokenizer,
    tokenize_texts,
)

tokenizer = load_tokenizer()
batch = tokenize_texts(tokenizer, ["Отличный фильм!", "Ужасный фильм."])
print(explain_tokenization("Transformers are amazing!", tokenizer))
```

## Эмбеддинги

Пошаговое выполнение задач Дня 2 — в `notebooks/02_embeddings.ipynb`.
Переиспользуемый код живёт в `src/sentiment_lab/embeddings.py`:

```python
from sentiment_lab.embeddings import get_embeddings, load_model, similarity
from sentiment_lab.tokenization import load_tokenizer

tokenizer = load_tokenizer()
model = load_model()
embeddings = get_embeddings(["Отличный фильм!", "Ужасный фильм."], tokenizer, model)
print(embeddings.shape)
print(similarity("Отличный фильм!", "Ужасный фильм.", tokenizer, model))
```

## Attention

Пошаговое выполнение задач Дня 3 — в `notebooks/03_attention.ipynb`.
Код в `src/sentiment_lab/attention.py`; heatmap сохраняются в `outputs/`:

```python
from sentiment_lab.attention import (
    attention_for_token,
    get_attention,
    load_attention_model,
    visualize_attention,
)
from sentiment_lab.tokenization import load_tokenizer

tokenizer = load_tokenizer()
model = load_attention_model()
tokens, attentions = get_attention("The amazing movie won many awards", tokenizer, model)
visualize_attention(tokens, attentions, tokenizer, layer=0, head=0)
print(attention_for_token(tokens, attentions, tokenizer, "movie", 0, 0))
```

## Baseline

Пошаговое выполнение задач Дня 4 — в `notebooks/04_baseline.ipynb`.
Логистическая регрессия на CLS-эмбеддингах в `src/sentiment_lab/baseline.py`,
результат сохраняется в `outputs/baseline_results.txt`:

```python
from sentiment_lab.baseline import save_results, train_baseline

result = train_baseline(embeddings, labels)
print(result.report)
print(f"Macro F1: {result.macro_f1:.3f}")
save_results(result)
```

## Fine-tuning

Пошаговое выполнение задач Дня 5 — в `notebooks/05_finetuning.ipynb`.
`SentimentDataset` в `sentiment_lab.dataset`, цикл обучения и сохранение —
в `sentiment_lab.training`. В transformers 5.x используется `torch.optim.AdamW`
(`transformers.AdamW` больше нет).

```python
from sentiment_lab.training import save_finetuned, train_model

result = train_model(model, train_loader, val_loader, num_epochs=3)
save_finetuned(model, tokenizer, result)
```

Модель сохраняется в `models/fine_tuned_model`, метрики — в
`outputs/fine_tuned_results.txt`.

## Инференс и сравнение

Пошаговое выполнение задач Дня 6 — в `notebooks/06_comparison.ipynb`.
Предсказания в `sentiment_lab.inference`, confusion matrix — в
`sentiment_lab.evaluation`:

```python
from sentiment_lab.inference import predict_baseline, predict_fine_tuned

preds_ft = predict_fine_tuned(["Great movie!"], model_ft, tokenizer)
preds_base = predict_baseline(
    ["Great movie!"], baseline_model, tokenizer, encoder_model
)
```

Базовый вариант из задания использует TF-IDF `vectorizer`; здесь baseline
обучен на CLS-эмбеддингах, поэтому `predict_baseline` принимает `tokenizer`
и `encoder_model`.

## Анализ ошибок

Пошаговое выполнение задач Дня 7 — в `notebooks/07_error_analysis.ipynb`.
Разбор FP/FN и сохранение отчёта — в `sentiment_lab.error_analysis`:

```python
from sentiment_lab.error_analysis import analyze_errors, save_error_report

analysis = analyze_errors(texts, true_labels, pred_labels)
path = save_error_report(analysis)
```

Отчёты: `outputs/error_analysis.txt` и `outputs/comparison_results.txt`.

### Результаты на 500 отзывах `imdb_test`

| Модель | Accuracy | Macro F1 |
|--------|----------|----------|
| Fine-tuned | 0.8080 | 0.8032 |
| Baseline | 0.7740 | 0.7739 |

Улучшение macro F1: **3.78%**. В ошибках fine-tuned модели 14 FP и 82 FN,
средняя длина ошибочных текстов 1327 символов против 1287 у всех.

## FastAPI-демо

```bash
uv sync --group demo
uv run uvicorn api:app --reload
```

POST http://127.0.0.1:8000/predict:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Great movie!\"}"
```

Ответ содержит `prediction`, `label` и вероятности по классам. Модель
двуклассовая (`Negative` / `Positive`); `Neutral` из примера задания здесь
нет, потому что IMDB размечен на 0/1.

## Тесты

```bash
uv run pytest
```

## Структура

```
.
├── data/                  # датасеты (в git не попадают)
├── notebooks/
│   ├── 01_tokenization.ipynb  # задачи Дня 1 по шагам
│   ├── 02_embeddings.ipynb    # задачи Дня 2 по шагам
│   ├── 03_attention.ipynb     # задачи Дня 3 по шагам
│   ├── 04_baseline.ipynb      # задачи Дня 4 по шагам
│   ├── 05_finetuning.ipynb    # задачи Дня 5 по шагам
│   ├── 06_comparison.ipynb    # задачи Дня 6 по шагам
│   └── 07_error_analysis.ipynb # задачи Дня 7 по шагам
├── scripts/
│   ├── check_env.py       # smoke-тест окружения
│   └── download_data.py   # загрузка IMDB -> CSV
├── src/sentiment_lab/
│   ├── __init__.py        # точка входа пакета
│   ├── tokenization.py    # токенизатор, батчи, разбор токенов
│   ├── embeddings.py      # hidden states, CLS-эмбеддинги, сходство
│   ├── attention.py       # attention-веса, heatmap, разбор по слоям/головам
│   ├── baseline.py        # LogisticRegression, метрики, сохранение результатов
│   ├── dataset.py         # SentimentDataset для классификации
│   ├── training.py        # train_epoch, evaluate, обучение, сохранение модели
│   ├── inference.py       # predict_fine_tuned / predict_baseline
│   ├── evaluation.py      # confusion matrix и сравнение метрик
│   └── error_analysis.py  # FP/FN, отчёт с наблюдениями
├── tests/                 # pytest-тесты
├── api.py                 # FastAPI /predict
├── main.py                # точка входа для PyCharm
├── pyproject.toml
└── README.md
```

Пакет называется `sentiment_lab`, а не `transformers`, специально: иначе
локальный `src/transformers` перекрыл бы библиотеку Hugging Face
`transformers` при импорте.

## План на 7 дней

| День | Тема | Что делаем | Ключевые API |
|------|------|------------|--------------|
| 1 | Токенизация | Word-level vs BPE, `input_ids`, `attention_mask`, `decode` | `AutoTokenizer` |
| 2 | Hidden states | Эмбеддинги последнего слоя, mean-pooling, косинусная близость | `AutoModel(..., output_hidden_states=True)` |
| 3 | Attention | Тепловые карты внимания по слоям и головам | `output_attentions=True` |
| 4 | Baseline | CLS-эмбеддинги + LogisticRegression, метрики | `sklearn`, `pandas` |
| 5 | Fine-tuning | Дообучение `distilbert-base-uncased` | `Dataset`, `DataLoader`, `torch.optim.AdamW` |
| 6 | Сравнение | Accuracy/F1, confusion matrix, скорость, размер | `evaluate`, `seaborn` |
| 7 | Анализ и демо | Разбор ошибок, FastAPI-приложение | `fastapi`, `uvicorn` (группа `demo`) |

### Рекомендуемые модели

- для скорости на CPU: `distilbert-base-uncased` (~265 МБ);
- для русского: `bert-base-multilingual-cased` или `xlm-roberta-base`
  (для второго нужен `sentencepiece`, он уже в зависимостях);
- для smoke-тестов: `hf-internal-testing/tiny-random-bert` (~1.7 МБ).
  `prajjwal1/bert-tiny` для этого не подходит: в его репозитории нет
  `tokenizer.json`, и transformers 5.x не собирает токенизатор из одного
  `vocab.txt`.

### Практические советы

- Начните с подвыборки 2–5k примеров: один epoch на CPU — это минуты, а не часы.
- Фиксируйте `random_seed`, сохраняйте метрики в `outputs/`.
- Модели и чекпойнты складывайте в `models/` (обе папки в `.gitignore`).
- Для IMDB метрики: accuracy и macro-F1; для 3 классов — confusion matrix.
