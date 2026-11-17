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

Опционально — демо на Gradio (день 7):

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

## Структура

```
.
├── data/                  # датасеты (в git не попадают)
├── notebooks/             # по одному ноутбуку на день
├── scripts/
│   ├── check_env.py       # smoke-тест окружения
│   └── download_data.py   # загрузка IMDB -> CSV
├── src/sentiment_lab/     # общий код, который переиспользуется из ноутбуков
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
| 4 | Baseline | TF-IDF + LogisticRegression, метрики | `sklearn`, `pandas` |
| 5 | Fine-tuning | Дообучение `distilbert-base-uncased` | `Trainer` / `TrainingArguments` |
| 6 | Сравнение | Accuracy/F1, confusion matrix, скорость, размер | `evaluate`, `seaborn` |
| 7 | Анализ и демо | Разбор ошибок, Gradio-приложение | `gradio` (группа `demo`) |

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
