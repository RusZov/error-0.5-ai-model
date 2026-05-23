# Error 0.5 AI Model Experiment

Научный проект для анализа согласованности и неопределённости ответов языковых моделей на основе теории «ошибки 0.5» Баркалова В.В.

## 📖 Описание

Этот проект реализует научную теорию «ошибки 0.5» (Баркалов В.В., 2026) для оценки:
- **K_rep** — коэффициент повторяемости (насколько модель выдаёт одинаковые ответы)
- **K_div** — коэффициент разнообразия (1 - K_rep)
- **H** — энтропия Шеннона (мера неопределённости)
- **D_lack** — индекс нехватки данных
- **Ω_0.5** — индекс полезной ошибки

Полный текст теории доступен в файле `docs/theory_full.md`.

## 🚀 Установка

```bash
# Клонируйте репозиторий
cd error-0.5-ai-model

# Установите зависимости
pip install -r requirements.txt

# Для работы с OpenAI API установите переменную окружения
export OPENAI_API_KEY="your-api-key-here"
```

### Минимальные зависимости

Базовый набор (работает всегда):
```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
pyyaml>=6.0
```

Опционально для OpenAI:
```
openai>=1.0.0
```

Опционально для локальных моделей:
```
transformers>=4.30.0
torch>=2.0.0
accelerate>=0.20.0
```

## 📋 Быстрый старт

### Пример 1: Базовый эксперимент

```python
from src import ExperimentRunner, load_config

# Загрузить конфигурацию
config = load_config('config.yaml')

# Создать runner
runner = ExperimentRunner(config)

# Запустить эксперимент
prompts = [
    "What is 2+2?",
    "Explain quantum computing in one sentence",
    "Is the sky blue?"
]

results = runner.run_experiment(
    prompts=prompts,
    n_repetitions=100,  # N прогонов
    temperature=0.7,
)

# Результаты сохранены в results/
```

### Пример 2: Сравнение температур

```python
from src import ExperimentRunner, load_config

config = load_config('config.yaml')
runner = ExperimentRunner(config)

prompts = ["What is the capital of France?"]

# Сравнить температуру 0.3 vs 1.0
results = runner.compare_experiments(
    prompts=prompts,
    params_a={'temperature': 0.3},
    params_b={'temperature': 1.0},
    n_repetitions=100,
)
```

### Пример 3: Использование через CLI

```bash
# Базовый запуск
python -m src.experiment --prompts "What is 2+2?" "Who is the president?"

# С указанием параметров
python -m src.experiment \
    --prompts "Test prompt" \
    --n-repetitions 50 \
    --temperature 0.8

# Режим сравнения
python -m src.experiment \
    --prompts "Test prompt" \
    --temperature 0.5 \
    --compare \
    --temp-b 1.5

# С использованием теории как контекста
python -m src.experiment \
    --prompts "Explain probability theory" \
    --use-theory-context
```

## 📊 Интерпретация коэффициентов

### K_rep (Repetition Coefficient)
- **Диапазон**: 0.0 – 1.0
- **Высокий (>0.8)**: Модель очень стабильна, выдаёт одинаковые ответы
- **Низкий (<0.3)**: Модель непредсказуема, ответы сильно различаются
- **Оптимальный**: ~0.5–0.7 (баланс стабильности и гибкости)

### K_div (Diversity Coefficient)
- **Формула**: K_div = 1 - K_rep
- **Высокий**: Большое разнообразие ответов
- **Низкий**: Ответы шаблонные

### H (Entropy)
- **Единицы**: биты
- **Высокая**: Высокая неопределённость, много категорий ответов
- **Низкая**: Предсказуемые ответы
- **Максимум**: log2(n_categories) при равномерном распределении

### D_lack (Data Lack Index)
- **Диапазон**: 0.0 – 1.0
- **Высокий (>0.5)**: Модель часто отвечает "не знаю" или противоречиво
- **Низкий (<0.2)**: Модель уверена в ответах
- **Интерпретация**: Доля неопределённых/противоречивых ответов

### Ω_0.5 (Useful Error Index)
- **Формула**: Ω_0.5 = P_gain × B - P_loss × L
- **Положительный**: Разнообразие приносит пользу (новые инсайты)
- **Отрицательный**: Ошибки перевешивают преимущества
- **Близкий к 0**: Баланс между пользой и вредом

## 📁 Структура проекта

```
error-0.5-ai-model/
├── README.md              # Этот файл
├── LICENSE                # MIT лицензия
├── requirements.txt       # Зависимости
├── config.yaml           # Конфигурация
├── main.py               # Точка входа (CLI)
├── docs/
│   └── theory_full.md    # Полная теория Баркалова В.В.
├── src/
│   ├── __init__.py       # Пакет
│   ├── experiment.py     # Запуск экспериментов
│   ├── metrics.py        # Расчёт метрик
│   ├── llm_client.py     # Клиенты LLM (OpenAI, local)
│   └── utils.py          # Утилиты
├── results/              # Вывод экспериментов
│   └── example_output.csv
└── notebooks/
    └── demo.ipynb        # Jupyter демо
```

## ⚙️ Конфигурация (config.yaml)

```yaml
llm:
  provider: "openai"  # или "local"
  openai:
    model: "gpt-3.5-turbo"
    max_tokens: 512

generation:
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  system_prompt: "You are a helpful assistant."

theory:
  enabled: true
  source_file: "docs/theory_full.md"
  use_as_system_prompt: false
  use_as_context: true

experiment:
  default_repetitions: 100
  noise_filter_enabled: true

metrics:
  quality_threshold: 0.8
  p_gain: 1.0
  b_value: 1.0
  p_loss: 0.5
  l_value: 0.5
```

## 🔬 Научная основа

Теория «ошибки 0.5» (Баркалов В.В., 2026) предполагает, что:
1. Идеальная модель должна показывать **K_rep ≈ 0.5** для творческих задач
2. Для фактологических задач **K_rep → 1.0** (стабильность важнее)
3. **Ω_0.5 > 0** указывает на продуктивную вариативность
4. **D_lack** коррелирует с недостатком обучающих данных по теме

Ключевая идея: не всякое отклонение является вредным. Иногда ошибка формально нарушает план, но приносит выигрыш («правильная ошибка»).

## 📝 Лицензия

MIT License — см. файл [LICENSE](LICENSE)

## 🤝 Вклад

Pull requests приветствуются! Для загрузки теории/документации:
1. Создайте issue с описанием
2. Прикрепите файл в обсуждении
3. Или сделайте PR с файлом в папку `docs/`

## 📧 Контакты

Для вопросов по теории «ошибки 0.5» и сотрудничеству открывайте issues на GitHub.
