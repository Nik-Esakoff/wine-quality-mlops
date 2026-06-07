# Wine Quality MLOps Project

Учебный MLOps-проект для предсказания качества красного вина.

Проект включает:

* обучение двух моделей;
* трекинг экспериментов через MLflow;
* версионирование данных и модели через DVC;
* API на FastAPI;
* автоматические тесты;
* Airflow DAG;
* GitLab CI pipeline.

## Структура проекта

```text
wine-quality-mlops/
├── config/
│   └── params.yaml
├── dags/
│   └── train_wine_quality_dag.py
├── data/
│   └── raw/
├── models/
├── src/
│   ├── train.py
│   └── api/
│       └── main.py
├── tests/
│   └── test_api.py
├── .gitlab-ci.yml
├── models.dvc
├── pyproject.toml
└── requirements.txt
```

## Данные

Используется датасет Red Wine Quality.

Целевая переменная:

```text
quality
```

Датасет не хранится напрямую в Git. Он версионируется через DVC и загружается из удалённого S3-совместимого хранилища.

Для загрузки данных и модели:

```bash
dvc pull
```

## Обучение модели

Параметры обучения находятся в:

```text
config/params.yaml
```

Запуск обучения:

```bash
python src/train.py
```

В проекте сравниваются две модели:

* Ridge Regression;
* Random Forest Regressor.

Для оценки используются метрики:

* MAE;
* RMSE;
* R².

## Результаты экспериментов

Лучшей моделью стала Random Forest Regressor.

Полученные метрики:

| Метрика | Значение |
| ------- | -------: |
| MAE     |   0.4619 |
| RMSE    |   0.5800 |
| R²      |   0.4853 |

Лучшая модель выбирается по минимальному значению RMSE.

Эксперименты и параметры моделей отслеживаются через MLflow.

## Версионирование модели

После обучения создаются файлы:

```text
models/wine_model.joblib
models/model_info.json
```

Они версионируются через DVC.

Для отправки новой версии модели в удалённое хранилище:

```bash
dvc add models
dvc push
```

## API

Запуск FastAPI:

```bash
uvicorn src.api.main:app --reload
```

Swagger-документация доступна по адресу:

```text
http://127.0.0.1:8000/docs
```

Реализованы эндпоинты:

### `GET /healthcheck`

Проверяет работоспособность API и наличие модели.

### `GET /model-info`

Возвращает название модели, метрики, список признаков и дату создания.

### `POST /predict`

Принимает характеристики вина и возвращает предсказанное качество.

Пример запроса:

```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "ph": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

Пример ответа:

```json
{
  "prediction": 5.152877454233127,
  "rounded_quality": 5
}
```

## Тестирование

Запуск тестов:

```bash
python -m pytest tests/ -v
```

Тесты проверяют:

* `/healthcheck`;
* `/model-info`;
* корректный запрос к `/predict`;
* валидацию некорректного запроса.

## Линтер

Для проверки качества кода используется Ruff:

```bash
ruff check src tests dags
```

## Airflow

DAG находится в файле:

```text
dags/train_wine_quality_dag.py
```

Пайплайн выполняет задачи в следующем порядке:

```text
Загрузка данных через DVC → Обучение модели → Сохранение модели в DVC
```

Расписание запуска:

```text
ежедневно
```

## CI/CD

В `.gitlab-ci.yml` настроены этапы:

1. проверка кода линтером;
2. проверка доступности данных и модели через `dvc pull`;
3. запуск API-тестов.

Pipeline запускается для merge request в ветку `main`.

Секретные ключи для DVC должны храниться в GitLab CI/CD Variables и не добавляться в репозиторий.

## Локальный запуск

Создать виртуальное окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

Загрузить данные и модель:

```bash
dvc pull
```

Запустить API:

```bash
uvicorn src.api.main:app --reload
```

