# HW5 MLOps — Reproducible ML Experiment

## Цель проекта

Построить минимальный MLOps-контур с воспроизводимым экспериментом: контроль версий данных (DVC), автоматизация пайплайна обучения, логирование экспериментов (MLflow).

## Структура проекта

```
├── data/
│   ├── raw/          # Сырые данные (версионируются через DVC)
│   └── processed/    # Подготовленные данные (генерируются пайплайном)
├── src/
│   ├── prepare.py    # Подготовка данных (train/test split)
│   └── train.py      # Обучение модели + логирование в MLflow
├── dvc.yaml          # Описание пайплайна
├── dvc.lock          # Фиксация версий пайплайна
├── params.yaml       # Гиперпараметры
├── requirements.txt  # Зависимости
└── README.md
```

## Как запустить

```bash
git clone <repo-url>
cd HW5
pip install -r requirements.txt
dvc pull
dvc repro
```

## Описание пайплайна

Пайплайн состоит из двух стадий (dvc.yaml):

1. **prepare** — читает `data/raw/data.csv`, делает train/test split согласно `params.yaml`, сохраняет в `data/processed/`
2. **train** — обучает RandomForestClassifier, логирует параметры и метрики в MLflow, сохраняет `model.pkl`

Параметры пайплайна (`params.yaml`):
- `split_ratio: 0.2`
- `random_state: 42`
- `n_estimators: 100`

## MLflow UI

Для просмотра экспериментов запустите:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Откройте в браузере: http://127.0.0.1:5000

В разделе **Experiments → iris_experiment** будут видны параметры, метрики (accuracy) и артефакт (model.pkl).
