import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_params(path: str = "config/params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")

    # На случай если файл окажется разделён запятыми, а не точкой с запятой
    if len(df.columns) == 1:
        df = pd.read_csv(path)

    return df


def build_models(params: dict) -> dict:
    return {
        "ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=params["models"]["ridge"]["alpha"])),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=params["models"]["random_forest"]["n_estimators"],
                        max_depth=params["models"]["random_forest"]["max_depth"],
                        random_state=params["models"]["random_forest"]["random_state"],
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model, x_test, y_test) -> dict:
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)

    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, predictions)),
    }


def main() -> None:
    params = load_params()

    data_path = params["data"]["path"]
    target_col = params["data"]["target"]

    df = read_dataset(data_path)

    x = df.drop(columns=[target_col])
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("wine-quality")

    models = build_models(params)

    best_model_name = None
    best_model = None
    best_metrics = None

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(x_train, y_train)
            metrics = evaluate_model(model, x_test, y_test)

            mlflow.log_param("model_name", model_name)
            mlflow.log_params(params["models"].get(model_name, {}))
            mlflow.log_metrics(metrics)

            print(f"{model_name}: {metrics}")

            if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
                best_model_name = model_name
                best_model = model
                best_metrics = metrics

    output_model_path = Path(params["output"]["model_path"])
    output_info_path = Path(params["output"]["model_info_path"])

    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, output_model_path)

    model_info = {
        "model_name": best_model_name,
        "metrics": best_metrics,
        "features": list(x.columns),
        "target": target_col,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(output_info_path, "w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=2)

    print()
    print(f"Best model: {best_model_name}")
    print(f"Saved model to: {output_model_path}")
    print(f"Saved model info to: {output_info_path}")


if __name__ == "__main__":
    main()
