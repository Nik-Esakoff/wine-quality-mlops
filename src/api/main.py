import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/wine_model.joblib")
MODEL_INFO_PATH = Path("models/model_info.json")

app = FastAPI(
    title="Wine Quality API",
    version="1.0.0",
)

model = None


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(ge=0)
    volatile_acidity: float = Field(ge=0)
    citric_acid: float = Field(ge=0)
    residual_sugar: float = Field(ge=0)
    chlorides: float = Field(ge=0)
    free_sulfur_dioxide: float = Field(ge=0)
    total_sulfur_dioxide: float = Field(ge=0)
    density: float = Field(ge=0)
    ph: float = Field(ge=0, le=14)
    sulphates: float = Field(ge=0)
    alcohol: float = Field(ge=0)


def load_model():
    global model

    if model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run dvc pull first.",
            )

        model = joblib.load(MODEL_PATH)

    return model


@app.get("/healthcheck")
def healthcheck() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    if not MODEL_INFO_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model information not found. Run dvc pull first.",
        )

    with MODEL_INFO_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.post("/predict")
def predict(features: WineFeatures) -> dict[str, float | int]:
    current_model = load_model()

    row = {
        "fixed acidity": features.fixed_acidity,
        "volatile acidity": features.volatile_acidity,
        "citric acid": features.citric_acid,
        "residual sugar": features.residual_sugar,
        "chlorides": features.chlorides,
        "free sulfur dioxide": features.free_sulfur_dioxide,
        "total sulfur dioxide": features.total_sulfur_dioxide,
        "density": features.density,
        "pH": features.ph,
        "sulphates": features.sulphates,
        "alcohol": features.alcohol,
    }

    dataframe = pd.DataFrame([row])
    prediction = float(current_model.predict(dataframe)[0])

    return {
        "prediction": prediction,
        "rounded_quality": int(round(prediction)),
    }
