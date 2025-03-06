from fastapi import FastAPI
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Cargar modelos guardados
modelo_regresion = CatBoostRegressor()
modelo_regresion.load_model("modelo_catboost_temperatura")

modelo_lluvia = CatBoostClassifier()
modelo_lluvia.load_model("modelo_catboost_lluvia")

# Crear la API con FastAPI
app = FastAPI()

# Agregar middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los encabezados
)

# Definir la estructura de los datos de entrada
class InputData(BaseModel):
    features: list[float]  # Lista de características para la predicción

@app.post("/predict_regression")
def predict_regression(data: InputData):
    """Predice un valor numérico usando el modelo de regresión (temperatura)."""
    X_input = np.array(data.features).reshape(1, -1)
    prediction = modelo_regresion.predict(X_input)
    return {"prediction": prediction.tolist()}


@app.post("/predict_classification")
def predict_classification(data: InputData):
    """Predice si va a llover (0 = No, 1 = Sí)."""
    X_input = np.array(data.features).reshape(1, -1)
    prediction = modelo_lluvia.predict(X_input)
    return {"rain_prediction": int(prediction[0])}
