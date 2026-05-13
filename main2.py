from fastapi import FastAPI, UploadFile, File
import joblib
import numpy as np
import os
from detectarGrieta5 import detect_cracks_with_labels
from vectorCaracteristicas import extraer_densenet
from vectorCaracteristicasVGG19 import extraer_fc6
from vectorCaracteristicasVGG19_fc7 import extraer_fc7

app = FastAPI(title="Crack Detection API", version="2.0")

# Cargar modelos entrenados
models = {
    "fc6": joblib.load("crack_detector_rf_fc6_final.pkl"),
    "fc7": joblib.load("crack_detector_rf_fc7_final.pkl"),
    "densenet121": joblib.load("crack_detector_rf_densenet121_final.pkl")
}

# Función para seleccionar el extractor correcto
def extract_features(img_path: str, model_name: str) -> np.ndarray:
    if model_name == "fc6":
        return extraer_fc6(img_path)
    elif model_name == "fc7":
        return extraer_fc7(img_path)
    elif model_name == "densenet121":
        return extraer_densenet(img_path)
    else:
        raise ValueError(f"Modelo '{model_name}' no soportado")

@app.get("/")
def home():
    return {"message": "API de detección de grietas funcionando con imágenes"}

@app.post("/predict_image")
async def predict_image(model_name: str, file: UploadFile = File(...)):
    if model_name not in models:
        return {"error": f"Modelo '{model_name}' no disponible"}

    # Guardar imagen temporalmente
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        # Extraer características
        features = extract_features(temp_path, model_name).reshape(1, -1)

        # Realizar predicción
        model = models[model_name]
        prediction = model.predict(features)[0]
        probas = model.predict_proba(features)[0].tolist() if hasattr(model, "predict_proba") else None
        if int(prediction) == 1:
            detect_cracks_with_labels(
                temp_path,
                clahe_clip=2.0,
                clahe_grid=(8,8),
                bh_kernel=(21,21),
                otsu=True,
                canny_sigma=0.33,
                close_iter=3,
                open_iter=1,
                min_length=30
            )

        return {
            "model": model_name,
            "prediction": int(prediction),
            "probabilities": probas,
            "pipeline_executed": bool(prediction == 1)
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)