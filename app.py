from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load mô hình đã huấn luyện
model = joblib.load("svm_model.pkl")

app = FastAPI(
    title="Iris Classification API",
    description="SVM model for the Iris dataset",
    version="1.0.0",
)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

species = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}

# URL ảnh minh họa cho từng loài hoa
species_images = {
    0: "https://commons.wikimedia.org/wiki/File:Iris_setosa_var._setosa_(2595031014).jpg",
    1: "https://commons.wikimedia.org/wiki/File:.00_7973_Verschiedenfarbige_Schwertlilie_(Iris_versicolor).jpg",
    2: "https://commons.wikimedia.org/wiki/File:Iris_virginica.jpg",
}

@app.get("/")
def home():
    return {"message": "Iris SVM API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]
    prediction = int(model.predict(features)[0])
    return {
        "class_id": prediction,
        "prediction": species[prediction],
        "image_url": species_images[prediction],
    }
