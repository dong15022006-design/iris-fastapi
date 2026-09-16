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
    0: "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_2.jpg",
    1: "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    2: "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica_2.jpg",
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
