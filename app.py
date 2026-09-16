from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

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

species_images = {
    0: "https://commons.wikimedia.org/wiki/File:Iris_setosa_var._setosa_(2595031014).jpg",
    1: "https://commons.wikimedia.org/wiki/File:.00_7973_Verschiedenfarbige_Schwertlilie_(Iris_versicolor).jpg",
    2: "https://commons.wikimedia.org/wiki/File:Iris_virginica.jpg",
}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <title>Iris Flower Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 20px; background: #f5f5f5; }
            h1 { color: #333; text-align: center; }
            .card { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            label { display: block; margin: 15px 0 5px; font-weight: bold; color: #555; }
            input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 16px; box-sizing: border-box; }
            button { width: 100%; padding: 12px; background: #4CAF50; color: white; border: none; border-radius: 6px; font-size: 18px; cursor: pointer; margin-top: 20px; }
            button:hover { background: #45a049; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🌸 Iris Flower Classifier</h1>
            <form action="/predict-form" method="post">
                <label>Sepal Length (cm)</label>
                <input type="number" step="0.1" name="sepal_length" value="5.1" required>
                <label>Sepal Width (cm)</label>
                <input type="number" step="0.1" name="sepal_width" value="3.5" required>
                <label>Petal Length (cm)</label>
                <input type="number" step="0.1" name="petal_length" value="1.4" required>
                <label>Petal Width (cm)</label>
                <input type="number" step="0.1" name="petal_width" value="0.2" required>
                <button type="submit">Dự đoán</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...),
):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = int(model.predict(features)[0])
    name = species[pred]
    img = species_images[pred]

    return f"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <title>Kết quả dự đoán</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 20px; background: #f5f5f5; text-align: center; }}
            .card {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; }}
            .species {{ color: #4CAF50; font-size: 32px; font-weight: bold; margin: 20px 0; text-transform: capitalize; }}
            img {{ max-width: 100%; border-radius: 12px; margin: 20px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }}
            .info {{ color: #666; margin: 15px 0; }}
            a {{ display: inline-block; margin-top: 20px; padding: 12px 30px; background: #2196F3; color: white; text-decoration: none; border-radius: 6px; font-size: 16px; }}
            a:hover {{ background: #1976D2; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Kết quả dự đoán</h1>
            <div class="species">{name}</div>
            <img src="{img}" alt="{name}">
            <div class="info">
                Class ID: <b>{pred}</b><br>
                Input: SL={sepal_length}, SW={sepal_width}, PL={petal_length}, PW={petal_width}
            </div>
            <a href="/">← Thử lại</a>
        </div>
    </body>
    </html>
    """

# Giữ endpoint /predict cho API thuần (Swagger test)
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

@app.get("/health")
def health():
    return {"status": "healthy"}
