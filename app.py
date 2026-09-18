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
    0: "https://raw.githubusercontent.com/dong15022006-design/iris-fastapi/main/images/setosa.jpg",
    1: "https://raw.githubusercontent.com/dong15022006-design/iris-fastapi/main/images/versicolor.jpg",
    2: "https://raw.githubusercontent.com/dong15022006-design/iris-fastapi/main/images/virginica.jpg",
}

# Màu chủ đạo cho từng loài
species_colors = {
    0: "#e74c3c",   # đỏ - setosa
    1: "#3498db",   # xanh dương - versicolor
    2: "#9b59b6",   # tím - virginica
}

# ============ TRANG CHỦ - FORM ============
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris Flower Classifier</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .container {
                background: white;
                max-width: 520px;
                width: 100%;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                text-align: center;
                color: #333;
                font-size: 28px;
                margin-bottom: 8px;
            }
            .subtitle {
                text-align: center;
                color: #888;
                font-size: 14px;
                margin-bottom: 30px;
            }
            .form-group { margin-bottom: 20px; }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #555;
                font-size: 14px;
            }
            input {
                width: 100%;
                padding: 12px 16px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 16px;
                transition: all 0.3s;
                outline: none;
            }
            input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            button {
                width: 100%;
                padding: 14px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 17px;
                font-weight: 600;
                cursor: pointer;
                margin-top: 10px;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            button:active { transform: translateY(0); }
            .hint {
                text-align: center;
                color: #aaa;
                font-size: 12px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌸 Iris Flower Classifier</h1>
            <p class="subtitle">Nhập 4 thông số để dự đoán loài hoa</p>
            <form action="/predict-form" method="post">
                <div class="form-group">
                    <label>Sepal Length (cm)</label>
                    <input type="number" step="0.1" name="sepal_length" value="5.1" required>
                </div>
                <div class="form-group">
                    <label>Sepal Width (cm)</label>
                    <input type="number" step="0.1" name="sepal_width" value="3.5" required>
                </div>
                <div class="form-group">
                    <label>Petal Length (cm)</label>
                    <input type="number" step="0.1" name="petal_length" value="1.4" required>
                </div>
                <div class="form-group">
                    <label>Petal Width (cm)</label>
                    <input type="number" step="0.1" name="petal_width" value="0.2" required>
                </div>
                <button type="submit">Dự đoán</button>
            </form>
            <p class="hint">Mô hình SVM • Độ chính xác 96.7%</p>
        </div>
    </body>
    </html>
    """

# ============ KẾT QUẢ DỰ ĐOÁN ============
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
    color = species_colors[pred]

    return f"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Kết quả: {name}</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }}
            .container {{
                background: white;
                max-width: 520px;
                width: 100%;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                text-align: center;
                animation: fadeIn 0.5s ease;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            h1 {{
                color: #333;
                font-size: 22px;
                margin-bottom: 10px;
                font-weight: 600;
            }}
            .species-badge {{
                display: inline-block;
                padding: 12px 32px;
                background: {color};
                color: white;
                border-radius: 50px;
                font-size: 24px;
                font-weight: 700;
                text-transform: capitalize;
                margin: 15px 0 25px;
                box-shadow: 0 8px 20px {color}55;
                letter-spacing: 1px;
            }}
            .image-wrapper {{
                margin: 20px 0;
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            }}
            .image-wrapper img {{
                width: 100%;
                height: auto;
                display: block;
            }}
            .info-box {{
                background: #f8f9fa;
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
                text-align: left;
                font-size: 14px;
                color: #555;
            }}
            .info-box .row {{
                display: flex;
                justify-content: space-between;
                padding: 6px 0;
                border-bottom: 1px solid #eee;
            }}
            .info-box .row:last-child {{ border-bottom: none; }}
            .info-box .label {{ font-weight: 600; color: #777; }}
            .info-box .value {{ color: #333; font-weight: 500; }}
            .btn-back {{
                display: inline-block;
                padding: 12px 36px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-decoration: none;
                border-radius: 10px;
                font-weight: 600;
                font-size: 15px;
                transition: transform 0.2s, box-shadow 0.2s;
                margin-top: 10px;
            }}
            .btn-back:hover {{
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Kết quả dự đoán</h1>
            <div class="species-badge">{name}</div>
            <div class="image-wrapper">
                <img src="{img}" alt="{name}">
            </div>
            <div class="info-box">
                <div class="row">
                    <span class="label">Class ID</span>
                    <span class="value">{pred}</span>
                </div>
                <div class="row">
                    <span class="label">Sepal Length</span>
                    <span class="value">{sepal_length} cm</span>
                </div>
                <div class="row">
                    <span class="label">Sepal Width</span>
                    <span class="value">{sepal_width} cm</span>
                </div>
                <div class="row">
                    <span class="label">Petal Length</span>
                    <span class="value">{petal_length} cm</span>
                </div>
                <div class="row">
                    <span class="label">Petal Width</span>
                    <span class="value">{petal_width} cm</span>
                </div>
            </div>
            <a href="/" class="btn-back">← Thử lại</a>
        </div>
    </body>
    </html>
    """

# ============ API THUẦN JSON (Swagger test) ============
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
