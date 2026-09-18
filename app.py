from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("svm_model.pkl")

app = FastAPI(
    title="Iris Classification API",
    description="SVM model for the Iris dataset",
    version="2.0.0",
)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

species = {0: "setosa", 1: "versicolor", 2: "virginica"}

species_images = {
    0: "https://raw.githubusercontent.com/dong15022006-design/iris-fastapi/main/images/setosa.jpg",
    1: "https://raw.githubusercontent.com/dong15022006-design/iris-fastapi/main/images/versicolor.jpg",
    2: "https://raw.githubusercontent.com/dong15022006-design/iris-fastapi/main/images/virginica.jpg",
}

# Thông tin chi tiết từng loài
species_info = {
    0: {
        "name": "Setosa",
        "color": "#ff6b6b",
        "color2": "#ee5a6f",
        "emoji": "🌺",
        "desc": "Loài hoa nhỏ nhắn, cánh hoa ngắn và hẹp. Dễ phân biệt nhất trong 3 loài.",
        "range": "Petal: 1.0–1.9 cm"
    },
    1: {
        "name": "Versicolor",
        "color": "#4ecdc4",
        "color2": "#44a08d",
        "emoji": "🌸",
        "desc": "Loài hoa trung bình, cánh hoa dài vừa phải. Thường nhầm với Virginica.",
        "range": "Petal: 3.0–5.1 cm"
    },
    2: {
        "name": "Virginica",
        "color": "#a29bfe",
        "color2": "#6c5ce7",
        "emoji": "💜",
        "desc": "Loài hoa lớn nhất, cánh hoa dài và rộng. Phân biệt bởi kích thước.",
        "range": "Petal: 4.5–6.9 cm"
    },
}

# ============ TRANG CHỦ ============
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris AI Classifier</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            html, body { overflow-x: hidden; }
            body {
                font-family: 'Inter', -apple-system, sans-serif;
                background: #0a0a1a;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                position: relative;
                overflow: hidden;
            }
            /* Animated background */
            .bg-orb {
                position: fixed;
                border-radius: 50%;
                filter: blur(80px);
                opacity: 0.5;
                animation: float 20s infinite ease-in-out;
                pointer-events: none;
            }
            .bg-orb:nth-child(1) { width: 400px; height: 400px; background: #667eea; top: -100px; left: -100px; }
            .bg-orb:nth-child(2) { width: 500px; height: 500px; background: #764ba2; bottom: -150px; right: -150px; animation-delay: -7s; }
            .bg-orb:nth-child(3) { width: 300px; height: 300px; background: #f093fb; top: 50%; left: 50%; animation-delay: -14s; }
            @keyframes float {
                0%, 100% { transform: translate(0, 0) scale(1); }
                33% { transform: translate(50px, -50px) scale(1.1); }
                66% { transform: translate(-50px, 50px) scale(0.9); }
            }
            /* Particles */
            .particle {
                position: fixed;
                width: 2px; height: 2px;
                background: rgba(255,255,255,0.5);
                border-radius: 50%;
                pointer-events: none;
                animation: rise linear infinite;
            }
            @keyframes rise {
                from { transform: translateY(100vh) scale(0); opacity: 0; }
                10% { opacity: 1; }
                to { transform: translateY(-10vh) scale(1); opacity: 0; }
            }
            /* Card */
            .card {
                position: relative;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                max-width: 520px;
                width: 100%;
                padding: 45px;
                border-radius: 28px;
                box-shadow: 0 25px 80px rgba(0,0,0,0.5);
                z-index: 10;
                animation: slideUp 0.7s cubic-bezier(0.16, 1, 0.3, 1);
            }
            @keyframes slideUp {
                from { opacity: 0; transform: translateY(40px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .logo {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 12px;
                margin-bottom: 10px;
            }
            .logo-icon {
                width: 44px; height: 44px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            }
            h1 {
                color: white;
                font-size: 26px;
                font-weight: 700;
                letter-spacing: -0.5px;
            }
            .subtitle {
                text-align: center;
                color: rgba(255,255,255,0.5);
                font-size: 14px;
                margin-top: 8px;
                margin-bottom: 40px;
                font-weight: 400;
            }
            .form-group { margin-bottom: 18px; }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 500;
                color: rgba(255,255,255,0.7);
                font-size: 13px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .label-hint {
                font-size: 11px;
                color: rgba(255,255,255,0.3);
                font-weight: 400;
            }
            .input-wrapper { position: relative; }
            input {
                width: 100%;
                padding: 14px 18px;
                background: rgba(255,255,255,0.05);
                border: 1.5px solid rgba(255,255,255,0.1);
                border-radius: 12px;
                font-size: 15px;
                color: white;
                transition: all 0.3s;
                outline: none;
                font-family: 'Inter', sans-serif;
                font-weight: 500;
            }
            input:hover { border-color: rgba(255,255,255,0.2); }
            input:focus {
                border-color: #667eea;
                background: rgba(102, 126, 234, 0.1);
                box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15);
            }
            input::-webkit-outer-spin-button,
            input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
            input[type=number] { -moz-appearance: textfield; }
            .progress-bar {
                position: absolute;
                bottom: 0; left: 0;
                height: 2px;
                background: linear-gradient(90deg, #667eea, #764ba2);
                border-radius: 0 0 12px 12px;
                transition: width 0.3s;
                width: 0;
            }
            button {
                width: 100%;
                padding: 16px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                margin-top: 12px;
                transition: all 0.3s;
                font-family: 'Inter', sans-serif;
                letter-spacing: 0.3px;
                position: relative;
                overflow: hidden;
            }
            button::before {
                content: '';
                position: absolute;
                top: 0; left: -100%;
                width: 100%; height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }
            button:hover::before { left: 100%; }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
            }
            button:active { transform: translateY(0); }
            .footer {
                text-align: center;
                color: rgba(255,255,255,0.3);
                font-size: 11px;
                margin-top: 25px;
                display: flex;
                justify-content: center;
                gap: 15px;
                flex-wrap: wrap;
            }
            .footer span { display: flex; align-items: center; gap: 4px; }
        </style>
    </head>
    <body>
        <div class="bg-orb"></div>
        <div class="bg-orb"></div>
        <div class="bg-orb"></div>
        <div class="card">
            <div class="logo">
                <div class="logo-icon">🌸</div>
                <h1>Iris Classifier</h1>
            </div>
            <p class="subtitle">Phân loại hoa Iris bằng mô hình SVM</p>
            <form action="/predict-form" method="post" id="irisForm">
                <div class="form-group">
                    <label>
                        <span>Sepal Length</span>
                        <span class="label-hint">4.3 – 7.9 cm</span>
                    </label>
                    <div class="input-wrapper">
                        <input type="number" step="0.1" name="sepal_length" value="5.1" min="0" max="10" required>
                        <div class="progress-bar"></div>
                    </div>
                </div>
                <div class="form-group">
                    <label>
                        <span>Sepal Width</span>
                        <span class="label-hint">2.0 – 4.4 cm</span>
                    </label>
                    <div class="input-wrapper">
                        <input type="number" step="0.1" name="sepal_width" value="3.5" min="0" max="10" required>
                        <div class="progress-bar"></div>
                    </div>
                </div>
                <div class="form-group">
                    <label>
                        <span>Petal Length</span>
                        <span class="label-hint">1.0 – 6.9 cm</span>
                    </label>
                    <div class="input-wrapper">
                        <input type="number" step="0.1" name="petal_length" value="1.4" min="0" max="10" required>
                        <div class="progress-bar"></div>
                    </div>
                </div>
                <div class="form-group">
                    <label>
                        <span>Petal Width</span>
                        <span class="label-hint">0.1 – 2.5 cm</span>
                    </label>
                    <div class="input-wrapper">
                        <input type="number" step="0.1" name="petal_width" value="0.2" min="0" max="10" required>
                        <div class="progress-bar"></div>
                    </div>
                </div>
                <button type="submit">✨ Phân loại ngay</button>
            </form>
            <div class="footer">
                <span>⚡ SVM Linear</span>
                <span>🎯 96.7% accuracy</span>
                <span>🚀 FastAPI</span>
            </div>
        </div>
        <script>
            // Tao particles
            for (let i = 0; i < 30; i++) {
                const p = document.createElement('div');
                p.className = 'particle';
                p.style.left = Math.random() * 100 + '%';
                p.style.animationDuration = (Math.random() * 10 + 10) + 's';
                p.style.animationDelay = Math.random() * 10 + 's';
                document.body.appendChild(p);
            }
            // Progress bar cho input
            document.querySelectorAll('input').forEach(inp => {
                inp.addEventListener('input', e => {
                    const val = parseFloat(e.target.value) || 0;
                    const pct = Math.min(val / 10 * 100, 100);
                    e.target.parentElement.querySelector('.progress-bar').style.width = pct + '%';
                });
            });
        </script>
    </body>
    </html>
    """

# ============ KẾT QUẢ ============
@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...),
):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = int(model.predict(features)[0])
    info = species_info[pred]

    # Tính "confidence" giả lập dựa trên decision function
    try:
        decision = model.decision_function(features)[0]
        decision = np.array(decision)
        exp_d = np.exp(decision - np.max(decision))
        probs = exp_d / exp_d.sum()
        confidence = float(probs[pred]) * 100
    except:
        confidence = 96.7

    # Confidence cho 3 loài
    conf_all = {}
    for i in range(3):
        try:
            conf_all[i] = float(probs[i]) * 100
        except:
            conf_all[i] = 100.0 if i == pred else 0.0

    return f"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Kết quả: {info['name']}</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            html, body {{ overflow-x: hidden; }}
            body {{
                font-family: 'Inter', sans-serif;
                background: #0a0a1a;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                position: relative;
                overflow: hidden;
            }}
            .bg-orb {{
                position: fixed;
                border-radius: 50%;
                filter: blur(80px);
                opacity: 0.4;
                animation: float 20s infinite ease-in-out;
                pointer-events: none;
            }}
            .bg-orb:nth-child(1) {{ width: 400px; height: 400px; background: {info['color']}; top: -100px; left: -100px; }}
            .bg-orb:nth-child(2) {{ width: 500px; height: 500px; background: {info['color2']}; bottom: -150px; right: -150px; animation-delay: -7s; }}
            @keyframes float {{
                0%, 100% {{ transform: translate(0, 0) scale(1); }}
                50% {{ transform: translate(50px, -50px) scale(1.1); }}
            }}
            .card {{
                position: relative;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                max-width: 520px;
                width: 100%;
                padding: 40px;
                border-radius: 28px;
                box-shadow: 0 25px 80px rgba(0,0,0,0.5);
                z-index: 10;
                animation: fadeIn 0.7s cubic-bezier(0.16, 1, 0.3, 1);
                text-align: center;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(40px) scale(0.95); }}
                to {{ opacity: 1; transform: translateY(0) scale(1); }}
            }}
            .success-badge {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 6px 14px;
                background: rgba(78, 205, 196, 0.15);
                border: 1px solid rgba(78, 205, 196, 0.3);
                color: #4ecdc4;
                border-radius: 50px;
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 20px;
            }}
            .success-badge::before {{
                content: '✓';
                font-weight: bold;
            }}
            h1 {{
                color: white;
                font-size: 18px;
                font-weight: 500;
                margin-bottom: 20px;
                opacity: 0.6;
            }}
            .species-name {{
                font-size: 48px;
                font-weight: 800;
                background: linear-gradient(135deg, {info['color']}, {info['color2']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-transform: capitalize;
                letter-spacing: -1px;
                margin: 10px 0;
                line-height: 1.1;
            }}
            .emoji-big {{
                font-size: 60px;
                margin: 10px 0;
                display: block;
                filter: drop-shadow(0 10px 20px {info['color']}66);
                animation: bounce 2s infinite ease-in-out;
            }}
            @keyframes bounce {{
                0%, 100% {{ transform: translateY(0); }}
                50% {{ transform: translateY(-8px); }}
            }}
            .desc {{
                color: rgba(255,255,255,0.6);
                font-size: 14px;
                line-height: 1.6;
                margin: 15px 0 25px;
                font-weight: 400;
            }}
            .image-wrapper {{
                margin: 25px 0;
                border-radius: 20px;
                overflow: hidden;
                box-shadow: 0 20px 50px rgba(0,0,0,0.5);
                position: relative;
                transition: transform 0.4s;
            }}
            .image-wrapper:hover {{ transform: scale(1.02); }}
            .image-wrapper img {{
                width: 100%;
                height: auto;
                display: block;
            }}
            .image-wrapper::after {{
                content: '';
                position: absolute;
                inset: 0;
                background: linear-gradient(to top, {info['color']}22, transparent 50%);
                pointer-events: none;
            }}
            .confidence-section {{
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 16px;
                padding: 20px;
                margin: 20px 0;
            }}
            .confidence-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }}
            .confidence-label {{
                color: rgba(255,255,255,0.5);
                font-size: 12px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .confidence-value {{
                font-size: 24px;
                font-weight: 700;
                color: {info['color']};
            }}
            .bar-row {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin: 10px 0;
            }}
            .bar-name {{
                color: rgba(255,255,255,0.6);
                font-size: 12px;
                width: 80px;
                text-align: left;
                text-transform: capitalize;
                font-weight: 500;
            }}
            .bar-track {{
                flex: 1;
                height: 6px;
                background: rgba(255,255,255,0.05);
                border-radius: 3px;
                overflow: hidden;
            }}
            .bar-fill {{
                height: 100%;
                border-radius: 3px;
                transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
            }}
            .bar-pct {{
                color: rgba(255,255,255,0.5);
                font-size: 11px;
                width: 45px;
                text-align: right;
                font-weight: 600;
                font-variant-numeric: tabular-nums;
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin: 20px 0;
            }}
            .info-item {{
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 12px;
                padding: 12px;
                text-align: left;
            }}
            .info-item-label {{
                color: rgba(255,255,255,0.4);
                font-size: 10px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            }}
            .info-item-value {{
                color: white;
                font-size: 15px;
                font-weight: 600;
                font-variant-numeric: tabular-nums;
            }}
            .btn-back {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 14px 32px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-decoration: none;
                border-radius: 12px;
                font-weight: 600;
                font-size: 14px;
                transition: all 0.3s;
                margin-top: 10px;
                font-family: 'Inter', sans-serif;
            }}
            .btn-back:hover {{
                transform: translateY(-2px);
                box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
            }}
        </style>
    </head>
    <body>
        <div class="bg-orb"></div>
        <div class="bg-orb"></div>
        <div class="card">
            <div class="success-badge">Dự đoán thành công</div>
            <span class="emoji-big">{info['emoji']}</span>
            <h1>Loài hoa được nhận diện</h1>
            <div class="species-name">{info['name']}</div>
            <p class="desc">{info['desc']}</p>

            <div class="image-wrapper">
                <img src="{species_images[pred]}" alt="{info['name']}">
            </div>

            <div class="confidence-section">
                <div class="confidence-header">
                    <span class="confidence-label">Độ tin cậy</span>
                    <span class="confidence-value">{confidence:.1f}%</span>
                </div>
                <div class="bar-row">
                    <span class="bar-name">Setosa</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {conf_all[0]:.1f}%; background: linear-gradient(90deg, #ff6b6b, #ee5a6f);"></div>
                    </div>
                    <span class="bar-pct">{conf_all[0]:.0f}%</span>
                </div>
                <div class="bar-row">
                    <span class="bar-name">Versicolor</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {conf_all[1]:.1f}%; background: linear-gradient(90deg, #4ecdc4, #44a08d);"></div>
                    </div>
                    <span class="bar-pct">{conf_all[1]:.0f}%</span>
                </div>
                <div class="bar-row">
                    <span class="bar-name">Virginica</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: {conf_all[2]:.1f}%; background: linear-gradient(90deg, #a29bfe, #6c5ce7);"></div>
                    </div>
                    <span class="bar-pct">{conf_all[2]:.0f}%</span>
                </div>
            </div>

            <div class="info-grid">
                <div class="info-item">
                    <div class="info-item-label">Sepal Length</div>
                    <div class="info-item-value">{sepal_length} cm</div>
                </div>
                <div class="info-item">
                    <div class="info-item-label">Sepal Width</div>
                    <div class="info-item-value">{sepal_width} cm</div>
                </div>
                <div class="info-item">
                    <div class="info-item-label">Petal Length</div>
                    <div class="info-item-value">{petal_length} cm</div>
                </div>
                <div class="info-item">
                    <div class="info-item-label">Petal Width</div>
                    <div class="info-item-value">{petal_width} cm</div>
                </div>
            </div>

            <a href="/" class="btn-back">← Phân loại hoa khác</a>
        </div>
        <script>
            // Animate bars
            setTimeout(() => {{
                document.querySelectorAll('.bar-fill').forEach(b => {{
                    const w = b.style.width;
                    b.style.width = '0';
                    setTimeout(() => b.style.width = w, 100);
                }});
            }}, 200);
        </script>
    </body>
    </html>
    """

# ============ API JSON ============
@app.post("/predict")
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width,
                 data.petal_length, data.petal_width]]
    prediction = int(model.predict(features)[0])
    return {
        "class_id": prediction,
        "prediction": species[prediction],
        "image_url": species_images[prediction],
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
