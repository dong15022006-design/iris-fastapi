from fastapi import FastAPI, Form, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import secrets
import hashlib
import math

model = joblib.load("svm_model.pkl")

app = FastAPI(
    title="Iris Classification API",
    description="SVM model for the Iris dataset",
    version="3.0.0",
)

# ============ CẤU HÌNH ĐĂNG NHẬP ============
# ĐỔI MẬT KHẨU TẠI ĐÂY
USERS = {
    "admin":     hashlib.sha256("Iris@2026".encode()).hexdigest(),
    "giangdong": hashlib.sha256("GiangDong@2026".encode()).hexdigest(),
}
SESSIONS = {}
SESSION_COOKIE = "iris_session"

def get_current_user(session_token: Optional[str]) -> Optional[str]:
    if not session_token:
        return None
    return SESSIONS.get(session_token)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ============ DỮ LIỆU ============
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

species_info = {
    0: {
        "name": "Setosa",
        "color": "#ff6b6b",
        "color2": "#ee5a6f",
        "emoji": "🌺",
        "desc": "Loài hoa nhỏ nhắn, cánh hoa ngắn và hẹp. Dễ phân biệt nhất trong 3 loài.",
        "avg": [5.01, 3.42, 1.46, 0.24]
    },
    1: {
        "name": "Versicolor",
        "color": "#4ecdc4",
        "color2": "#44a08d",
        "emoji": "🌸",
        "desc": "Loài hoa trung bình, cánh hoa dài vừa phải. Thường nhầm với Virginica.",
        "avg": [5.94, 2.77, 4.26, 1.33]
    },
    2: {
        "name": "Virginica",
        "color": "#a29bfe",
        "color2": "#6c5ce7",
        "emoji": "🌷",
        "desc": "Loài hoa lớn nhất, cánh hoa dài và rộng. Phân biệt bởi kích thước.",
        "avg": [6.59, 2.97, 5.55, 2.03]
    },
}


# ============ TRANG ĐĂNG NHẬP ============
@app.get("/login", response_class=HTMLResponse)
def login_page(error: str = ""):
    error_html = ""
    if error == "1":
        error_html = '<div class="error">❌ Sai tên đăng nhập hoặc mật khẩu</div>'
    elif error == "2":
        error_html = '<div class="error">⚠️ Vui lòng đăng nhập để tiếp tục</div>'

    return f"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Đăng nhập - Iris Classifier</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
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
                opacity: 0.5;
                animation: float 20s infinite ease-in-out;
                pointer-events: none;
            }}
            .bg-orb:nth-child(1) {{ width: 400px; height: 400px; background: #667eea; top: -100px; left: -100px; }}
            .bg-orb:nth-child(2) {{ width: 500px; height: 500px; background: #764ba2; bottom: -150px; right: -150px; animation-delay: -7s; }}
            @keyframes float {{
                0%, 100% {{ transform: translate(0, 0) scale(1); }}
                50% {{ transform: translate(50px, -50px) scale(1.1); }}
            }}
            .card {{
                background: rgba(255,255,255,0.05);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid rgba(255,255,255,0.1);
                max-width: 420px;
                width: 100%;
                padding: 45px;
                border-radius: 28px;
                box-shadow: 0 25px 80px rgba(0,0,0,0.5);
                z-index: 10;
                animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
            }}
            @keyframes slideUp {{
                from {{ opacity: 0; transform: translateY(40px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .logo {{
                width: 60px; height: 60px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                border-radius: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 30px;
                margin: 0 auto 20px;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
            }}
            h1 {{
                color: white;
                font-size: 24px;
                font-weight: 700;
                text-align: center;
                margin-bottom: 8px;
            }}
            .subtitle {{
                color: rgba(255,255,255,0.5);
                font-size: 14px;
                text-align: center;
                margin-bottom: 30px;
            }}
            .error {{
                background: rgba(255, 107, 107, 0.15);
                border: 1px solid rgba(255, 107, 107, 0.3);
                color: #ff6b6b;
                padding: 12px 16px;
                border-radius: 10px;
                font-size: 13px;
                margin-bottom: 20px;
                text-align: center;
            }}
            .form-group {{ margin-bottom: 18px; }}
            label {{
                display: block;
                color: rgba(255,255,255,0.7);
                font-size: 13px;
                font-weight: 500;
                margin-bottom: 8px;
            }}
            input {{
                width: 100%;
                padding: 14px 18px;
                background: rgba(255,255,255,0.05);
                border: 1.5px solid rgba(255,255,255,0.1);
                border-radius: 12px;
                font-size: 15px;
                color: white;
                outline: none;
                transition: all 0.3s;
                font-family: 'Inter', sans-serif;
            }}
            input:focus {{
                border-color: #667eea;
                background: rgba(102, 126, 234, 0.1);
                box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15);
            }}
            input::placeholder {{ color: rgba(255,255,255,0.3); }}
            button {{
                width: 100%;
                padding: 16px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                margin-top: 10px;
                transition: all 0.3s;
                font-family: 'Inter', sans-serif;
            }}
            button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
            }}
        </style>
    </head>
    <body>
        <div class="bg-orb"></div>
        <div class="bg-orb"></div>
        <div class="card">
            <div class="logo">🔐</div>
            <h1>Đăng nhập</h1>
            <p class="subtitle">Iris Classifier — SVM FastAPI</p>
            {error_html}
            <form action="/login" method="post">
                <div class="form-group">
                    <label>Tên đăng nhập</label>
                    <input type="text" name="username" placeholder="Nhập username" required autofocus>
                </div>
                <div class="form-group">
                    <label>Mật khẩu</label>
                    <input type="password" name="password" placeholder="Nhập password" required>
                </div>
                <button type="submit">Đăng nhập →</button>
            </form>
        </div>
    </body>
    </html>
    """


# ============ XỬ LÝ ĐĂNG NHẬP ============
@app.post("/login")
def login(
    username: str = Form(...),
    password: str = Form(...),
):
    if username in USERS and USERS[username] == hash_password(password):
        token = secrets.token_urlsafe(32)
        SESSIONS[token] = username
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(
            key=SESSION_COOKIE,
            value=token,
            httponly=True,
            max_age=3600,
            samesite="lax",
        )
        return response
    else:
        return RedirectResponse(url="/login?error=1", status_code=303)


# ============ ĐĂNG XUẤT ============
@app.get("/logout")
def logout(session: Optional[str] = Cookie(None, alias=SESSION_COOKIE)):
    if session and session in SESSIONS:
        del SESSIONS[session]
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


# ============ TRANG CHỦ ============
@app.get("/", response_class=HTMLResponse)
def home(session: Optional[str] = Cookie(None, alias=SESSION_COOKIE)):
    user = get_current_user(session)
    if not user:
        return RedirectResponse(url="/login?error=2", status_code=303)

    return f"""
    <!DOCTYPE html>
    <html lang="vi" data-theme="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Iris AI Classifier</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: #0a0a1a;
                --card-bg: rgba(255, 255, 255, 0.05);
                --card-border: rgba(255, 255, 255, 0.1);
                --text: #ffffff;
                --text-dim: rgba(255, 255, 255, 0.5);
                --text-dim2: rgba(255, 255, 255, 0.3);
                --input-bg: rgba(255, 255, 255, 0.05);
                --input-border: rgba(255, 255, 255, 0.1);
                --primary: #667eea;
                --secondary: #764ba2;
            }}
            [data-theme="light"] {{
                --bg: #f5f7fb;
                --card-bg: rgba(255, 255, 255, 0.85);
                --card-border: rgba(0, 0, 0, 0.08);
                --text: #1a1a2e;
                --text-dim: rgba(0, 0, 0, 0.5);
                --text-dim2: rgba(0, 0, 0, 0.3);
                --input-bg: rgba(255, 255, 255, 0.9);
                --input-border: rgba(0, 0, 0, 0.1);
            }}
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            html, body {{ overflow-x: hidden; }}
            body {{
                font-family: 'Inter', -apple-system, sans-serif;
                background: var(--bg);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                position: relative;
                overflow: hidden;
                transition: background 0.4s;
            }}
            .bg-orb {{
                position: fixed;
                border-radius: 50%;
                filter: blur(80px);
                opacity: 0.5;
                animation: float 20s infinite ease-in-out;
                pointer-events: none;
            }}
            .bg-orb:nth-child(1) {{ width: 400px; height: 400px; background: #667eea; top: -100px; left: -100px; }}
            .bg-orb:nth-child(2) {{ width: 500px; height: 500px; background: #764ba2; bottom: -150px; right: -150px; animation-delay: -7s; }}
            .bg-orb:nth-child(3) {{ width: 300px; height: 300px; background: #f093fb; top: 50%; left: 50%; animation-delay: -14s; }}
            @keyframes float {{
                0%, 100% {{ transform: translate(0, 0) scale(1); }}
                33% {{ transform: translate(50px, -50px) scale(1.1); }}
                66% {{ transform: translate(-50px, 50px) scale(0.9); }}
            }}
            .particle {{
                position: fixed;
                width: 2px; height: 2px;
                background: rgba(255,255,255,0.5);
                border-radius: 50%;
                pointer-events: none;
                animation: rise linear infinite;
            }}
            @keyframes rise {{
                from {{ transform: translateY(100vh) scale(0); opacity: 0; }}
                10% {{ opacity: 1; }}
                to {{ transform: translateY(-10vh) scale(1); opacity: 0; }}
            }}
            .card {{
                position: relative;
                background: var(--card-bg);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid var(--card-border);
                max-width: 520px;
                width: 100%;
                padding: 45px;
                border-radius: 28px;
                box-shadow: 0 25px 80px rgba(0,0,0,0.5);
                z-index: 10;
                animation: slideUp 0.7s cubic-bezier(0.16, 1, 0.3, 1);
            }}
            @keyframes slideUp {{
                from {{ opacity: 0; transform: translateY(40px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .top-bar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 8px;
                margin-bottom: 15px;
            }}
            .user-info {{
                display: flex;
                align-items: center;
                gap: 8px;
                color: var(--text-dim);
                font-size: 12px;
            }}
            .avatar {{
                width: 28px; height: 28px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                color: white;
                font-weight: 700;
            }}
            .top-actions {{
                display: flex;
                gap: 8px;
                align-items: center;
            }}
            .icon-btn {{
                width: 36px; height: 36px;
                border-radius: 10px;
                border: 1px solid var(--card-border);
                background: var(--input-bg);
                color: var(--text);
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                transition: all 0.3s;
                text-decoration: none;
            }}
            .icon-btn:hover {{
                border-color: var(--primary);
                background: rgba(102, 126, 234, 0.1);
                transform: translateY(-2px);
            }}
            .logout-btn:hover {{
                border-color: #ff6b6b;
                background: rgba(255, 107, 107, 0.1);
            }}
            .logo {{
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 12px;
                margin-bottom: 10px;
            }}
            .logo-icon {{
                width: 44px; height: 44px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            }}
            h1 {{
                color: var(--text);
                font-size: 26px;
                font-weight: 700;
                letter-spacing: -0.5px;
            }}
            .subtitle {{
                text-align: center;
                color: var(--text-dim);
                font-size: 14px;
                margin-top: 8px;
                margin-bottom: 30px;
                font-weight: 400;
            }}
            .species-thumbs {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 8px;
                margin-bottom: 25px;
            }}
            .thumb {{
                border-radius: 12px;
                overflow: hidden;
                aspect-ratio: 1;
                position: relative;
                cursor: pointer;
                transition: transform 0.3s;
                border: 2px solid transparent;
            }}
            .thumb:hover {{
                transform: translateY(-4px);
                border-color: var(--primary);
            }}
            .thumb img {{
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
            }}
            .thumb-label {{
                position: absolute;
                bottom: 0; left: 0; right: 0;
                padding: 4px;
                background: linear-gradient(to top, rgba(0,0,0,0.8), transparent);
                color: white;
                font-size: 10px;
                font-weight: 600;
                text-align: center;
                text-transform: capitalize;
            }}
            .form-group {{ margin-bottom: 18px; }}
            label {{
                display: flex;
                margin-bottom: 8px;
                font-weight: 500;
                color: var(--text-dim);
                font-size: 13px;
                justify-content: space-between;
                align-items: center;
            }}
            .label-hint {{
                font-size: 11px;
                color: var(--text-dim2);
                font-weight: 400;
            }}
            .input-wrapper {{ position: relative; }}
            input {{
                width: 100%;
                padding: 14px 18px;
                background: var(--input-bg);
                border: 1.5px solid var(--input-border);
                border-radius: 12px;
                font-size: 15px;
                color: var(--text);
                transition: all 0.3s;
                outline: none;
                font-family: 'Inter', sans-serif;
                font-weight: 500;
            }}
            input:hover {{ border-color: var(--text-dim2); }}
            input:focus {{
                border-color: var(--primary);
                background: rgba(102, 126, 234, 0.1);
                box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15);
            }}
            input::-webkit-outer-spin-button,
            input::-webkit-inner-spin-button {{ -webkit-appearance: none; margin: 0; }}
            input[type=number] {{ -moz-appearance: textfield; }}
            .progress-bar {{
                position: absolute;
                bottom: 0; left: 0;
                height: 2px;
                background: linear-gradient(90deg, #667eea, #764ba2);
                border-radius: 0 0 12px 12px;
                transition: width 0.3s;
                width: 0;
            }}
            button[type=submit] {{
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
            }}
            button[type=submit]::before {{
                content: '';
                position: absolute;
                top: 0; left: -100%;
                width: 100%; height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }}
            button[type=submit]:hover::before {{ left: 100%; }}
            button[type=submit]:hover {{
                transform: translateY(-2px);
                box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
            }}
            button[type=submit]:active {{ transform: translateY(0); }}
            .action-row {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin-top: 10px;
            }}
            .action-btn {{
                padding: 12px;
                background: var(--input-bg);
                border: 1.5px solid var(--input-border);
                color: var(--text);
                border-radius: 12px;
                font-size: 13px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.3s;
                font-family: 'Inter', sans-serif;
            }}
            .action-btn:hover {{
                border-color: var(--primary);
                background: rgba(102, 126, 234, 0.1);
                transform: translateY(-2px);
            }}
            .history-section {{
                margin-top: 25px;
                padding-top: 20px;
                border-top: 1px solid var(--card-border);
            }}
            .history-label {{
                color: var(--text-dim2);
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .history-clear {{
                color: var(--text-dim2);
                font-size: 10px;
                cursor: pointer;
                text-decoration: underline;
            }}
            .history-chips {{
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
                min-height: 28px;
            }}
            .history-empty {{
                color: var(--text-dim2);
                font-size: 12px;
                font-style: italic;
            }}
            .chip {{
                padding: 5px 12px;
                border-radius: 50px;
                font-size: 11px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
                border: 1px solid;
                font-family: 'Inter', sans-serif;
            }}
            .chip:hover {{ transform: translateY(-2px); }}
            .chip.setosa     {{ background: rgba(255,107,107,0.15); color: #ff6b6b; border-color: rgba(255,107,107,0.3); }}
            .chip.versicolor {{ background: rgba(78,205,196,0.15); color: #4ecdc4; border-color: rgba(78,205,196,0.3); }}
            .chip.virginica  {{ background: rgba(162,155,254,0.15); color: #a29bfe; border-color: rgba(162,155,254,0.3); }}
            .footer {{
                text-align: center;
                color: var(--text-dim2);
                font-size: 11px;
                margin-top: 25px;
                display: flex;
                justify-content: center;
                gap: 15px;
                flex-wrap: wrap;
            }}
            .footer span {{ display: flex; align-items: center; gap: 4px; }}
            .skeleton {{
                position: fixed;
                inset: 0;
                background: var(--bg);
                z-index: 999;
                display: none;
                align-items: center;
                justify-content: center;
                flex-direction: column;
                gap: 20px;
            }}
            .skeleton.active {{ display: flex; }}
            .spinner {{
                width: 50px; height: 50px;
                border: 3px solid var(--input-border);
                border-top-color: var(--primary);
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }}
            @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
            .skeleton-text {{
                color: var(--text-dim);
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="bg-orb"></div>
        <div class="bg-orb"></div>
        <div class="bg-orb"></div>
        <div class="card">
            <div class="top-bar">
                <div class="user-info">
                    <div class="avatar">{user[0].upper()}</div>
                    <span>Xin chào, <b>{user}</b></span>
                </div>
                <div class="top-actions">
                    <button class="icon-btn" id="themeToggle" title="Đổi chế độ sáng/tối">🌙</button>
                    <a href="/logout" class="icon-btn logout-btn" title="Đăng xuất">🚪</a>
                </div>
            </div>
            <div class="logo">
                <div class="logo-icon">🌸</div>
                <h1>Iris Classifier</h1>
            </div>
            <p class="subtitle">Phân loại hoa Iris bằng mô hình SVM</p>

            <div class="species-thumbs">
                <div class="thumb" onclick="loadSample(0)">
                    <img src="https://raw.githubusercontent.com/dong15022006-design/iris-fastapi/main/images/setosa.jpg" alt="Setosa">
                    <div class="thumb-label">Setosa</div>
                </div>
                <div class="thumb" onclick="loadSample(1)">
                    <img src="https://raw.githubusercontent.com/dong15022006-design/iris-fastapi/main/images/versicolor.jpg" alt="Versicolor">
                    <div class="thumb-label">Versicolor</div>
                </div>
                <div class="thumb" onclick="loadSample(2)">
                    <img src="https://raw.githubusercontent.com/dong15022006-design/iris-fastapi/main/images/virginica.jpg" alt="Virginica">
                    <div class="thumb-label">Virginica</div>
                </div>
            </div>

            <form action="/predict-form" method="post" id="irisForm">
                <div class="form-group">
                    <label><span>Sepal Length</span><span class="label-hint">4.3 – 7.9 cm</span></label>
                    <div class="input-wrapper">
                        <input type="number" step="0.1" name="sepal_length" id="sl" value="5.1" min="0" max="10" required>
                        <div class="progress-bar"></div>
                    </div>
                </div>
                <div class="form-group">
                    <label><span>Sepal Width</span><span class="label-hint">2.0 – 4.4 cm</span></label>
                    <div class="input-wrapper">
                        <input type="number" step="0.1" name="sepal_width" id="sw" value="3.5" min="0" max="10" required>
                        <div class="progress-bar"></div>
                    </div>
                </div>
                <div class="form-group">
                    <label><span>Petal Length</span><span class="label-hint">1.0 – 6.9 cm</span></label>
                    <div class="input-wrapper">
                        <input type="number" step="0.1" name="petal_length" id="pl" value="1.4" min="0" max="10" required>
                        <div class="progress-bar"></div>
                    </div>
                </div>
                <div class="form-group">
                    <label><span>Petal Width</span><span class="label-hint">0.1 – 2.5 cm</span></label>
                    <div class="input-wrapper">
                        <input type="number" step="0.1" name="petal_width" id="pw" value="0.2" min="0" max="10" required>
                        <div class="progress-bar"></div>
                    </div>
                </div>
                <button type="submit">✨ Phân loại ngay</button>
                <div class="action-row">
                    <button type="button" class="action-btn" onclick="randomSample()">🎲 Mẫu ngẫu nhiên</button>
                    <button type="button" class="action-btn" onclick="clearForm()">🧹 Xóa form</button>
                </div>
            </form>

            <div class="history-section">
                <div class="history-label">
                    <span>📜 Lịch sử dự đoán</span>
                    <span class="history-clear" onclick="clearHistory()">Xóa</span>
                </div>
                <div class="history-chips" id="historyChips">
                    <span class="history-empty">Chưa có dự đoán nào</span>
                </div>
            </div>

            <div class="footer">
                <span>⚡ SVM Linear</span>
                <span>🎯 96.7% accuracy</span>
                <span>🚀 FastAPI</span>
            </div>
        </div>

        <div class="skeleton" id="loading">
            <div class="spinner"></div>
            <div class="skeleton-text">Đang phân tích...</div>
        </div>

        <script>
            for (let i = 0; i < 30; i++) {{
                const p = document.createElement('div');
                p.className = 'particle';
                p.style.left = Math.random() * 100 + '%';
                p.style.animationDuration = (Math.random() * 10 + 10) + 's';
                p.style.animationDelay = Math.random() * 10 + 's';
                document.body.appendChild(p);
            }}

            document.querySelectorAll('input').forEach(inp => {{
                inp.addEventListener('input', e => {{
                    const val = parseFloat(e.target.value) || 0;
                    const pct = Math.min(val / 10 * 100, 100);
                    e.target.parentElement.querySelector('.progress-bar').style.width = pct + '%';
                }});
            }});

            const themeToggle = document.getElementById('themeToggle');
            const html = document.documentElement;
            const savedTheme = localStorage.getItem('iris-theme') || 'dark';
            html.setAttribute('data-theme', savedTheme);
            themeToggle.textContent = savedTheme === 'dark' ? '🌙' : '☀️';

            themeToggle.addEventListener('click', () => {{
                const current = html.getAttribute('data-theme');
                const next = current === 'dark' ? 'light' : 'dark';
                html.setAttribute('data-theme', next);
                localStorage.setItem('iris-theme', next);
                themeToggle.textContent = next === 'dark' ? '🌙' : '☀️';
            }});

            const SAMPLES = {{
                0: [5.1, 3.5, 1.4, 0.2],
                1: [6.0, 2.7, 5.1, 1.6],
                2: [6.5, 3.0, 5.2, 2.0]
            }};

            function loadSample(sp) {{
                const s = SAMPLES[sp];
                document.getElementById('sl').value = s[0];
                document.getElementById('sw').value = s[1];
                document.getElementById('pl').value = s[2];
                document.getElementById('pw').value = s[3];
                document.querySelectorAll('input').forEach(inp => inp.dispatchEvent(new Event('input')));
            }}

            function randomSample() {{
                const allSamples = [
                    [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
                    [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3],
                    [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5],
                    [5.5, 2.3, 4.0, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
                    [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1],
                    [6.3, 2.9, 5.6, 1.8], [6.5, 3.0, 5.8, 2.2], [7.6, 3.0, 6.6, 2.1],
                ];
                const s = allSamples[Math.floor(Math.random() * allSamples.length)];
                document.getElementById('sl').value = s[0];
                document.getElementById('sw').value = s[1];
                document.getElementById('pl').value = s[2];
                document.getElementById('pw').value = s[3];
                document.querySelectorAll('input').forEach(inp => inp.dispatchEvent(new Event('input')));
            }}

            function clearForm() {{
                document.getElementById('sl').value = '';
                document.getElementById('sw').value = '';
                document.getElementById('pl').value = '';
                document.getElementById('pw').value = '';
                document.querySelectorAll('input').forEach(inp => inp.dispatchEvent(new Event('input')));
            }}

            function getHistory() {{
                try {{ return JSON.parse(localStorage.getItem('iris-history') || '[]'); }}
                catch {{ return []; }}
            }}

            function renderHistory() {{
                const history = getHistory();
                const container = document.getElementById('historyChips');
                if (history.length === 0) {{
                    container.innerHTML = '<span class="history-empty">Chưa có dự đoán nào</span>';
                    return;
                }}
                container.innerHTML = history.map((h, i) => `
                    <div class="chip ${{h.species}}" onclick="loadFromHistory(${{i}})" title="SL=${{h.sl}}, SW=${{h.sw}}, PL=${{h.pl}}, PW=${{h.pw}}">
                        ${{h.species.charAt(0).toUpperCase() + h.species.slice(1)}}
                    </div>
                `).join('');
            }}

            function loadFromHistory(i) {{
                const h = getHistory()[i];
                if (!h) return;
                document.getElementById('sl').value = h.sl;
                document.getElementById('sw').value = h.sw;
                document.getElementById('pl').value = h.pl;
                document.getElementById('pw').value = h.pw;
                document.querySelectorAll('input').forEach(inp => inp.dispatchEvent(new Event('input')));
            }}

            function clearHistory() {{
                localStorage.removeItem('iris-history');
                renderHistory();
            }}

            document.getElementById('irisForm').addEventListener('submit', () => {{
                document.getElementById('loading').classList.add('active');
            }});

            document.addEventListener('keydown', e => {{
                if (e.key === 'Escape') clearForm();
                if (e.key === 'r' && e.ctrlKey) {{ e.preventDefault(); randomSample(); }}
            }});

            renderHistory();
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
    session: Optional[str] = Cookie(None, alias=SESSION_COOKIE),
):
    user = get_current_user(session)
    if not user:
        return RedirectResponse(url="/login?error=2", status_code=303)

    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = int(model.predict(features)[0])
    info = species_info[pred]
    avg = info["avg"]

    try:
        decision = model.decision_function(features)[0]
        decision = np.array(decision)
        exp_d = np.exp(decision - np.max(decision))
        probs = exp_d / exp_d.sum()
        confidence = float(probs[pred]) * 100
    except:
        probs = [0, 0, 0]
        probs[pred] = 1
        confidence = 96.7

    conf_all = {}
    for i in range(3):
        try:
            conf_all[i] = float(probs[i]) * 100
        except:
            conf_all[i] = 100.0 if i == pred else 0.0

    user_vals = [sepal_length, sepal_width, petal_length, petal_width]
    maxes = [7.9, 4.4, 6.9, 2.5]
    radar_user = [v / m * 100 for v, m in zip(user_vals, maxes)]
    radar_avg = [v / m * 100 for v, m in zip(avg, maxes)]

    def radar_points(values, r=80):
        n = len(values)
        pts = []
        for i, v in enumerate(values):
            angle = -math.pi / 2 + (2 * math.pi * i / n)
            x = 100 + r * (v / 100) * math.cos(angle)
            y = 100 + r * (v / 100) * math.sin(angle)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    grid_lines = ""
    for pct in [25, 50, 75, 100]:
        pts = []
        for i in range(4):
            angle = -math.pi / 2 + (2 * math.pi * i / 4)
            x = 100 + 80 * (pct / 100) * math.cos(angle)
            y = 100 + 80 * (pct / 100) * math.sin(angle)
            pts.append(f"{x:.1f},{y:.1f}")
        grid_lines += f'<polygon points="{" ".join(pts)}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>'

    axis_lines = ""
    for i in range(4):
        angle = -math.pi / 2 + (2 * math.pi * i / 4)
        x2 = 100 + 80 * math.cos(angle)
        y2 = 100 + 80 * math.sin(angle)
        axis_lines += f'<line x1="100" y1="100" x2="{x2:.1f}" y2="{y2:.1f}" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>'

    labels_radar = ["SL", "SW", "PL", "PW"]
    labels_svg = ""
    for i, lbl in enumerate(labels_radar):
        angle = -math.pi / 2 + (2 * math.pi * i / 4)
        x = 100 + 95 * math.cos(angle)
        y = 100 + 95 * math.sin(angle)
        labels_svg += f'<text x="{x:.1f}" y="{y:.1f}" fill="rgba(255,255,255,0.5)" font-size="9" text-anchor="middle" dominant-baseline="middle">{lbl}</text>'

    radar_user_pts = radar_points(radar_user)
    radar_avg_pts = radar_points(radar_avg)

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
            body {{
                font-family: 'Inter', sans-serif;
                background: #0a0a1a;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }}
            .card {{
                background: rgba(255,255,255,0.05);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255,255,255,0.1);
                max-width: 520px;
                width: 100%;
                padding: 40px;
                border-radius: 28px;
                box-shadow: 0 25px 80px rgba(0,0,0,0.5);
                text-align: center;
                color: white;
            }}
            .species-name {{
                font-size: 42px;
                font-weight: 800;
                background: linear-gradient(135deg, {info['color']}, {info['color2']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 15px 0;
                text-transform: capitalize;
            }}
            .image-wrapper {{
                margin: 20px 0;
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 15px 40px rgba(0,0,0,0.5);
            }}
            .image-wrapper img {{ width: 100%; display: block; }}
            .confidence-value {{
                font-size: 32px;
                font-weight: 700;
                color: {info['color']};
                margin: 10px 0;
            }}
            .radar-container {{
                background: rgba(255,255,255,0.03);
                border-radius: 16px;
                padding: 15px;
                margin: 20px 0;
            }}
            .btn-back {{
                display: inline-block;
                padding: 14px 32px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-decoration: none;
                border-radius: 12px;
                font-weight: 600;
                margin-top: 15px;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <div style="font-size: 50px;">{info['emoji']}</div>
            <h1 style="font-size: 16px; opacity: 0.6;">Loài hoa được nhận diện</h1>
            <div class="species-name">{info['name']}</div>
            <div class="image-wrapper">
                <img src="{species_images[pred]}" alt="{info['name']}">
            </div>
            <div class="confidence-value">{confidence:.1f}%</div>
            <div style="opacity: 0.5; font-size: 13px;">Độ tin cậy</div>

            <div class="radar-container">
                <div style="opacity: 0.5; font-size: 12px; margin-bottom: 10px;">So sánh với trung bình loài</div>
                <svg viewBox="0 0 200 200" style="width: 100%; max-width: 250px;">
                    {grid_lines}
                    {axis_lines}
                    <polygon points="{radar_avg_pts}" fill="{info['color']}22" stroke="{info['color']}88" stroke-width="1.5" stroke-dasharray="4,3"/>
                    <polygon points="{radar_user_pts}" fill="#667eea55" stroke="#667eea" stroke-width="2"/>
                    {labels_svg}
                </svg>
                <div style="display: flex; gap: 15px; justify-content: center; font-size: 11px; margin-top: 8px;">
                    <span><span style="display: inline-block; width: 12px; height: 2px; background: #667eea; vertical-align: middle;"></span> Input</span>
                    <span><span style="display: inline-block; width: 12px; height: 2px; background: {info['color']}; vertical-align: middle;"></span> Trung bình loài</span>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 20px 0;">
                <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px; text-align: left;">
                    <div style="font-size: 10px; opacity: 0.4; text-transform: uppercase;">Sepal Length</div>
                    <div style="font-weight: 600; font-size: 15px;">{sepal_length} cm</div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px; text-align: left;">
                    <div style="font-size: 10px; opacity: 0.4; text-transform: uppercase;">Sepal Width</div>
                    <div style="font-weight: 600; font-size: 15px;">{sepal_width} cm</div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px; text-align: left;">
                    <div style="font-size: 10px; opacity: 0.4; text-transform: uppercase;">Petal Length</div>
                    <div style="font-weight: 600; font-size: 15px;">{petal_length} cm</div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px; text-align: left;">
                    <div style="font-size: 10px; opacity: 0.4; text-transform: uppercase;">Petal Width</div>
                    <div style="font-weight: 600; font-size: 15px;">{petal_width} cm</div>
                </div>
            </div>

            <a href="/" class="btn-back">← Phân loại hoa khác</a>
        </div>
    </body>
    </html>
    """


# ============ API JSON ============
@app.post("/predict")
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = int(model.predict(features)[0])
    return {
        "class_id": prediction,
        "prediction": species[prediction],
        "image_url": species_images[prediction],
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
