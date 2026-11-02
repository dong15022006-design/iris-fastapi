# Iris SVM FastAPI 🌸

Hệ thống phân loại hoa Iris sử dụng thuật toán SVM (Support Vector Machine), đóng gói thành REST API bằng FastAPI và triển khai trên Render Cloud.

## 📌 Giới thiệu
Dự án này là bài tập lớn môn Học máy / Trí tuệ nhân tạo. Hệ thống cho phép người dùng nhập các thông số hình thái của hoa (chiều dài/rộng đài hoa và cánh hoa) để dự đoán loài hoa tương ứng (Setosa, Versicolor, Virginica).

- **Độ chính xác:** 96.7%

## 📂 Cấu trúc thư mục
```text
.
├── sample_data/
│   └── iris.csv                # Dữ liệu mẫu dùng để train
├── Procfile                    # Cấu hình để Render biết cách chạy app
├── app.py                      # File chạy chính của FastAPI (chứa API endpoints)
├── render.yaml                 # File cấu hình hạ tầng cho Render
├── requirements.txt            # Danh sách các thư viện cần thiết
├── svm_model.pkl               # Mô hình SVM đã được huấn luyện
├── train.py                    # File code dùng để huấn luyện mô hình
└── README.md                   # File mô tả dự án (file này)
Sinh viên thực hiện: Giang Đông

Lớp: 24CKDL

Trường: Trường Đại học Sư phạm – ĐH Đà Nẵng, Khoa Toán – Tin