import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Đọc dữ liệu từ file CSV
df = pd.read_csv("Iris.csv")

# Ánh xạ nhãn Species (string) sang số nguyên
species_map = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}
df["Species"] = df["Species"].map(species_map)

# Tách đặc trưng và nhãn
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
y = df["Species"].values

# Chia train/test để đánh giá
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Huấn luyện SVM
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Đánh giá
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Lưu mô hình
joblib.dump(model, "svm_model.pkl")
print("Model saved!")
