from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from joblib import dump
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np

def train_random_forest(data_train, data_test):
    # Tách đặc trưng và nhãn
    X_train = data_train.drop(columns=['label'])
    y_train = data_train['label']

    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']

    # Khởi tạo mô hình
    model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)

    # Huấn luyện mô hình
    print("🚀 Đang huấn luyện RandomForest...")
    model.fit(X_train, y_train)

    # Dự đoán trên tập test
    y_pred = model.predict(X_test)

    # Lưu mô hình định dạng joblib
    dump(model, "src/model/model/random_forest.pkl")

    # ✅ Chuyển đổi và lưu mô hình dưới dạng ONNX
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open("src/model/model/random_forest.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, digits=4)

    # In kết quả
    print(f"✅ Độ chính xác: {accuracy:.4f}")
    print("\n📌 Ma trận nhầm lẫn:\n", conf_matrix)
    print("\n📊 Báo cáo phân loại:\n", class_report)

    # Ghi vào file
    with open("src/model/report/metrics_report_random_forest.txt", "w", encoding="utf-8") as f:
        f.write("------------------random forest-----------------------\n")
        f.write(f"✅ Độ chính xác trên tập test: {accuracy:.4f}\n\n")
        f.write("📌 Ma trận nhầm lẫn:\n")
        f.write(str(conf_matrix) + "\n\n")
        f.write("📊 Báo cáo phân loại:\n")
        f.write(class_report)

if __name__ == "__main__":
    # Load dữ liệu huấn luyện và kiểm tra
    data_train = pd.read_csv('src/data_processing/feature/data_train.csv')
    data_test = pd.read_csv('src/data_processing/feature/data_test.csv')

    # Gọi hàm huấn luyện
    train_random_forest(data_train, data_test)
