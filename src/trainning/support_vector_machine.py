from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pandas as pd

def train_svm(data_train, data_test):
    # Tách đặc trưng (X) và nhãn (y)
    X_train = data_train.drop(columns=['label'])
    y_train = data_train['label']
    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("🚀 Đang huấn luyện SVM với tham số mặc định...")
    best_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    best_model.fit(X_train_scaled, y_train)

    # Dự đoán
    y_pred = best_model.predict(X_test_scaled)

    # Đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred,digits=4)

    # Lưu mô hình
    dump(best_model, "src/model/model/svm.pkl")

    # In kết quả
    print(f"✅ Độ chính xác trên tập test: {accuracy:.4f}")
    print("\n📌 Ma trận nhầm lẫn:\n", conf_matrix)
    print("\n📊 Báo cáo phân loại:\n", class_report)
    with open("src/model/report/metrics_report_svm.txt", "w", encoding="utf-8") as f:
        f.write("------------------svm-----------------------")
        f.write(f"✅ Độ chính xác trên tập test: {accuracy:.4f}\n\n")
        f.write("📌 Ma trận nhầm lẫn:\n")
        f.write(str(conf_matrix) + "\n\n")
        f.write("📊 Báo cáo phân loại:\n")
        f.write(class_report)
if __name__ == "__main__":
    # Load dữ liệu
    data_train = pd.read_csv('src/data_processing/feature/data_train_scaled.csv')
    data_test = pd.read_csv('src/data_processing/feature/data_test_scaled.csv')
    # Gọi hàm huấn luyện (bật True để tìm tham số)
    train_svm(data_train, data_test)
