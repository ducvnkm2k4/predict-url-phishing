from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pandas as pd

def train_knn(data_train, data_test):
    # Tách đặc trưng và nhãn
    X_train = data_train.drop(columns=['label'])
    y_train = data_train['label']
    
    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    best_model = KNeighborsClassifier(n_neighbors=3)
    best_model.fit(X_train_scaled,y_train)
    # Dự đoán
    y_pred = best_model.predict(X_test_scaled)
    
    # Đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,digits=4)
    
    # Lưu model & scaler
    dump(best_model, "src/model/model/knn.pkl")
    
    # In kết quả
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print("\n📌 Ma trận nhầm lẫn:\n", matrix)
    print("\n📊 Báo cáo phân loại:\n", report)
    with open("src/model/report/metrics_report_knn.txt", "w", encoding="utf-8") as f:
        f.write("------------------K-nearest neighbor-----------------------")
        f.write(f"✅ Độ chính xác trên tập test: {accuracy:.4f}\n\n")
        f.write("📌 Ma trận nhầm lẫn:\n")
        f.write(str(matrix) + "\n\n")
        f.write("📊 Báo cáo phân loại:\n")
        f.write(report)

if __name__ == "__main__":
    data_train = pd.read_csv('src/data_processing/feature/data_train_scaled.csv')
    data_test = pd.read_csv('src/data_processing/feature/data_test_scaled.csv')
    train_knn(data_train, data_test)
