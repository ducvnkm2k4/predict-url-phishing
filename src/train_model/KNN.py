from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pandas as pd
import os
def train_knn(data_train, data_test):
    os.makedirs('src/output/model',exist_ok=True)
    os.makedirs('src/output/report',exist_ok=True)
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
    dump(best_model, "src/output/model/knn.pkl")
    
    # In kết quả
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print("\n📌 Ma trận nhầm lẫn:\n", matrix)
    print("\n📊 Báo cáo phân loại:\n", report)
    with open("src/output/report/metrics_report_knn.txt", "w", encoding="utf-8") as f:
        f.write("------------------K-nearest neighbor-----------------------")
        f.write(f"✅ Độ chính xác trên tập test: {accuracy:.4f}\n\n")
        f.write("📌 Ma trận nhầm lẫn:\n")
        f.write(str(matrix) + "\n\n")
        f.write("📊 Báo cáo phân loại:\n")
        f.write(report)

if __name__ == "__main__":
    data_train = pd.read_csv('src/output/data/data_train.csv')
    data_test = pd.read_csv('src/output/data/data_test.csv')
    train_knn(data_train, data_test)
