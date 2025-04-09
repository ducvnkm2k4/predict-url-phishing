from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
import pandas as pd

def train_knn(data_train, data_test, k_max=15):
    # Tách đặc trưng và nhãn
    X_train = data_train.drop(columns=['label'])
    y_train = data_train['label']
    
    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']
    
    
    # GridSearchCV để tìm giá trị k tối ưu
    param_grid = {
        'n_neighbors': list(range(2, k_max + 1))  # từ 2 đến k_max
    }
    model = KNeighborsClassifier(n_jobs=-1)
    
    print(f"🔍 Đang tìm k tối ưu từ 2 đến {k_max}...")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_k = grid_search.best_params_['n_neighbors']
    
    # Dự đoán
    y_pred = best_model.predict(X_test)
    
    # Đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Lưu model & scaler
    dump(best_model, "model/knn_best.pkl")
    
    # In kết quả
    print(f"✅ Best k: {best_k}")
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print("\n📌 Ma trận nhầm lẫn:\n", matrix)
    print("\n📊 Báo cáo phân loại:\n", report)
    with open("model/metrics_report.txt", "w", encoding="utf-8") as f:
        f.write("------------------K-nearest neighbor-----------------------")
        f.write(f"✅ Độ chính xác trên tập test: {accuracy:.4f}\n\n")
        f.write("📌 Ma trận nhầm lẫn:\n")
        f.write(str(matrix) + "\n\n")
        f.write("📊 Báo cáo phân loại:\n")
        f.write(report)

# Ví dụ dùng:
# data_train = pd.read_csv('data_processing/feature/data_train.csv')
# data_test = pd.read_csv('data_processing/feature/data_test.csv')
# train_knn_gridsearch(data_train, data_test, k_max=20)
