from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from joblib import dump

def train_random_forest(data_train, data_test, is_find_best_model=False):
    # Tách đặc trưng (X) và nhãn (y)
    X_train = data_train.drop(columns=['label'])  # Loại bỏ cột label để lấy đặc trưng
    y_train = data_train['label']

    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']
    # Khởi tạo mô hình Random Forest với các tham số mặc định
    model = RandomForestClassifier(n_estimators=300, max_depth=15,random_state=42, n_jobs=-1)
    
    # Huấn luyện mô hình
    print("🚀 Đang huấn luyện RandomForest...")
    model.fit(X_train, y_train)

    # Dự đoán trên tập test
    y_pred = model.predict(X_test)
    
    # Lưu mô hình
    dump(model, "src/model/model/random_forest.pkl")
        
    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # In kết quả
    print(f"✅ Độ chính xác: {accuracy:.4f}")
    print("\n📌 Ma trận nhầm lẫn:\n", conf_matrix)
    print("\n📊 Báo cáo phân loại:\n", class_report)

    with open("src/model/report/metrics_report_random_forest.txt", "w", encoding="utf-8") as f:
        f.write("------------------random forest-----------------------")
        f.write(f"✅ Độ chính xác trên tập test: {accuracy:.4f}\n\n")
        f.write("📌 Ma trận nhầm lẫn:\n")
        f.write(str(conf_matrix) + "\n\n")
        f.write("📊 Báo cáo phân loại:\n")
        f.write(class_report)

if __name__ == "__main__":
    # Load dữ liệu
    data_train = pd.read_csv('src/data_processing/feature/data_train.csv')
    data_test = pd.read_csv('src/data_processing/feature/data_test.csv')
    # Gọi hàm với is_find_best_model=True để tìm tham số tốt nhất
    train_random_forest(data_train, data_test)
