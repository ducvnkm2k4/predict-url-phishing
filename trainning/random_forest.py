from sklearn.ensemble import RandomForestClassifier
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

    # Nếu is_find_best_model là True, sử dụng GridSearchCV để tìm tham số tốt nhất
    if is_find_best_model:
        print("🚀 Đang tìm tham số tốt nhất cho RandomForest với 7 nhân...")
        param_grid = {
            'n_estimators': [100, 150, 200, 250,300],  # Số cây
            'max_depth': [10, 15, 20],  # Độ sâu tối đa
            'bootstrap': [True, False]  # Sử dụng bootstrap hay không
        }
        
        # Khởi tạo mô hình Random Forest
        rf = RandomForestClassifier(random_state=42, n_jobs=7)
        
        # Khởi tạo GridSearchCV
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                                   cv=5, n_jobs=7, verbose=2, scoring='accuracy')
        
        # Tiến hành tìm kiếm tham số tốt nhất
        grid_search.fit(X_train, y_train)
        
        # In kết quả tham số tốt nhất
        print(f"✅ Tham số tốt nhất: {grid_search.best_params_}")
        print(f"✅ Độ chính xác tốt nhất: {grid_search.best_score_:.4f}")
        
        # Dự đoán trên tập test với mô hình tốt nhất
        y_pred = grid_search.best_estimator_.predict(X_test)
    else:
        # Khởi tạo mô hình Random Forest với các tham số mặc định
        model = RandomForestClassifier(n_estimators=200, max_depth=15,
                                       min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
        
        # Huấn luyện mô hình
        print("🚀 Đang huấn luyện RandomForest...")
        model.fit(X_train, y_train)

        # Dự đoán trên tập test
        y_pred = model.predict(X_test)
        
        # Lưu mô hình
        dump(model, "model/random_forest.pkl")
        
    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # In kết quả
    print(f"✅ Độ chính xác: {accuracy:.4f}")
    print("\n📌 Ma trận nhầm lẫn:\n", conf_matrix)
    print("\n📊 Báo cáo phân loại:\n", class_report)

    with open("model/metrics_report.txt", "w", encoding="utf-8") as f:
        f.write("------------------random forest-----------------------")
        f.write(f"✅ Độ chính xác trên tập test: {accuracy:.4f}\n\n")
        f.write("📌 Ma trận nhầm lẫn:\n")
        f.write(str(conf_matrix) + "\n\n")
        f.write("📊 Báo cáo phân loại:\n")
        f.write(class_report)

# # Load dữ liệu
# data_train = pd.read_csv('data_processing/feature/data_train_scaled.csv')
# data_test = pd.read_csv('data_processing/feature/data_test_scaled.csv')
# # Gọi hàm với is_find_best_model=True để tìm tham số tốt nhất
# train_random_forest(data_train, data_test)
