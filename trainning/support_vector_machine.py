from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
import pandas as pd

def train_svm(data_train, data_test, is_find_best_model=False):
    # Tách đặc trưng (X) và nhãn (y)
    X_train = data_train.drop(columns=['label'])
    y_train = data_train['label']
    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']


    if is_find_best_model:
        print("🚀 Đang tìm tham số tốt nhất cho SVM...")

        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01, 0.001],
            'kernel': ['rbf', 'linear']
        }

        grid_search = GridSearchCV(
            estimator=SVC(),
            param_grid=param_grid,
            cv=3,
            n_jobs=7,
            verbose=2,
            scoring='accuracy'
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"✅ Tham số tốt nhất: {grid_search.best_params_}")
        print(f"✅ Độ chính xác CV cao nhất: {grid_search.best_score_:.4f}")
    else:
        print("🚀 Đang huấn luyện SVM với tham số mặc định...")
        best_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        best_model.fit(X_train, y_train)

    # Dự đoán
    y_pred = best_model.predict(X_test)

    # Đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Lưu mô hình
    dump(best_model, "model/svm_best.pkl")

    # In kết quả
    print(f"✅ Độ chính xác trên tập test: {accuracy:.4f}")
    print("\n📌 Ma trận nhầm lẫn:\n", conf_matrix)
    print("\n📊 Báo cáo phân loại:\n", class_report)
    with open("model/metrics_report.txt", "w", encoding="utf-8") as f:
        f.write("------------------svm-----------------------")
        f.write(f"✅ Độ chính xác trên tập test: {accuracy:.4f}\n\n")
        f.write("📌 Ma trận nhầm lẫn:\n")
        f.write(str(conf_matrix) + "\n\n")
        f.write("📊 Báo cáo phân loại:\n")
        f.write(class_report)

# # Load dữ liệu
# data_train = pd.read_csv('dataset/feature/data_train.csv')
# data_test = pd.read_csv('dataset/feature/data_test.csv')
# # Gọi hàm huấn luyện (bật True để tìm tham số)
# train_svm(data_train, data_test, is_find_best_model=True)
