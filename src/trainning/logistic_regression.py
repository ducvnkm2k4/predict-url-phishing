from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
import pandas as pd
import pandas as pd

def train_logistic_regression(data_train, data_test, is_find_best_model=False):
    # Tách đặc trưng và nhãn
    X_train = data_train.drop(columns=['label'])
    y_train = data_train['label']
    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']
def train_logistic_regression(data_train, data_test, is_find_best_model=False):
    # Tách đặc trưng và nhãn
    X_train = data_train.drop(columns=['label'])
    y_train = data_train['label']
    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']

    if is_find_best_model:
        print("🚀 Đang tìm tham số tốt nhất cho Logistic Regression...")

        param_grid = {
            'C': [0.01, 0.1, 1, 10],  # Tham số regularization
            'penalty': ['l1', 'l2', 'elasticnet'],
            'l1_ratio': [0.0, 0.5, 1.0],  # chỉ dùng khi penalty = 'elasticnet'
            'solver': ['saga']
        }

        grid_search = GridSearchCV(
            LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42),
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            verbose=2,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        print(f"✅ Tham số tốt nhất: {grid_search.best_params_}")
        print(f"✅ Độ chính xác CV cao nhất: {grid_search.best_score_:.4f}")
    else:
        print("🚀 Đang huấn luyện Logistic Regression với tham số mặc định...")
        best_model = LogisticRegression(
            max_iter=2000,
            solver='saga',
            random_state=42,
            n_jobs=7
        )
        best_model.fit(X_train, y_train)
    if is_find_best_model:
        print("🚀 Đang tìm tham số tốt nhất cho Logistic Regression...")

        param_grid = {
            'C': [0.01, 0.1, 1, 10],  # Tham số regularization
            'penalty': ['l1', 'l2', 'elasticnet'],
            'l1_ratio': [0.0, 0.5, 1.0],  # chỉ dùng khi penalty = 'elasticnet'
            'solver': ['saga']
        }

        grid_search = GridSearchCV(
            LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42),
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            verbose=2,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        print(f"✅ Tham số tốt nhất: {grid_search.best_params_}")
        print(f"✅ Độ chính xác CV cao nhất: {grid_search.best_score_:.4f}")
    else:
        print("🚀 Đang huấn luyện Logistic Regression với tham số mặc định...")
        best_model = LogisticRegression(
            max_iter=2000,
            solver='saga',
            random_state=42,
            n_jobs=7
        )
        best_model.fit(X_train, y_train)

    # Dự đoán
    y_pred = best_model.predict(X_test)
    # Dự đoán
    y_pred = best_model.predict(X_test)

    # Đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    # Đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Lưu mô hình và scaler
    dump(best_model, "src/model/model/logistic_regression.pkl")

    # In kết quả
    print(f"✅ Độ chính xác trên tập test: {accuracy:.4f}")
    print("\n📌 Ma trận nhầm lẫn:\n", conf_matrix)
    print("\n📊 Báo cáo phân loại:\n", class_report)
    with open("src/model/report/metrics_report_logistic.txt", "w", encoding="utf-8") as f:
        f.write("------------------logistic regression-----------------------")
        f.write(f"✅ Độ chính xác trên tập test: {accuracy:.4f}\n\n")
        f.write("📌 Ma trận nhầm lẫn:\n")
        f.write(str(conf_matrix) + "\n\n")
        f.write("📊 Báo cáo phân loại:\n")
        f.write(class_report)


if __name__ == "__main__":
    # Load dữ liệu
    data_train = pd.read_csv('src/data_processing/feature/data_train_scaled.csv')
    data_test = pd.read_csv('src/data_processing/feature/data_test_scaled.csv')
    # Gọi hàm huấn luyện
    train_logistic_regression(data_train, data_test)
