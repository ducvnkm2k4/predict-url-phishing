from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from joblib import dump

def train_decision_tree(data_train, data_test, is_find_best_model=False):
    # Tách đặc trưng (X) và nhãn (y)
    X_train = data_train.drop(columns=['label'])
    y_train = data_train['label']
    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']

    if is_find_best_model:
        print("🚀 Đang tìm tham số tốt nhất cho Decision Tree...")
        param_grid = {
            'max_depth': list(range(10, 40, 1)),
        }

        base_model = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        print(f"✅ Tham số tốt nhất: {grid_search.best_params_}")
        print(f"✅ Độ chính xác cross-validation cao nhất: {grid_search.best_score_:.4f}")
        with open("src/model/report/metrics_report_decision_tree.txt", "w", encoding="utf-8") as f:
            f.write("------------------decision tree-----------------------\n")
            f.write(f"✅ Tham số tốt nhất: {grid_search.best_params_}\n")
    else:
        # Nếu không tìm mô hình tốt nhất thì dùng tham số mặc định
        print("🚀 Đang huấn luyện Decision Tree với tham số mặc định...")
        best_model = DecisionTreeClassifier(
            max_depth=15, random_state=42
        )
        best_model.fit(X_train, y_train)

    # Dự đoán trên tập test
    y_pred = best_model.predict(X_test)
    # Dự đoán trên tập test
    y_pred = best_model.predict(X_test)

    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    # Lưu mô hình
    dump(best_model, "src/model/model/decision_tree.pkl")

    # In kết quả
    print(f"✅ Độ chính xác trên tập test: {accuracy:.4f}")
    print("\n📌 Ma trận nhầm lẫn:\n", matrix)
    print("\n📊 Báo cáo phân loại:\n", report)
    # Lưu vào file
    with open("src/model/report/metrics_report_decision_tree.txt", "w", encoding="utf-8") as f:
        f.write(f"✅ Độ chính xác trên tập test: {accuracy:.4f}\n\n")
        f.write("📌 Ma trận nhầm lẫn:\n")
        f.write(str(matrix) + "\n\n")
        f.write("📊 Báo cáo phân loại:\n")
        f.write(report)

if __name__ == "__main__":
    # # Load dữ liệu
    data_train = pd.read_csv('src/data_processing/feature/data_train.csv')
    data_test = pd.read_csv('src/data_processing/feature/data_test.csv')
    # Gọi hàm (True = tìm model tốt nhất, False = chạy với default)
    train_decision_tree(data_train, data_test, is_find_best_model=True)
