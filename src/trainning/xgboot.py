import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from joblib import dump

def train_xgboost(data_train, data_test, is_find_best_model=False):
    # Tách đặc trưng (X) và nhãn (y)
    X_train = data_train.drop(columns=['label'])
    y_train = data_train['label']
    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']

    if is_find_best_model:
        print("🚀 Đang tìm tham số tốt nhất cho XGBoost với 7 nhân...")

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

        grid_search = GridSearchCV(
            estimator=base_model,
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
        print("🚀 Đang huấn luyện XGBoost với tham số mặc định...")
        best_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            # max_depth=10,
            # subsample=0.8,
            # use_label_encoder=False,
            # eval_metric='logloss',
            random_state=42,
        )
        best_model.fit(X_train, y_train)

    # Dự đoán
    y_pred = best_model.predict(X_test)

    # Đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Lưu mô hình
    dump(best_model, "model/model/xgboost.pkl")


    # In kết quả
    print(f"✅ Độ chính xác trên tập test: {accuracy:.4f}")
    print("\n📌 Ma trận nhầm lẫn:\n", conf_matrix)
    print("\n📊 Báo cáo phân loại:\n", class_report)
    with open("model/report/metrics_report_xgboot.txt", "w", encoding="utf-8") as f:

        f.write("------------------xgboot-----------------------")
        f.write(f"✅ Độ chính xác trên tập test: {accuracy:.4f}\n\n")
        f.write("📌 Ma trận nhầm lẫn:\n")
        f.write(str(conf_matrix) + "\n\n")
        f.write("📊 Báo cáo phân loại:\n")
        f.write(class_report)

if __name__ == "__main__":
    # Load dữ liệu
    data_train = pd.read_csv('data_processing/feature/data_train.csv')
    data_test = pd.read_csv('data_processing/feature/data_test.csv')
    # Gọi hàm để huấn luyện
    train_xgboost(data_train, data_test)

