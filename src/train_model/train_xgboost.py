import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from joblib import dump

def train_xgboost(data_train, data_test):
    # Tách đặc trưng (X) và nhãn (y)
    X_train = data_train.drop(columns=['label'])
    y_train = data_train['label']
    X_test = data_test.drop(columns=['label'])
    y_test = data_test['label']

    print("🚀 Đang huấn luyện XGBoost với tham số mặc định...")
    best_model = xgb.XGBClassifier(
        n_estimators=350,
        learning_rate=0.01,
        eval_metric='logloss',
        random_state=42,
    )
    best_model.fit(X_train, y_train)

    # Dự đoán
    y_pred = best_model.predict(X_test)

    # Đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred,digits=4)

    # Lưu mô hình
    dump(best_model, "src/output/model/xgboost.pkl")


    # In kết quả
    print(f"✅ Độ chính xác trên tập test: {accuracy:.4f}")
    print("\n📌 Ma trận nhầm lẫn:\n", conf_matrix)
    print("\n📊 Báo cáo phân loại:\n", class_report)
    with open("src/output/report/metrics_report_xgboot.txt", "w", encoding="utf-8") as f:

        f.write("------------------xgboot-----------------------")
        f.write(f"✅ Độ chính xác trên tập test: {accuracy:.4f}\n\n")
        f.write("📌 Ma trận nhầm lẫn:\n")
        f.write(str(conf_matrix) + "\n\n")
        f.write("📊 Báo cáo phân loại:\n")
        f.write(class_report)

if __name__ == "__main__":
    # Load dữ liệu
    data_train = pd.read_csv("src/output/data/data_train.csv")
    data_test = pd.read_csv("src/output/data/data_test.csv")
    # Gọi hàm để huấn luyện
    train_xgboost(data_train, data_test)

