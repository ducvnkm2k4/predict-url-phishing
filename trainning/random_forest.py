from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from joblib import dump
# Load dữ liệu
data_train = pd.read_csv('dataset/feature/data_train.csv')
data_test = pd.read_csv('dataset/feature/data_test.csv')

# Tách đặc trưng (X) và nhãn (y)
X_train = data_train.drop(columns=['label'])  # Loại bỏ cột label để lấy đặc trưng
y_train = data_train['label']

X_test = data_test.drop(columns=['label'])
y_test = data_test['label']

# Khởi tạo mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

# Huấn luyện mô hình
print("🚀 Đang huấn luyện RandomForest...")
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


dump(model,"model/random_forest.pkl")
# In kết quả
print(f"✅ Độ chính xác: {accuracy:.4f}")
print("\n📌 Ma trận nhầm lẫn:\n", conf_matrix)
print("\n📊 Báo cáo phân loại:\n", class_report)
