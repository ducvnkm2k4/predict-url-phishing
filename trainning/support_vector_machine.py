from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Load dữ liệu
data_train = pd.read_csv('dataset/feature/data_train.csv')
data_test = pd.read_csv('dataset/feature/data_test.csv')

# Tách đặc trưng (X) và nhãn (y)
X_train = data_train.drop(columns=['label'])
y_train = data_train['label']

X_test = data_test.drop(columns=['label'])
y_test = data_test['label']

# Chuẩn hóa dữ liệu (quan trọng với SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo mô hình SVM với kernel 'rbf'
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Huấn luyện mô hình
print("🚀 Đang huấn luyện SVM...")
model.fit(X_train_scaled, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test_scaled)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# In kết quả
print(f"✅ Độ chính xác: {accuracy:.4f}")
print("\n📌 Ma trận nhầm lẫn:\n", conf_matrix)
print("\n📊 Báo cáo phân loại:\n", class_report)
