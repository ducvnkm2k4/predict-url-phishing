from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from joblib import dump
# 1️⃣ Đọc dữ liệu từ file CSV
data_train = pd.read_csv('dataset/feature/data_train.csv')
data_test = pd.read_csv('dataset/feature/data_test.csv')

# 2️⃣ Tách feature và label (giả sử cột 'label' là nhãn phân loại)
X_train = data_train.drop(columns=['label'])  # Loại bỏ cột label, giữ lại feature
y_train = data_train['label']  # Nhãn của tập train

X_test = data_test.drop(columns=['label'])
y_test = data_test['label']

# 3️⃣ Khởi tạo mô hình Decision Tree
model = DecisionTreeClassifier()

# 4️⃣ Huấn luyện mô hình
model.fit(X_train, y_train)

# 5️⃣ Dự đoán trên tập test
y_pred = model.predict(X_test)

# 6️⃣ Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)  # Độ chính xác
report = classification_report(y_test, y_pred)  # Precision, Recall, F1-score


dump(model,"model/decision_tree.pkl")
# 7️⃣ In kết quả đánh giá
print(f" Accuracy: {accuracy:.4f}")
print(" Classification Report:")
print(report)
