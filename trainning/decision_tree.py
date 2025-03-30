from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from joblib import dump

# 1️⃣ Đọc dữ liệu từ file CSV
data_train = pd.read_csv('dataset/feature/data_train_processed.csv')
data_test = pd.read_csv('dataset/feature/data_test.csv')

# 2️⃣ Tách feature và label
X_train = data_train.drop(columns=['label'])
y_train = data_train['label']
X_test = data_test.drop(columns=['label'])
y_test = data_test['label']

# 3️⃣ Khởi tạo mô hình Decision Tree
model = DecisionTreeClassifier()

# 6️⃣ Huấn luyện với tập training
model.fit(X_train, y_train)


# 8️⃣ Dự đoán trên tập test
y_pred = model.predict(X_test)

# 9️⃣ Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 🔹 Lưu mô hình tốt nhất
dump(model, "model/decision_tree_best.pkl")

# 🔹 In kết quả
print(f"Best Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
