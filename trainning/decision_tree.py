from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from joblib import dump

# 1️⃣ Đọc dữ liệu từ file CSV
data_train = pd.read_csv('data_processing/feature/data_train.csv')
data_test = pd.read_csv('data_processing/feature/data_test.csv')

# 2️⃣ Tách feature và label
X_train = data_train.drop(columns=['label'])
y_train = data_train['label']
X_test = data_test.drop(columns=['label'])
y_test = data_test['label']

# 3️⃣ Khởi tạo mô hình Decision Tree
model = DecisionTreeClassifier(max_depth=15)

# 4️⃣ Xác định danh sách tham số cần tìm
param_grid = {
    'max_depth': list(range(10, 41, 1))  # max_depth từ 10 đến 40, bước nhảy 5
}
# best: max_depth=15(đã chạy)
# 5️⃣ Dùng GridSearchCV để tìm giá trị max_depth tốt nhất
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
model.fit(X_train, y_train)

# 6️⃣ Lấy mô hình tốt nhất
#best_model = grid_search.best_estimator_

# 7️⃣ Dự đoán trên tập test
y_pred = model.predict(X_test)

# 8️⃣ Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 🔹 Lưu mô hình tốt nhất
dump(model, "model/decision_tree_best.pkl")

# 🔹 In kết quả
#print(f"Best max_depth: {grid_search.best_params_['max_depth']}")
print(f"Best Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
