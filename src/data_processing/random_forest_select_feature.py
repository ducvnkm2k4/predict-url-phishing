import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Giả sử bạn đã có một DataFrame với các đặc trưng và nhãn
# Ví dụ: data_train chứa các đặc trưng và labels là cột nhãn
data_train=pd.read_csv('src/data_processing/feature/data_train.csv')
X = data_train.drop('label', axis=1)  # Tất cả các đặc trưng
y = data_train['label']  # Nhãn

# Tạo và huấn luyện mô hình Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Trích xuất các đặc trưng quan trọng
importances = model.feature_importances_

# Tạo DataFrame để dễ dàng theo dõi và sắp xếp
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sắp xếp các đặc trưng theo mức độ quan trọng
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Hiển thị bảng xếp hạng các đặc trưng quan trọng
print(importance_df)

# Vẽ biểu đồ để trực quan hóa các đặc trưng quan trọng
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance from Random Forest')
plt.gca().invert_yaxis()  # Đảo ngược trục y để đặc trưng quan trọng nhất ở trên
plt.show()
