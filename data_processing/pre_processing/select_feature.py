import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
data = pd.read_csv("dataset/feature/data_train.csv")

# Giả sử cột cuối cùng là nhãn (y), các cột còn lại là feature (X)
X = data.drop(columns=['label']) 
y = data['label']

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Lấy giá trị quan trọng của feature
feature_importance = rf.feature_importances_

# Chuyển thành DataFrame để dễ xem
feat_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance
})

# Sắp xếp theo mức độ quan trọng
feat_importance_df = feat_importance_df.sort_values(by="Importance", ascending=False)

# Vẽ biểu đồ
plt.figure(figsize=(10, 5))
plt.barh(feat_importance_df["Feature"][:10], feat_importance_df["Importance"][:10])
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features")
plt.show()
