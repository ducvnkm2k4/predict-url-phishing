from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

# Đọc dữ liệu
data_train = pd.read_csv('dataset/feature/data_train.csv')

# Giả sử cột 'label' là nhãn
X_train = data_train.drop(columns=['label'])  # Loại bỏ cột nhãn, chỉ lấy features
y_train = data_train['label']  # Lấy nhãn

# Chọn 10 đặc trưng quan trọng nhất
selector = SelectKBest(score_func=f_classif, k=35)
X_new = selector.fit_transform(X_train, y_train)

# In điểm quan trọng của từng feature
feature_scores = pd.DataFrame({'Feature': X_train.columns, 'Score': selector.scores_})
feature_scores = feature_scores.sort_values(by="Score", ascending=False)  # Sắp xếp theo điểm quan trọng
print(feature_scores)
