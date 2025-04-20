# BTL Khai phá dữ liệu và máy học trong an toàn hệ thống

## Tổng quan
Dự án này tập trung vào việc phát hiện domain độc hại sử dụng các kỹ thuật khai phá dữ liệu và máy học. Hệ thống phân tích các đặc trưng của URL để xác định các domain có khả năng gây hại cho người dùng.

## Các tính năng chính
- Trích xuất đặc trưng từ URL (độ dài, ký tự đặc biệt, từ khóa, v.v.)
- Tính toán xác suất xuất hiện ký tự
- So sánh với danh sách domain uy tín (Tranco top 100k)
- Huấn luyện và đánh giá nhiều mô hình máy học khác nhau:
  - Decision Tree
  - XGBoost
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Support Vector Machine (SVM)

## Cấu trúc dự án
- `src/`
  - `data_processing/`: Xử lý dữ liệu và trích xuất đặc trưng
  - `dataset/`: Chứa dữ liệu đầu vào
  - `model/`: Lưu trữ các mô hình đã huấn luyện
  - `trainning/`: Các mô hình máy học
  - `report_dataset/`: Báo cáo và đánh giá kết quả

## Quy trình xử lý
1. Merge và tiền xử lý dữ liệu
2. Tính toán xác suất ký tự
3. Trích xuất đặc trưng từ URL
4. Huấn luyện và đánh giá các mô hình
5. Lưu trữ kết quả và báo cáo

## Kết quả huấn luyện
Dự án đã huấn luyện và đánh giá 6 mô hình máy học khác nhau trên tập dữ liệu test với 19,648 mẫu. Kết quả chi tiết như sau:

### 1. Random Forest
- Độ chính xác: 94.13%
- Precision: 94.43% (class 0), 93.85% (class 1)
- Recall: 93.58% (class 0), 94.66% (class 1)
- F1-score: 94.01% (class 0), 94.25% (class 1)

### 2. Support Vector Machine (SVM)
- Độ chính xác: 93.63%
- Precision: 94.86% (class 0), 92.51% (class 1)
- Recall: 92.03% (class 0), 95.17% (class 1)
- F1-score: 93.42% (class 0), 93.82% (class 1)

### 3. XGBoost
- Độ chính xác: 93.32%
- Precision: 95.82% (class 0), 91.16% (class 1)
- Recall: 90.35% (class 0), 96.19% (class 1)
- F1-score: 93.00% (class 0), 93.60% (class 1)

### 4. Logistic Regression
- Độ chính xác: 90.79%
- Precision: 95.58% (class 0), 87.05% (class 1)
- Recall: 85.21% (class 0), 96.19% (class 1)
- F1-score: 90.09% (class 0), 91.39% (class 1)

### 5. Decision Tree
- Độ chính xác: 89.46%
- Precision: 88.08% (class 0), 90.88% (class 1)
- Recall: 90.86% (class 0), 88.11% (class 1)
- F1-score: 89.45% (class 0), 89.47% (class 1)

### 6. K-Nearest Neighbors (KNN)
- Độ chính xác: 89.24%
- Precision: 89.24% (class 0), 89.24% (class 1)
- Recall: 88.82% (class 0), 89.64% (class 1)
- F1-score: 89.03% (class 0), 89.44% (class 1)

## Phân tích kết quả
1. Random Forest cho kết quả tốt nhất với độ chính xác 94.13% và cân bằng tốt giữa precision và recall.
2. SVM và XGBoost cũng cho kết quả rất tốt với độ chính xác lần lượt là 93.63% và 93.32%.
3. Logistic Regression đạt độ chính xác 90.79% với precision cao cho class 0.
4. Decision Tree và KNN có hiệu suất thấp hơn nhưng vẫn đạt độ chính xác trên 89%.
5. Tất cả các mô hình đều cho thấy khả năng phát hiện domain độc hại tốt, với F1-score trên 89% cho cả hai lớp.

