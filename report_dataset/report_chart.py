import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

def show_feature_distribution(data, feature):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Tạo lưới 2x2
    axes = axes.flatten()  # Chuyển mảng về 1D

    # Histogram (phân bố dữ liệu)
    sns.histplot(data[feature], bins=30, kde=True, ax=axes[0], color="blue")
    axes[0].set_title(f"Histogram của {feature}")
    
    # Boxplot (phát hiện outliers)
    sns.boxplot(x=data[feature], ax=axes[1], color="orange")
    axes[1].set_title(f"Boxplot của {feature}")
    
    # QQ Plot (kiểm tra phân phối chuẩn)
    stats.probplot(data[feature], dist="norm", plot=axes[2])
    axes[2].set_title(f"QQ Plot của {feature}")
    
    # Violin Plot (so sánh mật độ dữ liệu)
    sns.violinplot(x=data[feature], ax=axes[3], color="green")
    axes[3].set_title(f"Violin Plot của {feature}")

    plt.tight_layout()  # Tránh chồng chéo nội dung
    plt.show()

# Đọc dữ liệu
data_train = pd.read_csv('data_processing/feature/data_train.csv')

feature_names = [
    "length", "tachar", "hasKeyWords", "hasspecKW", "tahex", "tadigit", 
    "numDots","taslash", "countUpcase", "numvo", "numco", "backslash",
    "maxsub30", "rapath","haspro", "hasExe", "redirect", "hasref",
    "hasIP", "hasport", "numsdm", "radomain","tinyUrl", "tanv", 
    "tanco", "tandi", "tansc", "tanhe", "is_digit",
    "domain_len", "ent_char", "eod", "rank", "tld", "label"
    ]
# Chọn đặc trưng để vẽ
feature = feature_names[10]

'''
    - 0.length: lệch trái
    - 1.tarchar: outlier -> lệch trái
    - 2.hasKeyWords: tạm bỏ 
    - 3.hasspecKW: tạm bỏ
    - 4.tahex: lệch trái
    - 5.tadigit: có ngoại lệ nhỏ và lệch trái 
    - 6.numDots: lệch trái
    - 7.taslash: có ngoại lệ 
    - 8.countUpcase: ổn
    - 9.numvo: ngoại lệ
    - 10.numco: ngoại lệ -> xử lý phân phối(không rõ trái hay phải)
    backslash,
    maxsub30, rapath,haspro, hasExe, redirect, hasref,
    hasIP, hasport, numsdm, radomain,tinyUrl, tanv, 
    tanco, tandi, tansc, tanhe, is_digit,
    domain_len, ent_char, eod, rank, tld, label
'''
# Hiển thị biểu đồ
show_feature_distribution(data_train, feature)
