import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

def show_feature_distribution(data, feature):
    """Hiển thị 4 biểu đồ cho một đặc trưng trong một cửa sổ duy nhất"""
    fig, axes = plt.subplots(2,2, figsize=(12, 10))  # Chỉ tạo một figure

    print(f'---------- Skewness của {feature} ----------')
    print(data.skew())

    # Histogram
    sns.histplot(data, bins=30, kde=True, ax=axes[0, 0], color="blue")
    axes[0, 0].set_title(f"Histogram của {feature}")
    
    # Boxplot
    sns.boxplot(x=data, ax=axes[0, 1], color="orange")
    axes[0, 1].set_title(f"Boxplot của {feature}")
    
    # QQ Plot
    stats.probplot(data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f"QQ Plot của {feature}")
    
    # Violin Plot
    sns.violinplot(x=data, ax=axes[1, 1], color="green")
    axes[1, 1].set_title(f"Violin Plot của {feature}")

    plt.tight_layout()
    plt.show()

def show_histogram(data, feature_names):
    fig, axes = plt.subplots(5,6, figsize=(16, 12))  # 4 hàng, 5 cột
    
    for i, feature in enumerate(feature_names):
        row, col = i // 6, i % 6  # Đúng công thức: hàng = i // 5, cột = i % 5
        sns.histplot(x=data[feature], bins=30, kde=True, ax=axes[row, col], color="blue")
        # axes[row, col].set_title(f"Histogram của {feature}")
    
    # Ẩn các ô trống nếu feature_names không đủ 20
    for j in range(len(feature_names), 30):
        fig.delaxes(axes.flatten()[j])
    
    plt.tight_layout()
    plt.show()

def show_boxplot(data, feature_names):
    fig, axes = plt.subplots(5, 6, figsize=(16, 12))  # 4 hàng, 5 cột
    
    for i, feature in enumerate(feature_names):
        row, col = i // 6, i % 6  # Đúng công thức: hàng = i // 5, cột = i % 5
        sns.boxplot(data[feature], ax=axes[row, col], color="orange")
        # axes[row, col].set_title(f"Histogram của {feature}")
    
    # Ẩn các ô trống nếu feature_names không đủ 20
    for j in range(len(feature_names), 30):
        fig.delaxes(axes.flatten()[j])
    
    plt.tight_layout()
    plt.show()

def show_probplot(data, feature_names):
    fig, axes = plt.subplots(5, 6, figsize=(16, 12))  # 4 hàng, 5 cột
    
    for i, feature in enumerate(feature_names):
        row, col = i // 6, i % 6  # Đúng công thức: hàng = i // 5, cột = i % 5
        stats.probplot(data[feature], dist="norm", plot=axes[row,col])
        axes[row, col].set_title(f" {feature}")
    
    # Ẩn các ô trống nếu feature_names không đủ 20
    for j in range(len(feature_names), 30):
        fig.delaxes(axes.flatten()[j])
    
    plt.tight_layout()
    plt.show()
# Đọc dữ liệu
data_train = pd.read_csv('src/data_processing/feature/data_train.csv')

feature_names = [
    "length", "tachar", "hasKeyWords", "tahex", 
    "tadigit", "numDots", "countUpcase", "numvo", "numco",
    "maxsub30", "rapath", "haspro",
    "numsdm", "radomain", "tinyUrl", "tanv", 
    "tanco", "tandi", "tansc",
    "domain_len", "ent_char", "eod", "rank", "tld",
    "hasSuspiciousTld", "label"
]
# show_probplot(data_train,feature_names)


feature=feature_names[0]
show_feature_distribution(data_train[feature], feature)
# Hiển thị từng đặc trưng trong một cửa sổ riêng biệt
# for feature in feature_names:
#     show_feature_distribution(data_train[feature], feature)
