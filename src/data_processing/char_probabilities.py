import pandas as pd
from collections import Counter

def char_pro():
    df = pd.read_csv("src/dataset/tranco_list/tranco_5897N.csv", header=None)

    print(df.head())

    all_chars = ''.join(df.iloc[:, 1].astype(str))

    char_counts = Counter(all_chars)

    total_chars = sum(char_counts.values())

    # Bước 5: Chuẩn hóa thành xác suất
    char_probabilities = {char: count / total_chars for char, count in char_counts.items()}

    char_prob_df = pd.DataFrame(list(char_probabilities.items()), columns=['Character', 'Probability'])

    char_prob_df.sort_values(by='Character', ascending=True, inplace=True)

    char_prob_df.to_csv("src/dataset/tranco_list/char_probabilities.csv", index=False)

    print(char_prob_df)
    return char_prob_df

if __name__ == "__main__":
    char_pro()
