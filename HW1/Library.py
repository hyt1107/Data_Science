import sys
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

# 讀取輸入參數
if len(sys.argv) != 4:
    print("Usage: python script.py [min_support] [input_file] [output_file]")
    sys.exit(1)

min_support = float(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]

# 讀取交易資料
transactions = []
with open(input_file, "r") as file:
    for line in file:
        transaction = list(map(int, line.strip().split(",")))
        transactions.append(transaction)

# 轉換成 pandas DataFrame
all_items = sorted({item for transaction in transactions for item in transaction})
one_hot_encoded = pd.DataFrame([{item: (item in transaction) for item in all_items} for transaction in transactions])

# 使用 FP-Growth 挖掘頻繁項目集
frequent_itemsets = fpgrowth(one_hot_encoded, min_support=min_support, use_colnames=True)

# 格式化輸出
with open(output_file, "w") as file:
    for _, row in frequent_itemsets.iterrows():
        items = ",".join(map(str, sorted(row["itemsets"])))
        file.write(f"{items}:{row['support']:.4f}\n")
