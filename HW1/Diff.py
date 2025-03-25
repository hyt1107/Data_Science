import pandas as pd

def parse_data(file_path):
    """ 讀取 FP-Growth 輸出文件，解析為字典格式 """
    result = {}
    with open(file_path, "r") as file:
        for line in file:
            if ":" in line:
                pattern, support = line.strip().split(":")
                result[pattern] = float(support)
    return result

def compare_fp_results(file1, file2):
    """ 比對兩個 FP-Growth 結果文件 """
    set_1 = parse_data(file1)
    set_2 = parse_data(file2)
    
    differences = []
    all_keys = set(set_1.keys()).union(set_2.keys())
    
    for key in all_keys:
        val_1 = set_1.get(key, None)
        val_2 = set_2.get(key, None)
        if val_1 != val_2:
            differences.append((key, val_1, val_2))
    
    df_differences = pd.DataFrame(differences, columns=["Pattern", "File1 Support", "File2 Support"])
    return df_differences

# 測試方法
file1 = "colab_large.txt"  # 第一個 FP-Growth 結果文件
file2 = "test_large.txt"  # 第二個 FP-Growth 結果文件

differences_df = compare_fp_results(file1, file2)
print(differences_df)  # 或者使用 GUI 來顯示差異
