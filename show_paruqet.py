import pandas as pd

# 1. 读取 Parquet 文件
file_path = "/root/data/searchR1_processed_direct/train.parquet"  # 替换为你的文件路径
df = pd.read_parquet(file_path)

# 2. 打印前五条记录
print("前 5 条记录内容如下：\n")

for i, row in df.head(5).iterrows():
    print(f"--- 第 {i + 1} 条记录 ---")
    for col in df.columns:
        print(f"{col}: {row[col]}")
    print()
