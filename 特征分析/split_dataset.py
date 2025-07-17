import pandas as pd
import os

# 定义输入输出路径
input_file = '../data/数据集.csv'
output_dir = '../data/'

# 读取前三行（元信息）
with open(input_file, 'r', encoding='utf-8-sig') as f:
    meta_lines = [next(f) for _ in range(3)]

# 读取数据部分（跳过前三行，不指定列名）
df = pd.read_csv(input_file, skiprows=3, header=0)

# 用AX列（第50列，索引49）判断
ax_col = df.columns[49]
# 尝试将AX列转为时间，能转成功的为就业
ax_time = pd.to_datetime(df[ax_col], errors='coerce')
employed_df = df[ax_time.notna()]
unemployed_df = df[ax_time.isna()]
# 保存为两个新的CSV文件，并在顶部加回前三行
employed_output_path = os.path.join(output_dir, '就业.csv')
unemployed_output_path = os.path.join(output_dir, '失业.csv')

# 写入就业数据
with open(employed_output_path, 'w', encoding='utf-8-sig') as f:
    f.writelines(meta_lines)
    employed_df.to_csv(f, index=False, encoding='utf-8-sig')

# 写入失业数据
with open(unemployed_output_path, 'w', encoding='utf-8-sig') as f:
    f.writelines(meta_lines)
    unemployed_df.to_csv(f, index=False, encoding='utf-8-sig')

print(f"处理完成！")
print(f"就业数据已保存至: {employed_output_path}")
print(f"失业数据已保存至: {unemployed_output_path}")
print(f"就业人数: {len(employed_df)}")
print(f"失业人数: {len(unemployed_df)}") 