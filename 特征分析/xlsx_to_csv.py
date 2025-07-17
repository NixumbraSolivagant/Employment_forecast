import pandas as pd
import os

# 输入文件路径
input_file = '../data/题目一的附件.xls'
output_dir = '../data/'

# 读取所有sheet名称
excel_file = pd.ExcelFile(input_file)
sheet_names = excel_file.sheet_names

# 遍历每个sheet并保存为csv
for sheet_name in sheet_names:
    df = pd.read_excel(input_file, sheet_name=sheet_name)
    # 构建输出文件名
    output_file = os.path.join(output_dir, f'{sheet_name}.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

print('转换完成！') 