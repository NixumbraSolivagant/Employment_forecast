import pandas as pd
import numpy as np
import os
import shutil

# 读取数据，第二行是字段名
input_file = '../data/数据集.csv'
df = pd.read_csv(input_file, skiprows=1, dtype=str)  # 先全部读为str

# 统一缺失值标记
df = df.replace({'\\N': np.nan})

# 剔除缺失值大于50%的特征
missing_ratio = df.isna().mean(axis=0)
keep_cols = missing_ratio[missing_ratio <= 0.5].index.tolist()
df = df[keep_cols]

# 自动识别数值型特征
numeric_cols = []
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
        numeric_cols.append(col)
    except Exception:
        pass

# 数值型特征用均值填补
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())
# 对c_acc028（失业注销时间）这一列，缺失值填补为0
if 'c_acc028' in df.columns:
    df['c_acc028'] = df['c_acc028'].fillna('0')

# 分类型/文本型特征用空字符串填补
non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
for col in non_numeric_cols:
    df[col] = df[col].fillna('')


# 删除缺失值比例大于80%的行
row_missing_ratio = df.isna().mean(axis=1)
df = df[row_missing_ratio <= 0.8]

# 备份原数据
backup_dir = '../data/backup/'
os.makedirs(backup_dir, exist_ok=True)
shutil.copy2(input_file, os.path.join(backup_dir, '数据集.csv'))

# 用清洗后的数据覆盖原数据
df.to_csv(input_file, index=False, encoding='utf-8-sig')
print('清洗完成，原数据已备份到 data/backup/，清洗结果已覆盖原数据。')
