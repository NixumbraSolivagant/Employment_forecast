import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import shap
import matplotlib.pyplot as plt
import os

# 读取重要特征列表
imp_path = '../data/lgbm_selected_orig.txt'
imp_df = pd.read_csv(imp_path)
features = imp_df['feature'].tolist()

# 读取数据
data_path = '../data/数据集.csv'
df = pd.read_csv(data_path, header=0)

# 目标变量（第42列，索引41，时间为就业，0为失业）
col_42 = df.columns[41]
is_time = pd.to_datetime(df[col_42], errors='coerce').notna()
is_zero = (df[col_42].astype(str).str.strip() == '0')
df['target'] = np.where(is_time, 1, 0)
df.loc[is_zero, 'target'] = 0
y = df['target']

# age分层
if 'age' in df.columns:
    bins = [0, 25, 35, 45, 60, 150]
    labels = ['18-25', '26-35', '36-45', '46-60', '60+']
    df['age_group'] = pd.cut(pd.to_numeric(df['age'], errors='coerce'), bins=bins, labels=labels, right=False)
    features = ['age_group' if f == 'age' else f for f in features]


# 只保留分析特征
X = df[features]

# 指定类别特征
cat_features = []
for col in X.columns:
    if col in ['age_group', 'c_aac180', 'c_aab299', 'c_aac182', 'c_aac009', 'c_aac011', 'sex']:
        cat_features.append(col)
    elif X[col].dtype == 'object':
        cat_features.append(col)

# 保证所有cat_features无NaN且为str
for col in cat_features:
    X[col] = X[col].astype(str).fillna('')

# 构建CatBoost数据池
train_pool = Pool(X, label=y, cat_features=cat_features)

# 训练CatBoost模型（GPU加速）
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    task_type='GPU',  # GPU加速
    devices='0',      # 指定GPU编号
    verbose=50
)
model.fit(train_pool)

# SHAP解释
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 创建输出目录
# out_dir = '../data/catboost_shap/'
# os.makedirs(out_dir, exist_ok=True)

# 创建分组统计输出目录
csv_out_dir = '../output/'
os.makedirs(csv_out_dir, exist_ok=True)

# 1. 总体特征重要性
# plt.figure()
# shap.summary_plot(shap_values, X, plot_type='bar', show=False)
# plt.tight_layout()
# plt.savefig(os.path.join(out_dir, 'shap_summary_bar.png'), dpi=200)
# plt.close()

# 2. 针对每个重要特征画SHAP依赖图，并保存分组统计csv
for i, feat in enumerate(features):
    # 当前特征的SHAP值
    if isinstance(shap_values, list):
        # 二分类时shap_values为list，取正类
        shap_col = shap_values[1][:, i]
    else:
        shap_col = shap_values[:, i]
    temp = pd.DataFrame({
        feat: X[feat],
        'target': y,
        'shap': shap_col
    })
    group_stats = temp.groupby(feat).agg(
        样本数=('target', 'count'),
        SHAP均值=('shap', 'mean'),
        就业率=('target', 'mean')
    ).reset_index()
    group_stats['失业率'] = 1 - group_stats['就业率']
    group_stats.to_csv(os.path.join(csv_out_dir, f'feature_{feat}_shap.csv'), index=False, encoding='utf-8-sig')

print(f'分析完成，分组统计已保存至 {csv_out_dir}') 