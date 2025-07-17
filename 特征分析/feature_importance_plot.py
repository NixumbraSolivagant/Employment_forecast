import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 读取特征重要性
imp_path = '../data/lgbm_selected_orig_features.txt'
imp_df = pd.read_csv(imp_path)

# 读取字段注释
comment_path = '../data/字段注释.csv'
comment_df = pd.read_csv(comment_path)

# 建立字段到注释的映射
field2comment = dict(zip(comment_df['字段'], comment_df['注释']))

# 映射注释，没有注释的用字段名
imp_df['注释'] = imp_df['feature'].map(lambda x: field2comment.get(x, x))

# 按重要性降序
imp_df = imp_df.sort_values('importance', ascending=False)

# 设置中文字体（适配不同系统）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
txt_path="../data/lgbm_selected_orig.txt"
imp_df.to_csv(txt_path, index=False, encoding='utf-8-sig')
# 画图
plt.figure(figsize=(10, 8))
sns.barplot(data=imp_df, y='注释', x='importance', palette='viridis')
plt.xlabel('特征重要性')
plt.ylabel('特征（字段注释）')
plt.title('LightGBM特征重要性')
plt.tight_layout()
plt.savefig('../data/feature_importance.png', dpi=200)
plt.show() 