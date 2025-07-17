import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

# 读取数据，第二行作为字段名
input_file = '../data/数据集.csv'
df = pd.read_csv(input_file, skiprows=1)

# 构造目标变量
ax_col = df.columns[49]
df['target'] = pd.to_datetime(df[ax_col], errors='coerce').notna().astype(int)

# 打印实际字段名
print('实际字段名:', df.columns.tolist())

# 你要用的原始特征字段（第二行字段名，自动与数据表字段取交集）
manual_features = [
    'sex','birthday','age','nation','marriage','edu_level','politic','reg_address','profession','religion','c_aac009','c_aab299','c_aac010','c_aac011','c_aac180','c_aac181','c_aac182','c_aac183','type','military_status','is_disability','is_teen','is_elder','change_type','is_living_alone','live_status','note','b_acc030','b_aab001','b_acc031','b_acc033','b_acc034','b_aae030','b_aae031','b_aab022','b_aab004','c_acc02e','c_acc020','c_ajc090','c_ajc093','c_aca111','c_aca112','c_aac013','c_acc026','c_acc027','c_acc03b','c_acc0m3','c_acc023','c_aab004','acc02y'
]
exclude = ['people_id', 'name', 'target']
features = [col for col in manual_features if col in df.columns and col not in exclude]

# 如果特征为空，自动用所有字段（排除exclude）
if not features:
    print('警告：手动特征列表与数据表无交集，自动用所有字段（排除exclude）！')
    features = [col for col in df.columns if col not in exclude]

X = df[features].copy()
y = df['target']
print('最终用于建模的特征:', features)
print('X shape:', X.shape)

print('填充数值型缺失值...')
for col in tqdm(X.select_dtypes(include='number').columns):
    X[col] = X[col].fillna(X[col].mean())
print('填充分类型缺失值...')
for col in tqdm(X.select_dtypes(include='object').columns):
    X[col] = X[col].fillna(X[col].mode().iloc[0])

print('进行独热编码...')
X_dummies = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dummies)

print('用LightGBM（CPU）进行特征重要性筛选...')
num_threads = os.cpu_count()
lgb_train = lgb.Dataset(X_scaled, y)
params = {
    'objective': 'binary',
    'device': 'cpu',
    'num_threads': num_threads,
    'verbosity': -1
}
gbm = lgb.train(params, lgb_train, num_boost_round=100)

# 获取特征重要性和特征名
importances = gbm.feature_importance()
feature_names = X_dummies.columns

# 统计每个原始字段的总重要性（最长前缀匹配）
def match_orig_field(col, orig_fields):
    for field in sorted(orig_fields, key=len, reverse=True):
        if col.startswith(field):
            return field
    return col

orig_fields = features  # 实际用于建模的原始字段名
orig_importance = {}
for fname, imp in zip(feature_names, importances):
    orig = match_orig_field(fname, orig_fields)
    orig_importance[orig] = orig_importance.get(orig, 0) + imp

# 输出被选中的原始特征
importances_df = pd.DataFrame(list(orig_importance.items()), columns=['feature', 'importance'])
importances_df = importances_df[importances_df['importance'] > 0]
importances_df = importances_df.sort_values('importance', ascending=False)
print('被LightGBM选中的原始特征：', list(importances_df['feature']))

# 保存结果到文件
txt_path = '../data/lgbm_selected_orig_features.txt'
importances_df.to_csv(txt_path, index=False, encoding='utf-8-sig')
print(f'原始特征及重要性已保存至: {txt_path}') 