import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

df = pd.read_csv('results/features/sheffield_nai_scores.csv').dropna(subset=['NAI'])
ctrl = df[df['group'] == 'CTRL']

# Fit age → NAI on CTRL
reg = LinearRegression().fit(ctrl[['age']], ctrl['NAI'])
df['NAI_adj'] = df['NAI'] - reg.predict(df[['age']])

y = (df['group'] == 'ASD').astype(int).values
print(f'Unadjusted AUC: {roc_auc_score(y, df["NAI"]):.4f}')
print(f'Age-adjusted AUC: {roc_auc_score(y, df["NAI_adj"]):.4f}')
print(f'Age-NAI correlation (CTRL): {reg.coef_[0]:.4f}')
