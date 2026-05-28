# %% [markdown]
# # Preprocessing & EDA
# 
# ## Setup
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fp = "heart_failure_clinical_records_dataset.csv"

if not os.path.exists(fp):
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    
    # Load the latest version
    df_download = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "guriya79/heart-failure-prediction-dataset",
        fp,
    )
    df_download.to_csv(fp, index=False)
    print("Dataset downloaded and saved.")

df = pd.read_csv(fp)
print(df.head())

# %%
df.info()

# %%
print(df.isnull().sum())

# %%
print(df.describe())

# %% [markdown]
# # Exploratory Data Analysis (EDA)
# 
# ## Variabel Kategorikal

# %%
categorical_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, data=df, ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('plot/categorical_dist.png')

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, col in enumerate(categorical_cols[:-1]):
    sns.countplot(x=col, hue='DEATH_EVENT', data=df, ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(f'{col} vs DEATH_EVENT')
plt.tight_layout()
plt.savefig('plot/categorical_vs_death.png')

# %% [markdown]
# ## Variabel Numerik

# %%
numerical_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], kde=True, ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('plot/numerical_dist.png')

# %%
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for i, col in enumerate(numerical_cols):
    sns.boxplot(x='DEATH_EVENT', y=col, data=df, ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(f'{col} vs DEATH_EVENT')
plt.tight_layout()
plt.savefig('plot/numerical_vs_death.png')

# %% [markdown]
# ## Matriks Korelasi

# %%
corr_matrix = df[numerical_cols + ['DEATH_EVENT']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('plot/correlation_matrix.png')

# %% [markdown]
# ## Deteksi Outlier

# %%
Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).sum()
print(outliers)

# %% [markdown]
# Terdapat beberapa outlier pada `creatinine_phosphokinase`, `ejection_fraction`, `platelets`, `serum_creatinine`, dan `serum_sodium`.
# 
# Transformasi log untuk variabel skewed (`creatinine_phosphokinase` dan `serum_creatinine`).

# %%
from sklearn.preprocessing import StandardScaler

df['log_creatinine_phosphokinase'] = np.log1p(df['creatinine_phosphokinase'])
df['log_serum_creatinine'] = np.log1p(df['serum_creatinine'])

df_processed = df.drop(columns=['creatinine_phosphokinase', 'serum_creatinine'])

features_to_scale = ['age', 'log_creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'log_serum_creatinine', 'serum_sodium']
scaler = StandardScaler()
df_scaled = df_processed.copy()
df_scaled[features_to_scale] = scaler.fit_transform(df_processed[features_to_scale])
print(df_scaled.head())
