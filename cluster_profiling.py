import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score

# --- assume df, data_combined, numeric_data_scaled, categorical_feats, label_encoders, clusters already exist ---

# 1) Silhouette Score on numeric features
sil_score_num = silhouette_score(numeric_data_scaled, df['Cluster'])
print(f"Silhouette Score (numeric features): {sil_score_num:.3f}")

# 2) Silhouette Diagram (numeric only)
sil_values = silhouette_samples(numeric_data_scaled, df['Cluster'])

plt.figure(figsize=(8,5))
y_lower = 10
for i in range(4):  # for each cluster
    ith_sil = sil_values[df['Cluster']==i]
    ith_sil.sort()
    y_upper = y_lower + len(ith_sil)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_sil, alpha=0.7)
    plt.text(-0.03, y_lower + len(ith_sil)/2, str(i))
    y_lower = y_upper + 10
plt.axvline(sil_score_num, color="red", linestyle="--")
plt.title("Silhouette Plot (Numeric features)")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")
plt.show()

# 3) Cluster Profiling: Numeric Means
print("\n=== Cluster Means (Numeric Features) ===")
cluster_means = df.groupby('Cluster')[numeric_feats].mean().round(2)
print(cluster_means)

# 4) Cluster Profiling: Categorical Modes
print("\n=== Cluster Modes (Categorical Features) ===")
modes = {}
for col in categorical_feats:
    modes[col] = df.groupby('Cluster')[col] \
                   .agg(lambda x: x.value_counts().idxmax())
modes_df = pd.DataFrame(modes)
print(modes_df)

# 5) Visualize Categorical Modes
for col in categorical_feats:
    plt.figure(figsize=(6,4))
    sns.countplot(x='Cluster', hue=col, data=df, palette='Set1')
    plt.title(f'{col} Distribution by Cluster')
    plt.tight_layout()
    plt.show()
