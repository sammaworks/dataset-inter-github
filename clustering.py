import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from kmodes.kprototypes import KPrototypes
from sklearn.decomposition import PCA

# Load the synthetic dataset
df = pd.read_csv('synthetic_security_alerts.csv', parse_dates=['timestamp'])

# Check if 'timestamp' is in the correct datetime format
if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Feature Engineering from timestamp
df['hour']       = df['timestamp'].dt.hour
df['weekday']    = df['timestamp'].dt.day_name()
df['is_weekend'] = df['weekday'].isin(['Saturday', 'Sunday']).astype(int)

# Ensure 'hour' and 'is_weekend' are created
print(df[['timestamp', 'hour', 'weekday', 'is_weekend']].head())

# Define numeric and categorical features
numeric_feats = [
    'cve_score', 'ip_reputation_score', 'login_attempts',
    'cpu_usage_percent', 'memory_usage_percent', 'payload_size',
    'time_to_remediate_hours', 'hour', 'is_weekend'
]
categorical_feats = [
    'incident_priority', 'user_role', 'system_context',
    'weekday', 'outcome'
]

# Preprocessing numeric features
numeric_data = df[numeric_feats]
scaler = StandardScaler()
numeric_data_scaled = scaler.fit_transform(numeric_data)

# Preprocessing categorical features
categorical_data = df[categorical_feats]
label_encoders = {}
for col in categorical_feats:
    le = LabelEncoder()
    categorical_data[col] = le.fit_transform(categorical_data[col].astype(str))
    label_encoders[col] = le

# Combine numeric and categorical data for clustering
data_combined = np.concatenate([numeric_data_scaled, categorical_data], axis=1)

# Indices for categorical data (for KPrototypes)
categorical_indices = list(range(len(numeric_feats), len(numeric_feats) + len(categorical_feats)))

# Run K-Prototypes clustering
kproto = KPrototypes(n_clusters=4, init='Cao', n_init=10, verbose=2)
clusters = kproto.fit_predict(data_combined, categorical=categorical_indices)

# Add the cluster labels to the dataframe
df['Cluster'] = clusters

# 1. Visualize the clusters in a 2D plane using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(numeric_data_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=df['Cluster'], palette="Set2", s=60)
plt.title('Clustering with K-Prototypes (PCA-reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# 2. Show the cluster centers (for numeric features)
centers = kproto.cluster_centroids_[:len(numeric_feats)]
centers_df = pd.DataFrame(centers, columns=numeric_feats)
print("\nCluster Centers (Numeric Features):")
print(centers_df)

# 3. Show cluster distributions for categorical features
for col in categorical_feats:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Cluster', hue=col, data=df, palette="Set2")
    plt.title(f'{col} distribution by cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.show()

# 4. Cluster profiling: inspect the clusters
print("\nCluster Profiling (Cluster mean values):")
print(df.groupby('Cluster')[numeric_feats].mean())

# 5. Save the clustering result
df.to_csv('synthetic_security_alerts_with_clusters.csv', index=False)

print("\nCluster analysis complete. Clustered dataset saved as 'synthetic_security_alerts_with_clusters.csv'.")
