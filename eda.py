import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_and_eda(
    input_csv: str = 'security_alerts_July2025.csv',
    test_size: float = 0.2,
    random_seed: int = 42
):
    # Load Data
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])

    # Initial Info
    print("\n=== DATAFRAME INFO ===")
    df.info()
    print("\n=== MISSING VALUES (%) ===")
    missing_vals = (df.isnull().mean() * 100).round(2)
    print(missing_vals[missing_vals > 0].sort_values(ascending=False))

    # Feature Engineering from timestamp
    df['hour']       = df['timestamp'].dt.hour
    df['weekday']    = df['timestamp'].dt.day_name()
    df['is_weekend'] = df['weekday'].isin(['Saturday','Sunday']).astype(int)

    # Train/Test Split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df['incident_priority']
    )

    # Numeric and Categorical Features
    numeric_feats = [
        'cve_score', 'ip_reputation_score', 'login_attempts',
        'cpu_usage_percent', 'memory_usage_percent', 'payload_size',
         'hour', 'is_weekend'
    ]
    categorical_feats = [ 
        'incident_priority', 'user_role', 'system_context',
        'weekday'
    ]

    # Preprocessing Pipeline

    # 1. Numeric Features: Impute missing values (Median) + Scale with StandardScaler
    num_imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train_num = num_imputer.fit_transform(train_df[numeric_feats])
    X_train_num = scaler.fit_transform(X_train_num)

    X_test_num = num_imputer.transform(test_df[numeric_feats])
    X_test_num = scaler.transform(X_test_num)

    # 2. Categorical Features: Label Encoding + One-Hot Encoding
    label_encoders = {}
    for col in categorical_feats:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat = ohe.fit_transform(train_df[categorical_feats])
    X_test_cat = ohe.transform(test_df[categorical_feats])

    # EDA

    # 1. Distribution of Incident Priority (Target Variable)
    plt.figure(figsize=(6, 4))
    sns.countplot(x='incident_priority', data=df, palette='viridis')
    plt.title('Distribution of Incident Priority')
    plt.show()

    # 2. Histograms for Numeric Features by Incident Priority
    for var in numeric_feats:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=var, hue='incident_priority', kde=True, multiple="stack", palette="viridis")
        plt.title(f'Distribution of {var} by Incident Priority')
        plt.show()

    # 3. Boxplots for Numeric Features by Incident Priority
    for var in numeric_feats:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='incident_priority', y=var, data=df, palette="viridis")
        plt.title(f'{var} by Incident Priority')
        plt.show()

    # 4. Correlation Heatmap (Numeric Variables)
    corr = df[numeric_feats].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Features')
    plt.show()

    # 5. Pairplots for Bivariate Analysis of Features with Target
    sns.pairplot(df, vars=numeric_feats, hue='incident_priority', palette="viridis")
    plt.suptitle('Pairwise Relationships of Numeric Features with Incident Priority', y=1.02)
    plt.show()

    # 6. Categorical Features Distribution
    plt.figure(figsize=(6, 4))
    for col in categorical_feats:
        if df[col].dtype == 'object':
            sns.countplot(x=col, data=df, palette='viridis')
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
            plt.show()

    # 7. Categorical vs Numeric: Incident Priority vs Key Numeric Features
    for var in numeric_feats:
        plt.figure(figsize=(6, 4))
        sns.violinplot(x='incident_priority', y=var, data=df, palette="viridis")
        plt.title(f'{var} Distribution by Incident Priority')
        plt.show()

    # 8. Alerts Over Time (Time Series)
    daily_alerts = df.set_index('timestamp').resample('D').size()
    plt.figure(figsize=(10, 6))
    daily_alerts.plot()
    plt.title('Number of Alerts per Day')
    plt.ylabel('Count of Alerts')
    plt.show()

    # Save Processed Data
    pd.DataFrame(X_train_num, columns=numeric_feats).to_csv('train_numeric.csv', index=False)
    pd.DataFrame(X_test_num, columns=numeric_feats).to_csv('test_numeric.csv', index=False)
    pd.DataFrame(X_train_cat, columns=ohe.get_feature_names_out(categorical_feats)).to_csv('train_ohe.csv', index=False)
    pd.DataFrame(X_test_cat, columns=ohe.get_feature_names_out(categorical_feats)).to_csv('test_ohe.csv', index=False)

    print("\nPreprocessing and EDA complete. Processed files saved.")

# Run the preprocessing and EDA
if __name__ == '__main__':
    preprocess_and_eda()
