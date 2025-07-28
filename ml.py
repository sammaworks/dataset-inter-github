import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score, classification_report

# Ordinal models
from mord import LogisticIT, OrdinalRidge

def load_and_engineer(path='synthetic_security_alerts.csv'):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df['hour']       = df['timestamp'].dt.hour
    df['weekday']    = df['timestamp'].dt.day_name()
    df['is_weekend'] = df['weekday'].isin(['Saturday','Sunday']).astype(int)
    return df

def prepare_data(df):
    # Map ordinal labels to integers
    priority_map = {'Low':0, 'Medium':1, 'High':2, 'Critical':3}
    df['priority_ord'] = df['incident_priority'].map(priority_map)
    
    X = df.drop(columns=['timestamp','source_ip','incident_priority','outcome','priority_ord'])
    y = df['priority_ord']
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def build_preprocessor():
    numeric_feats = [
        'cve_score','ip_reputation_score','login_attempts',
        'cpu_usage_percent','memory_usage_percent','payload_size',
        'time_to_remediate_hours','hour','is_weekend'
    ]
    categorical_feats = ['user_role','system_context','weekday']
    
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale',  StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe',    OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('cat', cat_pipe, categorical_feats)
    ]), numeric_feats, categorical_feats

def evaluate_model(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    print(f"  Accuracy:               {acc:.3f}")
    print(f"  Mean Absolute Error:    {mae:.3f}")
    print(f"  Quadratic Weighted Kappa: {qwk:.3f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, digits=3, target_names=['Low','Medium','High','Critical']))

def main():
    # Load & engineer
    df = load_and_engineer()
    
    # Prepare train/test
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Preprocessor
    preprocessor, num_feats, cat_feats = build_preprocessor()
    
    # Define models
    models = {
        'OrdinalLogistic': LogisticIT(),
        'OrdinalRidge':    OrdinalRidge(alpha=1.0),
        'RandomForest':    RandomForestClassifier(n_estimators=200, random_state=42)
    }
    
    # Fit & evaluate
    for name, model in models.items():
        print(f"\n== {name} ==")
        pipe = Pipeline([
            ('preproc', preprocessor),
            ('clf',     model)
        ])
        pipe.fit(X_train, y_train)
        evaluate_model(pipe, X_test, y_test)
    
    # Cross‑validation (accuracy only)
    print("\n== 5‑Fold CV Accuracy ==")
    for name, model in models.items():
        pipe = Pipeline([('preproc', preprocessor), ('clf', model)])
        scores = cross_val_score(pipe, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), 
                                 cv=5, scoring='accuracy')
        print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")

if __name__ == '__main__':
    main()
