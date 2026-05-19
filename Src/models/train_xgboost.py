import pandas as pd
import numpy as np
import os
import json
import yaml
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def train_and_evaluate_xgboost():
    print("=== [MULAI TRAINING AI (SUPER XGBOOST)] ===")
    
    base_dir = os.getcwd()
    
    # ==========================================
   # ==========================================
    # 1. MEMUAT DATA (GABUNGAN TSFEL + DTW)
    # ==========================================
    print("\n[1/5] Membaca Fitur TSFEL dan DTW (Hasil Anggota 3)...")
    
    path_tsfel = os.path.join(base_dir, 'Data', 'Processed', 'tsfel_features.csv')
    path_dtw = os.path.join(base_dir, 'Data', 'Processed', 'dtw_features.csv')
    
    # KITA UBAH NAMA FILE LABELNYA KE DATA SEBELUM SMOTE (22.934 baris)
    path_label = os.path.join(base_dir, 'Data', 'Processed', 'esp_data_final_preprocessed.csv')
    
    # Membaca ketiga file
    try:
        df_tsfel = pd.read_csv(path_tsfel)
        df_dtw = pd.read_csv(path_dtw)
        df_label = pd.read_csv(path_label)
    except FileNotFoundError as e:
        print(f"❌ ERROR: File tidak ditemukan! Pastikan Anggota 3 sudah mengekstrak fitur. Detail: {e}")
        return
    
    # Menggabungkan fitur dari Anggota 3 menjadi X
    X = pd.concat([df_tsfel.reset_index(drop=True), df_dtw.reset_index(drop=True)], axis=1)
    
    # Mengambil kunci jawaban dari kolom Downtime
    if 'Target' in df_label.columns:
        y = df_label['Target']
    elif 'Downtime' in df_label.columns:
        # Jika downtime > 0 jam, artinya rusak (1), jika 0 artinya normal (0)
        y = df_label['Downtime'].apply(lambda x: 1 if x > 0 else 0)
    else:
        y = df_label['Current'].apply(lambda x: 1 if x > 0 else 0)
        
    print(f"Total Fitur (X): {X.shape[1]} kolom")
    
    # MEMBAGI DATA (80% Train, 20% Test)
    # SANGAT KRITIS: shuffle=False agar urutan waktu tidak rusak!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Data Latih (Train): {X_train.shape[0]} baris")
    print(f"Data Uji (Test): {X_test.shape[0]} baris")

    # ==========================================
    # 2. DATA BALANCING (SMOTE) HANYA DI TRAIN SET
    # ==========================================
    print("\n[2/5] Menyeimbangkan Data Latih dengan SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Distribusi Train SESUDAH SMOTE: \n{y_train_res.value_counts().to_dict()}")

    # ==========================================
    # 3. HYPERPARAMETER TUNING (RandomizedSearchCV)
    # ==========================================
    print("\n[3/5] Mencari Settingan (Hyperparameter) XGBoost Terbaik...")
    
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=xgb_model, 
        param_distributions=param_grid, 
        n_iter=10,        
        scoring='recall', 
        cv=3,             
        verbose=1,
        n_jobs=-1         
    )
    
    random_search.fit(X_train_res, y_train_res)
    best_model = random_search.best_estimator_
    print(f"Settingan Terbaik Ditemukan: {random_search.best_params_}")

    # ==========================================
    # 4. EVALUASI PADA DATA UJI (TEST SET)
    # ==========================================
    print("\n[4/5] Menguji Model ke Data Ujian (Test Set)...")
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print("\n--- RAPOR AI (XGBOOST) ---")
    print(f"Akurasi   : {acc:.2%}")
    print(f"Presisi   : {prec:.2%}")
    print(f"Recall    : {rec:.2%}")
    print(f"F1-Score  : {f1:.2%}")

    # ==========================================
    # 5. MENYIMPAN HASIL
    # ==========================================
    print("\n[5/5] Menyimpan Model, Metrik, dan Konfigurasi...")
    
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'configs'), exist_ok=True)
    
    model_path = os.path.join(base_dir, 'models', 'xgboost_production.json')
    best_model.save_model(model_path)
    
    metrics_dict = {
        'accuracy': float(acc), 'precision': float(prec),
        'recall': float(rec), 'f1_score': float(f1),
        'confusion_matrix': cm
    }
    metrics_path = os.path.join(base_dir, 'reports', 'metrics_evaluation.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    config_dict = {
        'model_name': 'XGBoost ESP Anomaly Detector',
        'version': '2.0',
        'best_hyperparameters': random_search.best_params_
    }
    config_path = os.path.join(base_dir, 'configs', 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    print(f"SUKSES! Model disimpan di: {model_path}")

if __name__ == "__main__":
    train_and_evaluate_xgboost()