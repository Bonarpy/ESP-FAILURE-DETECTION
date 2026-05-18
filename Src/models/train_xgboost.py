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
    print("=== [MULAI TRAINING AI (XGBOOST) ===")
    
    # Setup path
    base_dir = os.getcwd()
    data_path = os.path.join(base_dir, 'ESP-FAILURE-DETECTION','Data', 'Processed', 'dataset_siap_training.csv')
    
    # ==========================================
    # MEMUAT DATA & TIME-BASED SPLIT
    # ==========================================
    print("\n[1/5] Membaca 'Super CSV' dan Membagi Data...")
    df = pd.read_csv(data_path)
    
    # Asumsi kolom label bernama 'Target' atau 'Downtime'
    if 'Target' in df.columns:
        y = df['Target']
    else:
        # Jika masih Downtime, ubah ke biner (1=Rusak, 0=Normal)
        y = df['Current'].apply(lambda x: 1 if x > 0 else 0)
        
    # Buang kolom yang bukan fitur (Target dan Well_ID jika ada)
    cols_to_drop = [col for col in ['Target', 'Downtime', 'Well_ID'] if col in df.columns]
    X = df.drop(columns=cols_to_drop)
    
    # MEMBAGI DATA (80% Train, 20% Test)
    # SANGAT KRITIS: shuffle=False agar urutan waktu tidak rusak!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Data Latih (Train): {X_train.shape[0]} baris")
    print(f"Data Uji (Test): {X_test.shape[0]} baris")

    # ==========================================
    # DATA BALANCING (SMOTE) HANYA DI TRAIN SET
    # ==========================================
    print("\n[2/5] Menyeimbangkan Data Latih dengan SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Distribusi Train SEBELUM SMOTE: \n{y_train.value_counts().to_dict()}")
    print(f"Distribusi Train SESUDAH SMOTE: \n{y_train_res.value_counts().to_dict()}")

    # ==========================================
    # HYPERPARAMETER TUNING (RandomizedSearchCV)
    # ==========================================
    print("\n[3/5] Mencari Settingan (Hyperparameter) XGBoost Terbaik...")
    
    # Menyiapkan kandidat pengaturan (Gigi mesin)
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    
    # Menggunakan RandomizedSearch agar proses tuning lebih cepat dari GridSearch
    random_search = RandomizedSearchCV(
        estimator=xgb_model, 
        param_distributions=param_grid, 
        n_iter=10,        # Mencoba 10 kombinasi acak
        scoring='recall', # Fokus utama adalah Recall (meminimalisir lolosnya deteksi kerusakan)
        cv=3,             # 3-fold cross validation
        verbose=1,
        n_jobs=-1         
    )
    
    # Latih model dengan data yang sudah di-SMOTE
    random_search.fit(X_train_res, y_train_res)
    best_model = random_search.best_estimator_
    print(f"Settingan Terbaik Ditemukan: {random_search.best_params_}")

    # ==========================================
    # EVALUASI PADA DATA UJI (TEST SET)
    # ==========================================
    print("\n[4/5] Menguji Model ke Data Ujian (Test Set)...")
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist() # Convert ke list agar bisa disave ke JSON
    
    print("\n--- RAPOR AI (XGBOOST) ---")
    print(f"Akurasi   : {acc:.2%}")
    print(f"Presisi   : {prec:.2%} (Seberapa akurat alarm kerusakannya)")
    print(f"Recall    : {rec:.2%} (Berapa banyak kerusakan asli yang berhasil ditebak)")
    print(f"F1-Score  : {f1:.2%}")

    # ==========================================
    # MENYIMPAN HASIL (DEPLOYMENT READINESS)
    # ==========================================
    print("\n[5/5] Menyimpan Model, Metrik, dan Konfigurasi...")
    
    # Buat folder jika belum ada
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'configs'), exist_ok=True)
    
    # Simpan Model (.json untuk XGBoost)
    model_path = os.path.join(base_dir, 'models', 'xgboost_production.json')
    best_model.save_model(model_path)
    
    # Simpan Metrik Evaluasi (.json)
    metrics_dict = {
        'accuracy': float(acc), 'precision': float(prec),
        'recall': float(rec), 'f1_score': float(f1),
        'confusion_matrix': cm
    }
    metrics_path = os.path.join(base_dir, 'reports', 'metrics_evaluation.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    # Simpan Konfigurasi Terbaik (.yaml)
    config_dict = {
        'model_name': 'XGBoost ESP Anomaly Detector',
        'version': '1.0',
        'best_hyperparameters': random_search.best_params_
    }
    config_path = os.path.join(base_dir, 'configs', 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
        
    # Simpan Visualisasi Confusion Matrix (.png)
    plt.figure(figsize=(8, 6))
    cm_array = np.array(cm)
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal (0)', 'Rusak (1)'], 
                yticklabels=['Normal (0)', 'Rusak (1)'])
    plt.title('Confusion Matrix - XGBoost')
    plt.ylabel('Nilai Aktual')
    plt.xlabel('Nilai Prediksi')
    cm_plot_path = os.path.join(base_dir, 'reports', 'confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"SUKSES! Model disimpan di: {model_path}")
    print(f"Rapor Metrik disimpan di: {metrics_path}")
    print(f"Konfigurasi disimpan di: {config_path}")
    print(f"Visualisasi CM disimpan di: {cm_plot_path}")

if __name__ == "__main__":
    train_and_evaluate_xgboost()