import sys
import os
import subprocess
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import pandas as pd
import xgboost as xgb

# KITA HAPUS import preprocessing di sini agar tidak langsung jalan otomatis
# Pastikan huruf besar/kecil 'src' atau 'Src' sesuai dengan nama folder di laptop Anda
from Src.features.build_tsfel import extract_tsfel_features
from Src.features.build_dtw import calculate_dtw_distances, load_anomaly_library

def run_pipeline():
    print("=== MEMULAI PIPELINE ESP FAILURE DETECTION ===")
    
    # --- TAMBAHKAN KAMUS JALUR (PATHS) INI DI SINI ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths = {
        'anomaly_library': os.path.join(base_dir, 'models', 'anomaly_library.pkl'),
        'xgboost_model': os.path.join(base_dir, 'models', 'xgboost_production.json'),
        'output_predictions': os.path.join(base_dir, 'reports', 'prediksi_kerusakan_esp.csv')
    }
    # -------------------------------------------------

    # ==========================================
    # TAHAP 1: DATA ENGINEERING
    # ==========================================
    print("\n[TAHAP 1] Menjalankan Data Engineer (Anggota 1)...")
    
    # Kita panggil kode Anggota 1 secara normal agar grafiknya muncul
    subprocess.run(["python", "Src/data/preprocessing.py"])
    
    # Mengambil file hasil jadinya Anggota 1
    file_matang_anggota1 = "Data/Processed/dataset_siap_training.csv"
    
    if not os.path.exists(file_matang_anggota1):
        print(f"\n[GAGAL] File {file_matang_anggota1} tidak ditemukan!")
        return

    df_matang = pd.read_csv(file_matang_anggota1)
    print(f"\n✅ Data matang berhasil dimuat oleh main.py! Bentuk: {df_matang.shape}")

    # ==========================================
    # TAHAP 2: FEATURE ENGINEERING
    # ==========================================
    print("\n[TAHAP 2] Menjalankan Feature Engineer (Anggota 3)...")
    
    # Ekstrak TSFEL
    df_tsfel = extract_tsfel_features(df_matang)
    
    # Ekstrak DTW
    if os.path.exists(paths['anomaly_library']):
        anomaly_centers = load_anomaly_library(paths['anomaly_library'])
        df_dtw = calculate_dtw_distances(df_matang, anomaly_centers)
    else:
        print("[Warning] Pustaka anomali tidak ditemukan. Melewati tahap DTW sementara.")
        df_dtw = pd.DataFrame() 
        
    # Gabungkan semua fitur
    print("\n[TAHAP 3] Menggabungkan Semua Fitur...")
    X_final = pd.concat([df_tsfel.reset_index(drop=True), df_dtw.reset_index(drop=True)], axis=1)
    print(f"Data final siap disuapkan ke AI! Bentuk: {X_final.shape}")
    
   # ==========================================
    # TAHAP 4: PREDIKSI XGBOOST
    # ==========================================
    print("\n[TAHAP 4] Memanggil AI XGBoost (Anggota 4)...")
    
    # KITA GUNAKAN PATH ABSOLUT UNTUK MENCEGAH SALAH BACA FILE LAMA
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "xgboost_production.json")
    
    if os.path.exists(model_path):
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        
        print("AI sedang memprediksi potensi kerusakan dengan 203 fitur TSFEL & DTW...")
        try:
            # Pastikan urutan kolom X_final sama persis dengan yang dipelajari AI saat training
            # XGBoost sangat sensitif terhadap urutan kolom
            prediksi = model.predict(X_final)
        except Exception as e:
            print(f"❌ ERROR Prediksi: {e}")
            return
        
        df_matang['Prediksi_Kerusakan'] = prediksi
        
        # Simpan hasil laporan
        output_path = os.path.join(base_dir, "reports", "prediksi_kerusakan_esp.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_matang.to_csv(output_path, index=False)
        print(f"\n[SELESAI] Laporan prediksi berhasil disimpan di: {output_path} 🎉")
    else:
        print(f"❌ ERROR: File model {model_path} tidak ditemukan!")
        print("Harap jalankan perintah ini dulu di terminal: python Src/models/train_xgboost.py")

if __name__ == "__main__":
    run_pipeline()