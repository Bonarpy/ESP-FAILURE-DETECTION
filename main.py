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
    
    # 1. Baca Panel Kontrol
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    paths = config['paths']
    
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
    if os.path.exists(paths['xgboost_model']):
        model = xgb.XGBClassifier()
        model.load_model(paths['xgboost_model'])
        
        print("AI sedang memprediksi potensi kerusakan dengan fitur TSFEL & DTW...")
        
        # KITA GUNAKAN X_final KARENA AI SUDAH PINTAR!
        try:
            prediksi = model.predict(X_final)
        except Exception as e:
            print(f"❌ ERROR Prediksi: {e}")
            return
        
        df_matang['Prediksi_Kerusakan'] = prediksi
        
        # Simpan hasil laporan
        os.makedirs(os.path.dirname(paths['output_predictions']), exist_ok=True)
        df_matang.to_csv(paths['output_predictions'], index=False)
        print(f"\n[SELESAI] Laporan prediksi berhasil disimpan di: {paths['output_predictions']} 🎉")
    else:
        print(f"[Warning] File model {paths['xgboost_model']} belum ada! Prediksi dibatalkan.")

if __name__ == "__main__":
    run_pipeline()