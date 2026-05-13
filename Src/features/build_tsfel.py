import pandas as pd
import numpy as np
import tsfel
import warnings
import os

# Mengabaikan peringatan (warnings) dari tsfel agar terminal tetap bersih
warnings.filterwarnings('ignore')

def extract_tsfel_features(df):
    """
    Fungsi untuk mengekstrak fitur statistik dasar dari 9 parameter sensor ESP
    menggunakan pustaka TSFEL, lalu membersihkannya dari nilai inf/NaN.
    """
    print("[Anggota 3 - TSFEL] Memulai ekstraksi fitur statistik untuk 9 Parameter Sensor...")
    
    # 1. Definisi 9 'Kolom Emas' hasil kerja Anggota 1 (Part 4)
    # Kita tidak memasukkan Well_ID dan Downtime karena itu bukan sinyal
    sensor_cols = [
        'Freq', 'Current', 'Vibration', 'Intake_Press', 
        'Motor_Temp', 'Gas_Sat', 'Delta_Press', 'Pwr_Diff', 'Liq_Intake'
    ]
    
    # Pastikan semua kolom tersebut benar-benar ada di dataframe
    missing_cols = [col for col in sensor_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ditemukan di data: {missing_cols}")
        
    # 2. Ambil hanya 9 kolom fitur yang akan diekstrak
    features_only = df[sensor_cols]
    
    # 3. Setup TSFEL (Gunakan domain 'statistical' agar cepat dan relevan)
    cgf_settings = tsfel.get_features_by_domain('statistical')
    
    # 4. Proses Ekstraksi
    # window_size=1 karena kita mengekstrak fitur pada setiap baris harian
    df_tsfel = tsfel.time_series_features_extractor(
        cgf_settings, 
        features_only, 
        window_size=1, 
        verbose=0 # verbose=0 agar terminal tidak kepenuhan bar loading
    )
    
    # ==========================================
    # 5. SANITASI DATA (SANGAT PENTING!)
    # ==========================================
    print("[Anggota 3 - TSFEL] Membersihkan nilai Infinity dan NaN hasil ekstraksi...")
    df_tsfel = df_tsfel.replace([np.inf, -np.inf], np.nan)
    df_tsfel = df_tsfel.fillna(0)
    
    print(f"[Anggota 3 - TSFEL] Selesai! Menghasilkan {df_tsfel.shape[1]} kolom fitur baru yang sudah bersih.")
    
    return df_tsfel


# ==========================================
# BLOK PENGUJIAN
# ==========================================
if __name__ == "__main__":
    print("=== EKSEKUSI PENUH: BUILD_TSFEL.PY (9 PARAMETER) ===")
    
    base_dir = os.getcwd()
    file_path = os.path.join(base_dir, 'Data', 'Processed', 'esp_data_final_preprocessed.csv')
    
    try:
        print(f"Membaca data dari: {file_path}")
        df_asli = pd.read_csv(file_path)
        print(f"Total data yang akan diproses: {df_asli.shape[0]} baris.")
        print("Mengekstrak fitur TSFEL... (Ini akan memakan waktu lama, silakan buat kopi ☕)")

        # EKSEKUSI PENUH TANPA BATASAN BARIS
        hasil_tsfel = extract_tsfel_features(df_asli)
        
        # PROSES PENYIMPANAN KE CSV
        output_dir = os.path.join(base_dir, 'Data', 'Processed')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'tsfel_features.csv')
        hasil_tsfel.to_csv(output_path, index=False)
        
        print(f"\n✅ SUKSES! {hasil_tsfel.shape[1]} Fitur TSFEL berhasil diekstrak dari seluruh data.")
        print(f"📂 File fisik tersimpan di: {output_path}")
        
    except FileNotFoundError:
        print(f"\n❌ [GAGAL] File {file_path} tidak ditemukan.")
    except Exception as e:
        print(f"\n❌ [GAGAL] Terjadi error: {e}")