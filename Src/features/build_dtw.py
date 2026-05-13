import pandas as pd
import numpy as np
from tslearn.metrics import dtw
import os

def load_anomaly_library(filepath):
    """
    Fungsi untuk memuat file .npy berisi titik pusat klaster (barycenters) 
    hasil clustering Anggota 2.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File pustaka anomali tidak ditemukan di: {filepath}")
    
    print(f"[Anggota 3 - DTW] Memuat Pustaka Anomali dari {filepath}...")
    return np.load(filepath)

def calculate_dtw_distances(df, anomaly_centers):
    """
    Menghitung jarak DTW Multivariate (9 Dimensi) dari setiap baris data
    ke masing-masing pusat klaster anomali.
    """
    print("[Anggota 3 - DTW] Mulai menghitung jarak DTW ke pusat klaster (Multivariate 9D)...")
    
    # 1. Definisi 9 'Kolom Emas' yang akan diukur jaraknya
    sensor_cols = [
        'Freq', 'Current', 'Vibration', 'Intake_Press', 
        'Motor_Temp', 'Gas_Sat', 'Delta_Press', 'Pwr_Diff', 'Liq_Intake'
    ]
    
    # Pastikan kolom yang dibutuhkan tersedia
    missing_cols = [col for col in sensor_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ditemukan di data: {missing_cols}")
        
    # 2. Ekstrak data dan ubah bentuknya untuk tslearn (Multivariate)
    # tslearn butuh format 3D: (jumlah_sampel, panjang_waktu, jumlah_fitur)
    # KUNCI UTAMA: jumlah_fitur sekarang adalah 9, bukan lagi 2!
    X_data = df[sensor_cols].values
    X_ts = X_data.reshape((X_data.shape[0], 1, 9))
    
    distances = []
    
    # 3. Hitung jarak tiap baris ke tiap pusat klaster
    for i, x in enumerate(X_ts):
        dist_vector = [dtw(x, center) for center in anomaly_centers]
        distances.append(dist_vector)
        
        # Print progress setiap 5000 baris
        if (i + 1) % 5000 == 0:
            print(f"   ... {i + 1} baris telah diproses.")
            
    # 4. Ubah hasil perhitungan menjadi DataFrame
    num_clusters = len(anomaly_centers)
    col_names = [f'DTW_Dist_C{i}' for i in range(num_clusters)]
    
    df_dtw = pd.DataFrame(distances, columns=col_names)
    print(f"[Anggota 3 - DTW] Ekstraksi DTW selesai! Menghasilkan {num_clusters} kolom fitur baru.")
    
    return df_dtw


# ==========================================
# BLOK PENGUJIAN
# ==========================================
if __name__ == "__main__":
    print("=== EKSEKUSI PENUH: BUILD_DTW.PY (9 PARAMETER) ===")
    
    base_dir = os.getcwd()
    path_data_asli = os.path.join(base_dir, 'Data', 'Processed', 'esp_data_final_preprocessed.csv')
    path_pustaka_anomali = os.path.join(base_dir, 'models', 'anomaly_library_centers.npy') 
    
    try:
        print(f"Membaca data dari: {path_data_asli}")
        df_asli = pd.read_csv(path_data_asli)
        print(f"Total data yang akan diproses: {df_asli.shape[0]} baris.")
        
        if os.path.exists(path_pustaka_anomali):
            anomaly_centers = load_anomaly_library(path_pustaka_anomali)
        else:
            print("\n[Peringatan] File Pustaka Anomali belum ada.")
            print("Menggunakan 5 Pola Cadangan (Dummy) untuk eksekusi penuh...")
            anomaly_centers = np.random.rand(5, 1, 9)

        print("Menghitung jarak DTW... (Ini proses komputasi berat, mohon bersabar ⏳)")

        # EKSEKUSI PENUH TANPA BATASAN BARIS
        hasil_dtw = calculate_dtw_distances(df_asli, anomaly_centers)
        
        # PROSES PENYIMPANAN KE CSV
        output_dir = os.path.join(base_dir, 'Data', 'Processed')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'dtw_features.csv')
        hasil_dtw.to_csv(output_path, index=False)
        
        print("\n✅ SUKSES! Fitur Jarak DTW berhasil diekstrak dari seluruh data.")
        print(f"📂 File fisik tersimpan di: {output_path}")

    except FileNotFoundError as e:
        print(f"\n❌ [GAGAL] Error: {e}")
    except Exception as e:
        print(f"\n❌ [GAGAL] Terjadi error: {e}")