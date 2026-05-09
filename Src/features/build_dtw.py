import pandas as pd
import numpy as np
from tslearn.metrics import dtw
import os

def load_anomaly_library(filepath):
    """
    Fungsi untuk memuat file .npy berisi titik pusat klaster (barycenters) 
    yang nantinya akan dihasilkan oleh Anggota 2.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File pustaka anomali tidak ditemukan di: {filepath}")
    
    print(f"[Anggota 3 - DTW] Memuat Pustaka Anomali dari {filepath}...")
    return np.load(filepath)

def calculate_dtw_distances(df, anomaly_centers):
    """
    Fungsi utama untuk menghitung jarak DTW dari setiap baris data
    ke masing-masing pusat klaster anomali.
    """
    print("[Anggota 3 - DTW] Mulai menghitung jarak DTW ke pusat klaster...")
    
    # 1. Pastikan kolom yang dibutuhkan tersedia
    if not {'Freq_Norm', 'Load_Norm'}.issubset(df.columns):
        raise ValueError("Dataframe harus memiliki kolom 'Freq_Norm' dan 'Load_Norm'")
        
    # 2. Ekstrak data dan ubah bentuknya untuk tslearn
    # tslearn butuh format 3D: (jumlah_sampel, panjang_waktu, jumlah_fitur)
    X_data = df[['Freq_Norm', 'Load_Norm']].values
    X_ts = X_data.reshape((X_data.shape[0], 1, 2))
    
    distances = []
    
    # 3. Hitung jarak tiap baris ke tiap pusat klaster
    # Catatan: Karena kita mengeksekusi banyak baris, proses ini mungkin memakan waktu
    for i, x in enumerate(X_ts):
        dist_vector = [dtw(x, center) for center in anomaly_centers]
        distances.append(dist_vector)
        
        # Opsional: Print progress setiap 10.000 baris agar tidak dikira error/nge-hang
        if (i + 1) % 10000 == 0:
            print(f"  ... {i + 1} baris telah diproses.")
            
    # 4. Ubah hasil perhitungan menjadi DataFrame
    num_clusters = len(anomaly_centers)
    col_names = [f'DTW_Dist_C{i}' for i in range(num_clusters)]
    
    df_dtw = pd.DataFrame(distances, columns=col_names)
    print(f"[Anggota 3 - DTW] Ekstraksi DTW selesai! Menghasilkan {num_clusters} kolom fitur baru.")
    
    return df_dtw



if __name__ == "__main__":
    print("=== MENGUJI MODUL BUILD_DTW.PY DENGAN DATA ASLI ===")
    
    # 1. Jalur file input (Hasil kerja Anda sebagai Anggota 1)
    path_data_asli = "C:\\Users\\Iman Fath Hatta\\ESP-FAILURE-DETECTION\\Data\\Processed\\processed_data.csv"
    # Jalur file dari Anggota 2 (Sesuaikan jika Anggota 2 sudah memberi file)
    path_pustaka_anomali = "models/anomaly_library_centers.npy" 
    
    try:
        # A. Muat Data Asli
        print(f"Membaca data asli dari: {path_data_asli}")
        df_asli = pd.read_csv(path_data_asli)
        
        # B. Muat Pustaka Anomali (Hasil Anggota 2)
        if os.path.exists(path_pustaka_anomali):
            anomaly_centers = load_anomaly_library(path_pustaka_anomali)
        else:
            print(f"[Peringatan] File {path_pustaka_anomali} belum ada.")
            print("Menggunakan 'Pola Cadangan' (Dummy) untuk simulasi...")
            # Simulasi 6 pola anomali (1 timestep, 2 fitur)
            anomaly_centers = np.random.rand(6, 1, 2)

        # C. Jalankan Perhitungan DTW
        # Kita coba tes pada 100 baris pertama saja dulu agar tidak terlalu lama
        print("\nMencoba menghitung DTW untuk 100 baris pertama...")
        hasil_dtw = calculate_dtw_distances(df_asli.head(100), anomaly_centers)
        
        print("\n=== HASIL EKSTRAKSI DTW ===")
        print("Bentuk Dataframe Output:", hasil_dtw.shape)
        print("Tampilan 5 Baris Pertama Jarak ke Tiap Klaster:")
        print(hasil_dtw.head())
        
        print("\n[SUKSES] Modul build_dtw.py Anda berhasil memproses data!")

    except Exception as e:
        print(f"\n[GAGAL] Terjadi error: {e}")