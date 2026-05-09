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
    print("=== MENGUJI MODUL BUILD_DTW.PY ===")
    
    # 1. Buat Data Dummy (Seolah-olah dari Anggota 1)
    df_dummy = pd.DataFrame({
        'Freq_Norm': np.random.rand(5),  # 5 baris data dummy
        'Load_Norm': np.random.rand(5)
    })
    
    print("Data Input Dummy:")
    print(df_dummy)
    print("-" * 30)

    # 2. Buat Klaster Anomali Dummy (Seolah-olah dari Anggota 2)
    # Anggap ada 6 klaster, 1 timestep, 2 fitur
    dummy_centers = np.random.rand(6, 1, 2)
    
    # 3. Panggil fungsi yang sudah Anda definisikan di atas!
    try:
        hasil_dtw = calculate_dtw_distances(df_dummy, dummy_centers)
        
        print("\n=== HASIL EKSTRAKSI ===")
        print("Bentuk Dataframe Output:", hasil_dtw.shape)
        print("Hasil Tabel Jarak:")
        print(hasil_dtw)
        print("\n[SUKSES] Modul build_dtw.py Anda berfungsi sempurna!")
        
    except Exception as e:
        print(f"\n[GAGAL] Terjadi error: {e}")