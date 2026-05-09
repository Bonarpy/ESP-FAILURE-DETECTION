import pandas as pd
import numpy as np
import tsfel
import warnings

# Mengabaikan peringatan (warnings) dari tsfel agar terminal tetap bersih
warnings.filterwarnings('ignore')

def extract_tsfel_features(df):
    """
    Fungsi untuk mengekstrak fitur statistik dasar dari data deret waktu
    menggunakan pustaka TSFEL, lalu membersihkannya dari nilai inf/NaN.
    """
    print("[Anggota 3 - TSFEL] Memulai ekstraksi fitur statistik...")
    
    # 1. Pastikan kolom yang dibutuhkan tersedia
    if not {'Freq_Norm', 'Load_Norm'}.issubset(df.columns):
        raise ValueError("Dataframe harus memiliki kolom 'Freq_Norm' dan 'Load_Norm'")
        
    # 2. Ambil hanya kolom fitur yang akan diekstrak
    features_only = df[['Freq_Norm', 'Load_Norm']]
    
    # 3. Setup TSFEL (Gunakan domain 'statistical' saja agar komputasi efisien)
    # Anda bisa menggantinya dengan 'all' nanti jika ingin mengekstrak spektral/temporal
    cgf_settings = tsfel.get_features_by_domain('statistical')
    
    # 4. Proses Ekstraksi
    # window_size=1 karena kita mengekstrak fitur pada setiap baris harian
    df_tsfel = tsfel.time_series_features_extractor(
        cgf_settings, 
        features_only, 
        window_size=1, 
        verbose=0 # verbose=0 agar tidak memunculkan bar loading yang terlalu panjang
    )
    
    # ==========================================
    # 5. SANITASI DATA (SANGAT PENTING!)
    # ==========================================
    print("[Anggota 3 - TSFEL] Membersihkan nilai Infinity dan NaN hasil ekstraksi...")
    # Ubah semua nilai inf dan -inf menjadi NaN
    df_tsfel = df_tsfel.replace([np.inf, -np.inf], np.nan)
    # Isi semua NaN dengan angka 0
    df_tsfel = df_tsfel.fillna(0)
    
    print(f"[Anggota 3 - TSFEL] Selesai! Menghasilkan {df_tsfel.shape[1]} kolom fitur baru yang sudah bersih.")
    
    return df_tsfel



if __name__ == "__main__":
    print("=== MENGUJI MODUL BUILD_TSFEL.PY DENGAN DATA ASLI ===")
    
    # 1. Mengambil data matang hasil kerja Anggota 1
    # Pastikan jalur filenya benar jika di-run dari root folder ESP-FAILURE-DETECTION
    file_path = "C:\\Users\\Iman Fath Hatta\\ESP-FAILURE-DETECTION\\Data\\Processed\\processed_data.csv" 
    
    try:
        print(f"Membaca data dari: {file_path}")
        df_asli = pd.read_csv(file_path)
        
        # Tampilkan sedikit info data aslinya
        print("Data Input Asli (5 baris pertama):")
        print(df_asli[['Well_ID', 'Freq_Norm', 'Load_Norm']].head())
        print("-" * 30)

        # 2. Panggil fungsi ekstraksi TSFEL menggunakan data asli!
        hasil_tsfel = extract_tsfel_features(df_asli)
        
        print("\n=== HASIL EKSTRAKSI ===")
        print(f"Bentuk Dataframe Output: {hasil_tsfel.shape[0]} baris x {hasil_tsfel.shape[1]} kolom")
        
        print("\nMenampilkan 5 Kolom Pertama (sebagai sampel):")
        print(hasil_tsfel.iloc[:, :5].head()) 
        
        print("\n[SUKSES] Modul build_tsfel.py Anda berhasil memproses data asli!")
        
    except FileNotFoundError:
        print(f"\n[GAGAL] File {file_path} tidak ditemukan. Pastikan Anda sudah menjalankan normalize.py milik Anggota 1.")
    except Exception as e:
        print(f"\n[GAGAL] Terjadi error: {e}")