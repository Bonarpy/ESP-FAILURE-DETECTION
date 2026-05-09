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
    print("=== MENGUJI MODUL BUILD_TSFEL.PY ===")
    
    # 1. Buat Data Dummy (Seolah-olah dari Anggota 1)
    # Sengaja kita buat nilai yang konstan di Load_Norm untuk memastikan
    # fitur pembersihan (sanitasi) inf/NaN kita bekerja.
    df_dummy = pd.DataFrame({
        'Freq_Norm': [0.5, 0.6, 0.7, 0.8, 0.9],
        'Load_Norm': [1.0, 1.0, 1.0, 1.0, 1.0] 
    })
    
    print("Data Input Dummy:")
    print(df_dummy)
    print("-" * 30)

    # 2. Panggil fungsi yang sudah Anda definisikan di atas!
    try:
        hasil_tsfel = extract_tsfel_features(df_dummy)
        
        print("\n=== HASIL EKSTRAKSI ===")
        print(f"Bentuk Dataframe Output: {hasil_tsfel.shape[0]} baris x {hasil_tsfel.shape[1]} kolom")
        
        print("\nMenampilkan 5 Kolom Pertama (sebagai sampel):")
        # Hanya mencetak 5 kolom pertama agar terminal tidak kepenuhan teks
        print(hasil_tsfel.iloc[:, :5].head()) 
        
        print("\n[SUKSES] Modul build_tsfel.py Anda berfungsi sempurna!")
        
    except Exception as e:
        print(f"\n[GAGAL] Terjadi error: {e}")