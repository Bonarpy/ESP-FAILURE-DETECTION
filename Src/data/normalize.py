import pandas as pd
import numpy as np
import os

def normalize_features(input_path, output_path):
    """
    Fungsi untuk membaca data interim, membuat label anomali, 
    dan menormalisasi nilai Frekuensi & Beban berdasarkan median sumur.
    """
    print(f"[Data Engineer] Membaca data interim dari: {input_path}")
    
    # 1. Pastikan folder output-nya ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 2. Baca data dari tahap sebelumnya
    df = pd.read_csv(input_path)
    
    print("[Data Engineer] Membuat Label Target dan Menormalisasi Skala...")
    # 3. Buat Label Target (1 = Anomali, 0 = Normal)
    df['Is_Anomaly'] = (df['Downtime'] > 0).astype(int)
    
    # 4. Normalisasi
    df['Freq_Norm'] = df['Frequency'] / df['Frequency'].max()
    df['Load_Norm'] = df.groupby('Well_ID')['Load'].transform(lambda x: x / x.median())
    
    print("[Data Engineer] Melakukan sanitasi akhir (Pembersihan inf/NaN)...")
    # 5. Pembersihan Akhir (Sanitasi inf dan NaN jadi 0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # 6. Simpan hasil akhir (Data Matang)
    df.to_csv(output_path, index=False)
    print(f"[Data Engineer] Tahap 2 Selesai! Data matang siap pakai disimpan di: {output_path}")
    
    return df

# BLOK PENGUJIAN (Agar bisa di-run langsung di terminal)
if __name__ == "__main__":
    print("=== MENGUJI MODUL NORMALIZE.PY ===")
    file_interim = "C:\\Users\\Iman Fath Hatta\\ESP-FAILURE-DETECTION\\Data\\Interim\\interim_data.csv"
    file_matang = "C:\\Users\\Iman Fath Hatta\\ESP-FAILURE-DETECTION\\Data\\Processed\\processed_data.csv"

    try:
        df_matang = normalize_features(file_interim, file_matang)
        print("\nDistribusi Label Anomali:")
        print(df_matang['Is_Anomaly'].value_counts())
        print("\n5 Baris pertama Fitur Normalisasi:\n", df_matang[['Well_ID', 'Freq_Norm', 'Load_Norm', 'Is_Anomaly']].head())
    except Exception as e:
        print(f"Error: {e}")