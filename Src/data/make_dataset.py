import pandas as pd
import os

def clean_raw_data(input_path, output_path):
    """
    Fungsi untuk membaca data mentah, mengambil kolom penting, 
    dan menambal data sensor yang kosong (interpolasi).
    """
    print(f"[Data Engineer] Membaca data mentah dari: {input_path}")
    
    # 1. Pastikan folder output-nya ada, kalau belum ada kita buatkan
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 2. Baca data mentah (hanya 4 kolom utama)
    df = pd.read_csv(input_path, usecols=['Well_ID', 'ESP Data - Output Frequency', 'ESP Data - Drive Current', 'DOWN_TIME_HOURS'])
    df.columns = ['Well_ID', 'Frequency', 'Load', 'Downtime']
    
    print("[Data Engineer] Melakukan interpolasi pada data yang kosong...")
    # 3. Tambal data yang bolong
    df['Frequency'] = df['Frequency'].interpolate().bfill()
    df['Load'] = df['Load'].interpolate().bfill()
    df.dropna(inplace=True)
    
    # 4. Simpan hasilnya sebagai data "setengah matang" (interim)
    df.to_csv(output_path, index=False)
    print(f"[Data Engineer] Tahap 1 Selesai! Data bersih disimpan di: {output_path}")
    
    return df

# BLOK PENGUJIAN (Agar bisa di-run langsung di terminal)
if __name__ == "__main__":
    print("=== MENGUJI MODUL MAKE_DATASET.PY ===")
    # Asumsi file dailyData.csv ada di folder utama proyek Anda
    file_mentah = "C:\\Users\\Iman Fath Hatta\\ESP-FAILURE-DETECTION\\Data\\Raw\\dailyData.csv"  # Sesuaikan path ini jika perlu
    file_interim = "C:\\Users\\Iman Fath Hatta\\ESP-FAILURE-DETECTION\\Data\\Interim\\interim_data.csv"
    
    try:
        df_bersih = clean_raw_data(file_mentah, file_interim)
        print("\nBentuk data:", df_bersih.shape)
        print("5 Baris pertama:\n", df_bersih.head())
    except Exception as e:
        print(f"Error: {e}")