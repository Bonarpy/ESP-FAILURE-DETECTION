import pandas as pd
import numpy as np
import os

def load_and_extract_data():
    print("--- Part 1: Memulai Ekstraksi 11 Parameter Utama ---")
    
    # 1. Setup Jalur File (Cross-Platform Compatibility)
    # Menggunakan os.path.join agar script tidak rusak saat dipindah antar OS
    base_dir = os.getcwd()
    raw_data_path = os.path.join(base_dir, 'Data', 'Raw', 'dailyData.csv')
    
    # 2. Definisi Mapping Kolom
    # Kita gunakan dictionary agar mapping antara nama kolom mentah (Raw) 
    # dan nama variabel (Clean) terlihat transparan dan mudah diubah.
    column_mapping = {
        'Well_ID': 'Well_ID',
        'DOWN_TIME_HOURS': 'Downtime',
        'ESP Data - Output Frequency': 'Freq',
        'ESP Data - Drive Current': 'Current',
        'ESP Data - Vibration X': 'Vibration',
        'ESP Data - Intake Pressure': 'Intake_Press',
        'ESP Data - Motor Winding Temperature': 'Motor_Temp',
        'Gas_Saturation_in_Pump': 'Gas_Sat',
        'Pump_Delta_Pressure': 'Delta_Press',
        'Power_Difference': 'Pwr_Diff',
        'Liquid_Intake': 'Liq_Intake'
    }

    # 3. Load Data dengan 'usecols'
    # Teknik Pro: Kita tidak meload seluruh CSV ke RAM, 
    # melainkan hanya kolom yang kita butuhkan saja. Ini menghemat memori laptop Anda.
    try:
        df = pd.read_csv(raw_data_path, usecols=column_mapping.keys())
        
        # 4. Rename Kolom
        df = df.rename(columns=column_mapping)
        
        print(f"✅ Data berhasil dimuat.")
        print(f"   - Total Baris: {df.shape[0]}")
        print(f"   - Total Kolom: {df.shape[1]}")
        
        # Menampilkan ringkasan awal data
        print("\nPratinjau 5 Baris Pertama:")
        print(df.head())
        
        return df

    except FileNotFoundError:
        print(f"❌ ERROR: File tidak ditemukan di {raw_data_path}. Pastikan struktur folder sudah benar.")
        return None

# Eksekusi Part 1
df = load_and_extract_data()

def handle_missing_values(df):
    print("\n--- Part 2: Memulai Penanganan Missing Value (Restorasi) ---")
    
    # 1. Analisis Awal: Cek berapa banyak 'lubang' di data kita
    null_report = df.isnull().sum()
    print("Jumlah Missing Value awal:")
    print(null_report[null_report > 0])

    # 2. Penanganan Kolom Target (Downtime)
    # Jika label target hilang, baris tersebut tidak punya nilai edukatif bagi AI.
    # Kita hapus baris yang Downtime-nya NaN.
    initial_count = len(df)
    df = df.dropna(subset=['Downtime'])
    print(f"\n- Menghapus {initial_count - len(df)} baris karena 'Downtime' kosong.")

    # 3. Restorasi Fitur Sensor (Grouped Interpolation)
    # Kita tidak boleh mencampur data antar sumur saat interpolasi.
    sensor_features = [
        'Freq', 'Current', 'Vibration', 'Intake_Press', 
        'Motor_Temp', 'Gas_Sat', 'Delta_Press', 'Pwr_Diff', 'Liq_Intake'
    ]

    print(f"- Melakukan interpolasi linear pada {len(sensor_features)} fitur sensor...")
    
    for col in sensor_features:
        # Penjelasan fungsi:
        # groupby('Well_ID') -> Mengisolasi tiap sumur agar tidak terkontaminasi data sumur lain
        # transform -> Menerapkan fungsi dan mengembalikan hasil ke index asli
        # interpolate -> Menghitung nilai antara dua titik (linear)
        df[col] = df.groupby('Well_ID')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )

    # 4. Final Patching (Cadangan)
    # Jika ada sumur yang memiliki blok kosong di awal/akhir, kita gunakan median sumur tersebut
    # agar tetap memiliki nilai yang representatif terhadap 'kebiasaan' sumur tersebut.
    for col in sensor_features:
        df[col] = df[col].fillna(df.groupby('Well_ID')[col].transform('median'))

    # 5. Verifikasi Akhir
    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls == 0:
        print("✅ SUKSES: Semua missing value telah tertangani.")
    else:
        # Jika masih ada, biasanya karena satu Well_ID isinya NaN semua di kolom tertentu
        print(f"⚠️ PERINGATAN: Masih ada {remaining_nulls} nilai kosong. Melakukan drop baris sisa...")
        df.dropna(inplace=True)

    print(f"   Total baris setelah Part 2: {len(df)}")
    return df

# Eksekusi Part 2
df = handle_missing_values(df)

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

def handle_outliers_iqr_log(df):
    print("\n--- Part 3: Outlier Handling (Log Transform + IQR 1.5) ---")
    
    # 1. Log Transformation (Merapatkan Skala)
    # Sangat penting untuk fitur yang skewness-nya tinggi seperti Vibration/Pwr_Diff
    cols_to_log = ['Vibration', 'Current', 'Pwr_Diff', 'Delta_Press']
    print(f"- Menerapkan Log1p pada: {cols_to_log}")
    for col in cols_to_log:
        df[col] = np.log1p(df[col])

    # 2. IQR Filtering (Membersihkan Sisa Outlier)
    # Kita terapkan ke semua parameter fitur
    features = [
        'Freq', 'Current', 'Vibration', 'Intake_Press', 
        'Motor_Temp', 'Gas_Sat', 'Delta_Press', 'Pwr_Diff', 'Liq_Intake'
    ]
    
    initial_rows = len(df)
    
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        
        # Filter: Simpan hanya data yang masuk dalam batas IQR
        df = df[(df[col] >= lower_limit) & (df[col] <= upper_limit)]
    
    print(f"✅ Selesai! Menghapus {initial_rows - len(df)} baris outlier statistik.")

    # 3. Visualisasi Akhir (Pembuktian)
    plt.figure(figsize=(15, 6))
    df.boxplot(column=['Freq', 'Current', 'Vibration', 'Motor_Temp'])
    plt.title("Data SETELAH Log + IQR Filtering (Batas 1.5)")
    plt.show()

    return df

# Jalankan Part 3
df = handle_outliers_iqr_log(df)

#-------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
import os

def normalize_features(df):
    print("\n--- Part 4: Memulai Normalisasi (Min-Max Scaling 0-1) ---")
    
    # 1. Seleksi Kolom Fitur
    # Kita HANYA menormalisasi fitur sensor. 
    # 'Well_ID' (identitas) dan 'Downtime' (label target) tidak boleh disentuh!
    feature_cols = [
        'Freq', 'Current', 'Vibration', 'Intake_Press', 
        'Motor_Temp', 'Gas_Sat', 'Delta_Press', 'Pwr_Diff', 'Liq_Intake'
    ]
    
    # 2. Inisialisasi Scaler
    # feature_range=(0, 1) memastikan batas bawah adalah 0 dan batas atas adalah 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 3. Eksekusi Transformasi
    # Kita buat salinan dataframe (df_final) agar data asli sebelum normalisasi tetap ada di memori
    df_final = df.copy()
    
    print(f"- Menormalisasi {len(feature_cols)} kolom fitur...")
    df_final[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # 4. Verifikasi Detail (Audit)
    # Kita cek apakah benar-benar sudah 0 dan 1. 
    # Jika ada angka yang minus atau lebih dari 1, berarti ada error di proses sebelumnya.
    check_stats = df_final[feature_cols].agg(['min', 'max']).T
    print("\nVerifikasi Skala Hasil Normalisasi:")
    print(check_stats)
    
    # 5. Penyimpanan Data 'Processed' (Hasil Akhir Preprocessing)
    # Folder Processed adalah tempat data yang sudah 'siap tempur' oleh AI
    output_dir = os.path.join(os.getcwd(), 'Data', 'Processed')
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = 'esp_data_final_preprocessed.csv'
    output_path = os.path.join(output_dir, file_name)
    df_final.to_csv(output_path, index=False)
    
    print(f"\n✅ SUKSES! Data final telah dinormalisasi dan disimpan di: {output_path}")
    print(f"   Total Data: {df_final.shape[0]} baris dan {df_final.shape[1]} kolom.")
    
    return df_final

# Jalankan Part 4
df_final = normalize_features(df)

#-------------------------------------------------------------------

from imblearn.over_sampling import SMOTE
import pandas as pd

def balance_data(df):
    print("\n--- Part 5: Memulai Data Balancing (SMOTE) ---")
    
    # 1. Konversi Downtime menjadi Label Biner (Target)
    # Kita anggap jika ada jam downtime, berarti terjadi kerusakan/anomali
    df['Target'] = df['Downtime'].apply(lambda x: 1 if x > 0 else 0)
    
    # 2. Pisahkan Fitur (X) dan Target (y)
    feature_cols = [
        'Freq', 'Current', 'Vibration', 'Intake_Press', 
        'Motor_Temp', 'Gas_Sat', 'Delta_Press', 'Pwr_Diff', 'Liq_Intake'
    ]
    X = df[feature_cols]
    y = df['Target']
    
    print(f"Distribusi Target Awal: \n{y.value_counts()}")

    # 3. Eksekusi SMOTE
    # k_neighbors=5 adalah standar untuk mencari tetangga data terdekat
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # 4. Gabungkan kembali menjadi DataFrame yang seimbang
    df_balanced = pd.DataFrame(X_res, columns=feature_cols)
    df_balanced['Target'] = y_res
    
    print(f"\nDistribusi Target Setelah SMOTE: \n{df_balanced['Target'].value_counts()}")
    
    # 5. Simpan file final untuk TRAINING
    final_path = os.path.join(os.getcwd(), 'Data', 'Processed', 'dataset_siap_training.csv')
    df_balanced.to_csv(final_path, index=False)
    
    print(f"\n✅ SEMUA TAHAP PREPROCESSING SELESAI!")
    print(f"File siap untuk AI Training: {final_path}")
    
    return df_balanced

# Jalankan Part 5
df_final_balanced = balance_data(df_final)