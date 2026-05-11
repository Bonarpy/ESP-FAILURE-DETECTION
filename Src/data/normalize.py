import pandas as pd
import numpy as np
import os

print("1. Membaca Data Interim (Hasil pembersihan Step 2)...")
# Membaca file interim yang udah lu bikin sebelumnya
df = pd.read_csv('/Users/kevinsantosap/ESP-FAILURE-DETECTION/Data/Interim/interim_data.csv')

print("2. Membuat Label Target untuk AI...")
# AI butuh tahu kapan mesin rusak. Kalau ada Downtime, kita kasih label 1 (Anomali)
df['Is_Anomaly'] = (df['Downtime'] > 0).astype(int)

print("3. Memulai Proses Scaling & Normalisasi...")
# Tipe A: Ditekan jadi skala 0 sampai 1 (Dibagi nilai maksimalnya)
df['Freq_Norm'] = df['Freq'] / (df['Freq'].max() + 1e-9)
df['Vibration_Norm'] = df['Vibration'] / (df['Vibration'].max() + 1e-9)
df['Gas_Sat_Norm'] = df['Gas_Sat'] / (df['Gas_Sat'].max() + 1e-9)

# Tipe B: Dibagi kebiasaan / nilai tengah (median) masing-masing sumur
# Biar AI tahu batas wajar tiap sumur itu beda-beda
kolom_median = ['Current', 'Intake_Press', 'Motor_Temp', 'Delta_Press', 'Pwr_Diff', 'Liq_Intake']

for col in kolom_median:
    df[f'{col}_Norm'] = df.groupby('Well_ID')[col].transform(lambda x: x / (x.median() + 1e-9))

print("4. Membersihkan sisa error pembagian...")
# Jaga-jaga kalau ada pembagian dengan angka 0 yang bikin hasil jadi tak terhingga (inf)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

# Buang kolom mentah yang angkanya masih gede-gede biar AI nggak bingung
kolom_mentah_dibuang = [
    'Freq', 'Current', 'Vibration', 'Intake_Press', 'Motor_Temp', 
    'Gas_Sat', 'Delta_Press', 'Pwr_Diff', 'Liq_Intake', 'Downtime'
]
df = df.drop(columns=kolom_mentah_dibuang)

print("5. Menyimpan Data Final...")
os.makedirs('/Users/kevinsantosap/ESP-FAILURE-DETECTION/Data/Processed', exist_ok=True)
df.to_csv('/Users/kevinsantosap/ESP-FAILURE-DETECTION/Data/Processed/processed_data.csv', index=False)

print("✅ SUKSES BESAR! Data udah matang. Cek folder 'data/processed/processed_data.csv'")