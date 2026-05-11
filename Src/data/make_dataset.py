import pandas as pd
import numpy as np
import os

print("1. Memulai proses pembersihan...")

# Bikin folder otomatis untuk menyimpan hasilnya
os.makedirs('//Users//kevinsantosap//ESP-FAILURE-DETECTION//Data//Interim', exist_ok=True)

# Membaca file CSV Anda
print("2. Membaca file dailyData.csv...")
df = pd.read_csv('//Users//kevinsantosap//ESP-FAILURE-DETECTION//Data//Raw//dailyData.csv', usecols=[
    'Well_ID', 'DOWN_TIME_HOURS', 'ESP Data - Output Frequency', 
    'ESP Data - Drive Current', 'ESP Data - Vibration X', 
    'ESP Data - Intake Pressure', 'ESP Data - Motor Winding Temperature', 
    'Gas_Saturation_in_Pump', 'Pump_Delta_Pressure', 'Power_Difference', 'Liquid_Intake'
])

# Mengganti nama kolom biar lebih pendek
df.columns = ['Well_ID', 'Downtime', 'Freq', 'Current', 'Vibration', 'Intake_Press', 'Motor_Temp', 'Gas_Sat', 'Delta_Press', 'Pwr_Diff', 'Liq_Intake']
    
# Membuang data sensor yang rusak (nilai minus)
print("3. Membuang data error dan menambal yang kosong...")
df.loc[df['Freq'] < 0, 'Freq'] = np.nan
df.loc[df['Vibration'] < 0, 'Vibration'] = np.nan

# Menambal data yang kosong (NaN)
for col in df.columns:
    if col not in ['Well_ID', 'Downtime']:
        df[col] = df[col].interpolate().bfill()
df.dropna(inplace=True)

# Menyimpan hasil akhirnya
print("4. Menyimpan data bersih...")
df.to_csv('//Users//kevinsantosap//ESP-FAILURE-DETECTION//Data//Interim//interim_data.csv', index=False)

print("✅ SUKSES! Silakan cek folder 'data' di sebelah kiri.")