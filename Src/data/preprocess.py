import pandas as pd
import numpy as np
import os

def process_and_clean_data(input_filepath, output_filepath):
    print("Membaca dan membersihkan data mentah")
    df = pd.read_csv(input_filepath)
    df = df[['Well_ID', 'ESP Data - Output Frequency', 'ESP Data - Drive Current', 'DOWN_TIME_HOURS']]
    df.columns = ['Well_ID', 'Frequency', 'Load', 'Downtime']
    
    print("Menambal data kosong (Interpolasi)")
    df['Frequency'] = df['Frequency'].interpolate(method='linear')
    df['Load'] = df['Load'].interpolate(method='linear')
    df.dropna(inplace=True)
    
    print("Membuat label & Normalisasi skala")
    df['Is_Anomaly'] = (df['Downtime'] > 0).astype(int)
    freq_max = df['Frequency'].max()
    df['Freq_Norm'] = df['Frequency'] / freq_max
    df['Load_Norm'] = df.groupby('Well_ID')['Load'].transform(lambda x: x / x.median())
    
    # Bersihkan error dari pembagian
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    df_final = df[['Well_ID', 'Freq_Norm', 'Load_Norm', 'Is_Anomaly']]
    
    # Pastikan folder tempat menyimpan data sudah ada
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df_final.to_csv(output_filepath, index=False)
    print(f"-> Selesai! Data bersih disimpan di {output_filepath}\n")
    return df_final

