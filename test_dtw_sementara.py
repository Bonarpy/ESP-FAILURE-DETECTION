import pandas as pd
import numpy as np
from Src.features.build_dtw import calculate_dtw_distances

print("=== MENGUJI MODUL BUILD_DTW.PY ===")

# 1. Buat Data Asal-asalan (Seolah-olah dari Anggota 1)
df_dummy = pd.DataFrame({
    'Freq_Norm': np.random.rand(50),  # 50 baris data dummy
    'Load_Norm': np.random.rand(50)
})

# 2. Buat Klaster Anomali Dummy (Seolah-olah dari Anggota 2)
# Array 3D: 6 klaster, 1 timestep, 2 fitur (Freq_Norm & Load_Norm)
dummy_centers = np.random.rand(6, 1, 2)

# 3. Panggil fungsi yang baru Anda buat!
try:
    hasil_dtw = calculate_dtw_distances(df_dummy, dummy_centers)
    
    print("\n=== HASIL EKSTRAKSI ===")
    print("Bentuk Dataframe Output:", hasil_dtw.shape)
    print("5 Baris Pertama:")
    print(hasil_dtw.head())
    print("\n[SUKSES] Modul build_dtw.py Anda berfungsi sempurna!")
    
except Exception as e:
    print(f"\n[GAGAL] Terjadi error: {e}")
