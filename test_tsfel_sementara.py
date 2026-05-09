import pandas as pd
import numpy as np
from Src.features.build_tsfel import extract_tsfel_features

print("=== MENGUJI MODUL BUILD_TSFEL.PY ===")

# 1. Buat Data Dummy
# Sengaja kita buat nilai yang identik di Load_Norm untuk memancing TSFEL menghasilkan nilai inf/NaN
df_dummy = pd.DataFrame({
    'Freq_Norm': [0.5, 0.6, 0.7, 0.8, 0.9],
    'Load_Norm': [1.0, 1.0, 1.0, 1.0, 1.0] # Data konstan (Varians = 0)
})

print("Data Input:")
print(df_dummy)
print("-" * 30)

# 2. Panggil fungsi dari modul yang baru Anda buat
try:
    hasil_tsfel = extract_tsfel_features(df_dummy)
    
    print("\n=== HASIL EKSTRAKSI ===")
    print(f"Bentuk Dataframe Output: {hasil_tsfel.shape}")
    
    # Kita cek apakah fungsi pembersihan inf/NaN bekerja
    jumlah_inf = np.isinf(hasil_tsfel).sum().sum()
    jumlah_nan = hasil_tsfel.isna().sum().sum()
    
    print(f"\nPengecekan Sanitasi:")
    print(f"- Jumlah Infinity: {jumlah_inf} (Harusnya 0)")
    print(f"- Jumlah NaN: {jumlah_nan} (Harusnya 0)")
    
    if jumlah_inf == 0 and jumlah_nan == 0:
        print("\n[SUKSES] Modul build_tsfel.py Anda berfungsi sempurna dan datanya bersih!")
    else:
        print("\n[PERINGATAN] Masih ada data kotor!")
        
except Exception as e:
    print(f"\n[GAGAL] Terjadi error: {e}")