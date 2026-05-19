## 👥 Alur Kerja & Kontribusi Tim (Modular Pipeline)

Proyek ini terbagi menjadi beberapa tahapan modular yang saling terhubung erat lewat berkas CSV fisik di dalam folder `Data/Processed/`:

### 🏗️ 1. Tahap Data Engineering (Anggota 1 - `preprocessing.py`)

Mengekstrak 11 parameter utama sumur, melakukan **Restorasi Missing Value** berbasis *Grouped Linear Interpolation* per sumur minyak (`Well_ID`) agar tidak terjadi kontaminasi data antar sumur. Dilanjutkan dengan pembersihan pencilan statistik menggunakan kombinasi **Log1p Transformation & IQR (Batas 1.5)**, serta normalisasi skala fitur (0-1) menggunakan **MinMaxScaler**.

### 🔍 2. Tahap Anomaly Library (Anggota 2)

Mengidentifikasi pusat-pusat pola anomali historis pada operasional pompa ESP yang akan dijadikan pustaka acuan (*anomaly centers*) untuk perhitungan kemiripan gelombang sensor.

### 📐 3. Tahap Feature Engineering (Anggota 3 - `tsfel_features.csv` & `dtw_features.csv`)

Mengekstrak ratusan fitur canggih dari data sensor bersih:

* **TSFEL (Time Series Feature Extraction Library):** Mengekstrak fitur dari domain statistik, temporal, dan frekuensi (seperti *absolute energy, entropy, kurtosis, skewness*, dll).
* **DTW (Dynamic Time Warping):** Menghitung jarak linguistik/kemiripan gelombang sensor sumur saat ini terhadap pustaka pusat anomali (Anggota 2) untuk mendeteksi pergeseran pola sebelum kerusakan terjadi.

### 🧠 4. Tahap Machine Learning (Anggota 4 - `train_xgboost.py`)

Membangun dan melatih model **XGBoost Classifier**. Menggunakan teknik pembagian data berbasis waktu (*Time-Based Split*) tanpa acak (`shuffle=False`) untuk mencegah kebocoran data masa depan (*data leakage*). Menggunakan **SMOTE** khusus pada *Train Set* untuk mengatasi ketimpangan kelas, serta melakukan otomatisasi pencarian parameter terbaik melalui **RandomizedSearchCV** dengan optimasi utama berfokus pada nilai **Recall** (meminimalisir alarm kerusakan yang lolos).

### 🎛️ 5. Tahap Sistem Integrasi (Anggota 5 / Manajer - `main.py`)

Bertindak sebagai konduktor utama yang menjalankan pipeline secara otomatis dari hulu ke hilir. `main.py` akan memanggil proses preprocessing secara eksternal, menarik data fitur milik Anggota 3, menggabungkannya secara dinamis menjadi tabel raksasa `X_final` (203 kolom), lalu memanggil otak AI XGBoost untuk mengeluarkan berkas prediksi akhir kerusakannya.

---

## 🛠️ Persyaratan Sistem (Prerequisites)

Pastikan pustaka Python berikut sudah terpasang di komputer Anda sebelum menjalankan sistem:

```bash
pip install pandas numpy xgboost scikit-learn imbalanced-learn matplotlib seaborn pyyaml tsfel

```

---

## 🚀 Cara Menjalankan Sistem (How to Run)

Sistem ini memiliki dua mode operasi utama:

### A. Fase Pelatihan AI (Training Phase)

Jika Anda ingin melatih ulang otak AI menggunakan fitur baru hasil ekstraksi Anggota 3, jalankan perintah berikut di terminal Anda:

```bash
python Src/models/train_xgboost.py

```

*Proses ini akan membaca fitur TSFEL & DTW, mengeksekusi RandomizedSearchCV, menampilkan rapor performa AI (Akurasi, Presisi, Recall), dan memperbarui file otak AI di folder `models/xgboost_production.json` secara otomatis.*

### B. Fase Produksi / Prediksi Otomatis (Production Pipeline)

Untuk menjalankan seluruh ekosistem secara otomatis dari pembersihan data mentah hingga pengeluaran hasil prediksi kerusakan akhir, cukup jalankan berkas pemandu utama:

```bash
python main.py

```

*Catatan: Saat Tahap 1 berjalan, jendela grafik Boxplot dari hasil preprocessing akan muncul (Pop-up). Silakan klik tombol silang (X) pada jendela grafik tersebut untuk mengizinkan sistem melanjutkan langkah otomatis ke TSFEL, DTW, dan Prediksi XGBoost hingga selesai.*

---

## 📊 Hasil Akhir (Outputs)

Setelah `main.py` selesai dieksekusi dengan sukses, berkas laporan akhir prediksi potensi kerusakan pompa ESP akan tersimpan rapi di dalam jalur direktori yang telah dikonfigurasi pada berkas `configs/config.yaml`.
