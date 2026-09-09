# 🇮🇩 Household Vulnerability & Social Protection Monitor

Proyek ini adalah *Dashboard* Analitik interaktif berbasis **Streamlit** untuk memetakan kesenjangan perlindungan sosial (*Social Protection Gap*) dan memprediksi profil risiko kerentanan rumah tangga menggunakan algoritma *Machine Learning*. 

Analisis ini menggunakan dataset **Survei Sosial Ekonomi Nasional (SUSENAS) September 2025** (Modul Kesehatan & Perumahan, Konsumsi Pengeluaran, dan Modul Individu).

**🌐 [Lihat Live Dashboard Disini](https://pjjda-vulnerabilty-djsef.streamlit.app/)** *(Ganti dengan URL asli Anda jika berbeda)*

---

## 🎯 Tujuan Proyek
Proyek *Action Learning* ini **tidak** dirancang untuk mengukur dampak kausalitas dari program bantuan sosial, melainkan berfungsi sebagai alat diagnostik alokasi dan prediksi risiko dengan tujuan:
1. **Membangun Indeks Kerentanan Multidimensi:** Mendefinisikan kerentanan berdasarkan 3 deprivasi dasar (Ekonomi, Pangan/Gizi, dan Infrastruktur Perumahan).
2. **Memetakan Social Protection Gap:** Mendeteksi *exclusion error* (rumah tangga rentan yang luput dari bansos) dan *residual vulnerability* (rumah tangga yang menerima bansos namun tetap rentan).
3. **Membangun Predictive Risk Profiling:** Menggunakan model *Machine Learning* untuk mengekstrak faktor pendorong utama (*Top Vulnerability Drivers*) guna penargetan kebijakan berbasis bukti (*evidence-based targeting*).

---

## 📂 Struktur Repositori & Alur Kerja (Data Pipeline)

Proyek ini dibangun melalui 5 tahapan pemrosesan data (dari data mentah hingga *deployment*), yang direpresentasikan oleh *file* berikut secara berurutan:

### 1. Data Discovery
* **`cek_struktur.py`** 
  Skrip utilitas untuk melakukan *data profiling* awal (membaca struktur kolom, mendeteksi *primary key*, dan mengekstrak metadata) dari file `.dbf` SUSENAS berukuran besar tanpa membebani memori (*streaming read*).

### 2. Proses ETL (Extract, Transform, Load)
* **`etl_susenas.py`**
  Skrip untuk mengekstrak data dari berbagai modul SUSENAS (`kp_blok43`, `mod_rt`, `mod_ind`), melakukan transformasi variabel demografi dan pengeluaran, lalu menggabungkannya berdasarkan kunci BPS (`PSU`, `SSU`, `URUT`) menjadi satu *dataset* analitik (`susenas_vulnerability_dataset.csv`).

### 3. Exploratory Data Analysis & Modeling
* **`analisis_kerentanan.ipynb`**
  Jupyter Notebook yang mendokumentasikan eksperimen analitik: pembuatan *rule-based multidimensional target*, rekayasa fitur (*One-Hot Encoding* untuk variabel provinsi dan pendidikan), serta evaluasi metrik (ROC-AUC, Precision, Recall) antara algoritma *Random Forest* dan *Logistic Regression* sebagai *baseline*.

### 4. Model Serialization (Optimasi Cloud)
* **`save_artifacts.py`**
  Skrip untuk melatih ulang model *Random Forest* terbaik dan menyimpannya beserta seluruh metrik pendukung ke dalam *file* biner ringan (`model_artifacts.joblib`). Langkah *pre-computation* ini mengeliminasi masalah *CPU throttling* saat aplikasi di-*deploy* di *cloud*.

### 5. Deployment Dashboard
* **`app.py`**
  Skrip *frontend* utama menggunakan Streamlit. Terdiri dari 4 tab analitik (Overview, Vulnerability Gap Matrix, Model Insights, dan Risk Simulator) yang dirancang agar hasil pemodelan *Machine Learning* dapat diinterpretasikan dengan mudah oleh pembuat kebijakan (audiens non-teknis).

---

## 🛠️ Metodologi Machine Learning
* **Algoritma Utama:** Random Forest Classifier (Dioptimasi dengan `class_weight='balanced'` untuk menangani ketidakseimbangan target).
* **Baseline Model:** Logistic Regression.
* **Explainability:** Menggunakan *Grouped Feature Importance* (MDI) untuk mengagregasi fitur spasial (Provinsi) dan demografi (Pendidikan) agar bobot risiko struktural mudah diinterpretasikan tanpa kebingungan kardinalitas.

---

## 💻 Cara Menjalankan Aplikasi Secara Lokal

Jika Anda ingin menjalankan *dashboard* ini di komputer lokal, ikuti langkah berikut:

1. *Clone* repositori ini:
   ```bash
   git clone [https://github.com/username-anda/nama-repo-anda.git](https://github.com/username-anda/nama-repo-anda.git)
   cd nama-repo-anda