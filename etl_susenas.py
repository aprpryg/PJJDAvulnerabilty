import pandas as pd
from dbfread import DBF
import os

print("Memulai proses ETL SUSENAS September 2025...")

# --- 1. EXTRACT ---
def load_dbf_to_df(filepath):
    print(f"Membaca {filepath}...")
    table = DBF(filepath, load=True)
    return pd.DataFrame(iter(table))

df_kp43 = load_dbf_to_df('ssn202509_kp_blok43.dbf')
df_mod_rt = load_dbf_to_df('ssn202509_mod_rt.dbf')
df_mod_ind = load_dbf_to_df('ssn202509_mod_ind.dbf')

# --- 2. TRANSFORM ---

# A. Modul Individu (KRT)
print("Memproses data individu (KRT)...")
df_krt = df_mod_ind[df_mod_ind['M403'] == 1].copy()
# Baris di bawah ini yang sebelumnya hilang:
cols_krt = ['PSU', 'SSU', 'URUT', 'M405', 'M407', 'M506', 'M605_A']
df_krt = df_krt[cols_krt]

df_krt.rename(columns={
    'M405': 'krt_jk',
    'M407': 'krt_umur',
    'M506': 'krt_pendidikan',
    'M605_A': 'krt_bekerja'
}, inplace=True)

# B. KP Blok 43
print("Memproses data pengeluaran dan nutrisi...")
cols_kp = ['PSU', 'SSU', 'URUT', 'EXPEND', 'KAPITA', 'CAL', 'PROT']
df_kp43 = df_kp43[cols_kp]

# C. Modul RT (dengan tambahan M101 untuk Provinsi)
print("Memproses data rumah tangga dan perlindungan sosial...")
cols_rt = ['PSU', 'SSU', 'URUT', 'M101', 'M1501', 'M1504', 'M1701', 'M1703', 'M1704', 'FWT']
df_mod_rt = df_mod_rt[cols_rt]

df_mod_rt['penerima_bansos'] = (
    (df_mod_rt['M1701'] == 1) | 
    (df_mod_rt['M1703'] == 1) | 
    (df_mod_rt['M1704'] == 1)
).astype(int)

# --- 3. MERGE & LOAD ---
print("Menggabungkan dataset...")
df_merged = pd.merge(df_mod_rt, df_kp43, on=['PSU', 'SSU', 'URUT'], how='inner')
df_final = pd.merge(df_merged, df_krt, on=['PSU', 'SSU', 'URUT'], how='left')

output_filename = 'susenas_vulnerability_dataset.csv'
df_final.to_csv(output_filename, index=False)

print(f"\nETL Selesai! Dataset berhasil disimpan sebagai '{output_filename}'")