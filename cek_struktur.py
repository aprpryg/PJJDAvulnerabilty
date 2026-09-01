import os
try:
    from dbfread import DBF
except ImportError:
    print("Silakan install dbfread terlebih dahulu: pip install dbfread")
    exit()

# Daftar file yang akan diprofiling
files = [
    "ssn202509_kp_blok41.dbf",
    "ssn202509_kp_blok42.dbf",
    "ssn202509_kp_blok43.dbf",
    "ssn202509_mod_rt.dbf",
    "ssn202509_mod_ind.dbf"
]

output_file = "hasil_profiling.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for file in files:
        if not os.path.exists(file):
            pesan = f"FILE TIDAK DITEMUKAN: {file}\n\n"
            print(pesan)
            f.write(pesan)
            continue
            
        try:
            print(f"Membaca {file}...")
            # load=False membaca secara streaming, aman untuk file bergiga-giga
            table = DBF(file, load=False) 
            
            f.write(f"=== {file.upper()} ===\n")
            f.write(f"Total baris : {len(table)}\n")
            f.write(f"Total kolom : {len(table.fields)}\n\n")
            
            # 1. Ekstrak Metadata Kolom
            f.write("STRUKTUR KOLOM:\n")
            f.write(f"{'Nama Field':<15} | {'Tipe':<4} | {'Panjang':<7}\n")
            f.write("-" * 35 + "\n")
            for field in table.fields:
                f.write(f"{field.name:<15} | {field.type:<4} | {field.length:<7}\n")
            
            # 2. Cari Kandidat Key
            kandidat_key = ['PSU', 'SSU', 'STRATA', 'NKS', 'RUTA', 'URUT', 'ART']
            keys_ditemukan = [field.name for field in table.fields if field.name in kandidat_key]
            
            f.write("\nKANDIDAT KEY DITEMUKAN:\n")
            if keys_ditemukan:
                f.write(", ".join(keys_ditemukan) + "\n")
            else:
                f.write("Tidak ada key standar BPS yang ditemukan\n")
            
            # 3. Ambil 2 Baris Pertama sebagai Sample
            f.write("\nSAMPLE DATA (2 Baris Pertama):\n")
            for i, record in enumerate(table):
                if i >= 2:
                    break
                f.write(str(dict(record)) + "\n")
                
            f.write("\n" + "="*60 + "\n\n")
            
        except Exception as e:
            f.write(f"GAGAL MEMBACA {file}: {e}\n\n")

print(f"\nSelesai! Buka file '{output_file}' di VSCode untuk melihat hasilnya.")