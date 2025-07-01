import random
import time
import os

def bersihkan_layar():
    """Membersihkan layar terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')

def dapatkan_info_pemain():
    """Meminta jumlah pemain dan nama mereka."""
    while True:
        try:
            # Game lebih menarik dengan minimal 2 pemain untuk fitur menembak orang lain
            jumlah_pemain = int(input("Masukkan jumlah pemain (2-4): "))
            if 2 <= jumlah_pemain <= 4:
                break
            else:
                print("Jumlah pemain harus antara 2 hingga 4.")
        except ValueError:
            print("Masukkan angka yang valid.")

    nama_pemain = []
    for i in range(jumlah_pemain):
        while True:
            nama = input(f"Masukkan nama Pemain {i+1}: ").strip()
            if nama and nama not in nama_pemain:
                nama_pemain.append(nama)
                break
            elif nama in nama_pemain:
                print("Nama sudah digunakan. Gunakan nama lain.")
            else:
                print("Nama tidak boleh kosong.")
    return nama_pemain

def dapatkan_pengaturan_game():
    """Meminta dealer memasukkan jumlah peluru."""
    print("\n--- Pengaturan oleh Dealer ---")
    while True:
        try:
            jumlah_peluru = int(input("Masukkan jumlah peluru yang akan dimasukkan ke revolver (1-11): "))
            if 1 <= jumlah_peluru <= 11: # Maksimal 11 agar permainan tidak langsung berakhir
                return jumlah_peluru
            else:
                print("Jumlah peluru harus antara 1 hingga 11.")
        except ValueError:
            print("Masukkan angka yang valid.")

def main():
    """Fungsi utama untuk menjalankan seluruh alur permainan."""
    bersihkan_layar()
    print("Selamat Datang di Permainan Revolver Chamber!")
    print("============================================")
    print("Setiap giliran, Anda bisa memilih: Tembak Diri Sendiri atau Tembak Pemain Lain.")
    print("Pemain terakhir yang bertahan akan menjadi pemenang.\n")

    pemain = dapatkan_info_pemain()
    jumlah_peluru = dapatkan_pengaturan_game()

    # Membuat "silinder" revolver (12 ruang)
    silinder = [1] * jumlah_peluru + [0] * (12 - jumlah_peluru)
    
    random.shuffle(silinder)
    
    print("\nDealer telah memasukkan peluru dan memutar silinder...")
    time.sleep(2)
    print("Permainan dimulai!")
    time.sleep(1)

    pemain_aktif = list(pemain)
    giliran_index = 0
    ronde = 1

    # Loop utama permainan
    while len(pemain_aktif) > 1 and len(silinder) > 0:
        bersihkan_layar()
        print(f"--- Ronde {ronde} ---")
        print(f"Pemain yang tersisa: {', '.join(pemain_aktif)}")
        print(f"Sisa ruang di silinder: {len(silinder)} dari 12\n")

        if giliran_index >= len(pemain_aktif):
            giliran_index = 0
            
        pemain_sekarang = pemain_aktif[giliran_index]
        
        print(f"Sekarang giliran {pemain_sekarang}.")
        
        # --- LOGIKA PILIHAN BARU ---
        pilihan = ''
        while pilihan not in ['1', '2']:
            pilihan = input("Pilihan Anda: [1] Tembak Diri Sendiri [2] Tembak Pemain Lain -> ")

        # Menembak diri sendiri
        if pilihan == '1':
            input(f"{pemain_sekarang} memutuskan untuk menembak diri sendiri. Tekan Enter untuk melanjutkan...")
            print("Menarik pelatuk ke arah sendiri...")
            time.sleep(2)
            hasil = silinder.pop(0)
            
            if hasil == 1:
                print(f"💥 BANG! {pemain_sekarang} terkena tembakan dan keluar dari permainan.")
                pemain_aktif.pop(giliran_index)
                # Giliran tidak berpindah, karena pemain selanjutnya menempati indeks ini
            else:
                print(f"✨ Klik! {pemain_sekarang} selamat.")
                giliran_index += 1 # Giliran berpindah ke pemain selanjutnya
        
        # Menembak pemain lain
        elif pilihan == '2':
            # Membuat daftar target (pemain lain yang masih aktif)
            daftar_target = [p for p in pemain_aktif if p != pemain_sekarang]
            
            print("\nPilih target untuk ditembak:")
            for i, target in enumerate(daftar_target):
                print(f"[{i+1}] {target}")
            
            pilihan_target_idx = -1
            while pilihan_target_idx < 0 or pilihan_target_idx >= len(daftar_target):
                try:
                    pilihan_target_idx = int(input("Masukkan nomor target: ")) - 1
                except ValueError:
                    print("Pilihan tidak valid.")

            target_terpilih = daftar_target[pilihan_target_idx]
            
            input(f"{pemain_sekarang} memutuskan untuk menembak {target_terpilih}. Tekan Enter...")
            print(f"Mengarahkan revolver ke {target_terpilih} dan menarik pelatuk...")
            time.sleep(2)
            
            hasil = silinder.pop(0)
            
            if hasil == 1:
                print(f"💥 BANG! {target_terpilih} terkena tembakan dan keluar dari permainan!")
                pemain_aktif.remove(target_terpilih)
            else:
                print(f"✨ Klik! {target_terpilih} ternyata selamat.")
            
            # Apapun hasilnya, giliran pemain sekarang selesai dan berpindah
            giliran_index += 1

        ronde += 1
        time.sleep(3)

    # Menentukan hasil akhir permainan
    bersihkan_layar()
    print("\n--- PERMAINAN BERAKHIR ---")
    if len(pemain_aktif) == 1:
        print(f"🎉 Selamat, {pemain_aktif[0]} adalah satu-satunya yang bertahan dan menjadi PEMENANG!")
    elif len(pemain_aktif) > 1:
        print("🎉 Semua ruang di silinder sudah kosong!")
        print(f"Para pemain yang bertahan: {', '.join(pemain_aktif)} dinyatakan sebagai pemenang bersama!")
    else:
        print("Tidak ada yang selamat dari permainan ini.")

if __name__ == "__main__":
    main()