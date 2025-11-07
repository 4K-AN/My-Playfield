import random
import os
import time

# --- FUNGSI-FUNGSI INTI PERMAINAN BLACKJACK ---

def buat_deck():
    """Membuat dan mengacak satu set kartu Blackjack."""
    # Angka 11 merepresentasikan AS, yang nilainya bisa 1 atau 11.
    # Ada empat kartu bernilai 10 (10, Jack, Queen, King).
    kartu = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
    random.shuffle(kartu)
    return kartu

def hitung_nilai_tangan(tangan):
    """Menghitung total nilai dari kartu di tangan, dengan logika AS."""
    # Jika total nilai lebih dari 21 dan ada kartu AS (11),
    # nilai AS diubah menjadi 1.
    nilai = sum(tangan)
    jumlah_as = tangan.count(11)

    while nilai > 21 and jumlah_as > 0:
        nilai -= 10
        jumlah_as -= 1
    return nilai

def tampilkan_tangan(pemain, tangan_pemain, dealer, tangan_dealer, sembunyikan_kartu_dealer):
    """Menampilkan kartu dan skor pemain serta dealer ke layar."""
    os.system('cls' if os.name == 'nt' else 'clear') # Membersihkan layar
    print("--- PERMAINAN BLACKJACK ---")
    print(f"PEMAIN: {pemain}")
    
    # Menampilkan kartu dealer
    if sembunyikan_kartu_dealer:
        print(f"Tangan Dealer: [{tangan_dealer[0]}, ?]")
    else:
        skor_dealer = hitung_nilai_tangan(tangan_dealer)
        print(f"Tangan Dealer: {tangan_dealer} (Total: {skor_dealer})")

    # Menampilkan kartu pemain
    skor_pemain = hitung_nilai_tangan(tangan_pemain)
    print(f"Tangan Anda: {tangan_pemain} (Total: {skor_pemain})")
    print("-" * 27)


# --- FUNGSI UNTUK TARUHAN DAN PERMAINAN ---
# --- FUNGSI UNTUK TARUHAN DAN PERMAINAN ---
def dapatkan_taruhan(uang_maksimal):
    """Meminta pemain memasukkan jumlah taruhan yang valid."""
    while True:
        try:
            taruhan = int(input(f"Masukkan jumlah taruhan Anda (Uang Anda: ${uang_maksimal}): $"))
            if taruhan <= 0:
                print("Taruhan harus lebih dari 0.")
            elif taruhan > uang_maksimal:
                print(f"Anda tidak bisa bertaruh lebih dari uang yang Anda miliki (${uang_maksimal}).")
            else:
                return taruhan
        except ValueError:
            print("Masukkan angka yang valid.")

def main_game():
    """Fungsi utama untuk menjalankan seluruh alur permainan."""
    uang_pemain = 1000 # Uang virtual awal

    while uang_pemain > 0:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Selamat datang di meja Blackjack! Uang Anda saat ini: ${uang_pemain}")
        
        taruhan = dapatkan_taruhan(uang_pemain)
        
        # Persiapan ronde baru
        deck = buat_deck()
        tangan_pemain = [deck.pop(), deck.pop()]
        tangan_dealer = [deck.pop(), deck.pop()]

        # Loop untuk giliran pemain
        giliran_pemain_selesai = False
        while not giliran_pemain_selesai:
            tampilkan_tangan(uang_pemain, tangan_pemain, taruhan, tangan_dealer, sembunyikan_kartu_dealer=True)
            
            # Cek jika pemain langsung dapat Blackjack
            if hitung_nilai_tangan(tangan_pemain) == 21:
                print("BLACKJACK! Anda langsung menang.")
                giliran_pemain_selesai = True
                continue

            pilihan = input("Pilihan Anda: [T]ambah kartu atau [D]iam? ").lower()
            if pilihan == 't':
                tangan_pemain.append(deck.pop())
                if hitung_nilai_tangan(tangan_pemain) > 21:
                    tampilkan_tangan(uang_pemain, tangan_pemain, taruhan, tangan_dealer, sembunyikan_kartu_dealer=True)
                    print("BUST! Nilai Anda lebih dari 21. Anda kalah.")
                    giliran_pemain_selesai = True
            elif pilihan == 'd':
                giliran_pemain_selesai = True
            else:
                print("Pilihan tidak valid. Coba lagi.")
                time.sleep(1)

        # Giliran dealer (jika pemain tidak bust)
        skor_pemain = hitung_nilai_tangan(tangan_pemain)
        if skor_pemain <= 21:
            tampilkan_tangan(uang_pemain, tangan_pemain, taruhan, tangan_dealer, sembunyikan_kartu_dealer=False)
            print("\nGiliran Dealer...")
            time.sleep(1)
            
            while hitung_nilai_tangan(tangan_dealer) < 17:
                tangan_dealer.append(deck.pop())
                tampilkan_tangan(uang_pemain, tangan_pemain, taruhan, tangan_dealer, sembunyikan_kartu_dealer=False)
                print("Dealer menambah kartu...")
                time.sleep(1)

        # Menentukan pemenang dan hasil taruhan
        skor_dealer = hitung_nilai_tangan(tangan_dealer)
        print("\n--- HASIL AKHIR ---")
        tampilkan_tangan(uang_pemain, tangan_pemain, taruhan, tangan_dealer, sembunyikan_kartu_dealer=False)

        if skor_pemain > 21: # Pemain bust
            print(f"Anda kalah. Uang Anda berkurang ${taruhan}.")
            uang_pemain -= taruhan
        elif skor_dealer > 21 or skor_pemain > skor_dealer:
            # Jika pemain Blackjack, bayarannya 3:2 (1.5x taruhan)
            if skor_pemain == 21 and len(tangan_pemain) == 2:
                bayaran = int(taruhan * 1.5)
                print(f"BLACKJACK! Anda menang besar! Uang Anda bertambah ${bayaran}.")
                uang_pemain += bayaran
            else:
                print(f"Selamat, Anda menang! Uang Anda bertambah ${taruhan}.")
                uang_pemain += taruhan
        elif skor_dealer > skor_pemain:
            print(f"Dealer menang. Uang Anda berkurang ${taruhan}.")
            uang_pemain -= taruhan
        else: # Seri (Push)
            print("Hasilnya seri (Push). Uang Anda kembali.")

        if uang_pemain <= 0:
            print("\nUang Anda sudah habis. Terima kasih telah bermain!")
            break

        main_lagi = input("\nApakah Anda ingin bermain lagi? [Y/N]: ").lower()
        if main_lagi != 'y':
            print("Terima kasih telah bermain!")
            break

# Menjalankan permainan
if __name__ == "__main__":
    main_game()