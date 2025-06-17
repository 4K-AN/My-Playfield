import random

# Daftar pilihan dan aturan menang-kalah
choices = ["rock", "paper", "scissors", "lizard", "spock"]
rules = {
    ("scissors", "paper"): "memotong",
    ("paper", "rock"): "membungkus",
    ("rock", "lizard"): "menghancurkan",
    ("lizard", "spock"): "meracuni",
    ("spock", "scissors"): "menghancurkan",
    ("scissors", "lizard"): "menyayat",
    ("lizard", "paper"): "memakan",
    ("paper", "spock"): "menolak",
    ("spock", "rock"): "menguapkan",
    ("rock", "scissors"): "menghancurkan"
}

def decide_winner(player, computer):
    if player == computer:
        return "draw"
    elif (player, computer) in rules:
        return "player"
    else:
        return "computer"

def pretty_name(choice):
    return choice.capitalize()

def main():
    print("=== Rock-Paper-Scissors-Lizard-Spock ===")
    player_score = 0
    computer_score = 0
    round_num = 1

    while True:
        print(f"\n-- Round {round_num} --")
        print("Pilih salah satu:", ", ".join(choices))
        player = input("Kamu pilih: ").strip().lower()

        if player == "exit":
            print("Keluar dari permainan. Terima kasih sudah bermain!")
            break
        if player not in choices:
            print("Pilihan tidak valid! Ketik 'exit' untuk keluar.")
            continue

        computer = random.choice(choices)
        print(f"Komputer memilih: {pretty_name(computer)}")

        result = decide_winner(player, computer)
        if result == "draw":
            print("Hasil: Seri!")
        elif result == "player":
            action = rules[(player, computer)]
            print(f"Hasil: Kamu menang! {pretty_name(player)} {action} {pretty_name(computer)}.")
            player_score += 1
        else:
            action = rules[(computer, player)]
            print(f"Hasil: Kamu kalah! {pretty_name(computer)} {action} {pretty_name(player)}.")
            computer_score += 1

        print(f"Skor -> Kamu: {player_score} | Komputer: {computer_score}")
        round_num += 1

    print(f"\nSkor Akhir -> Kamu: {player_score} | Komputer: {computer_score}")
    print("Sampai jumpa lagi!")

if __name__ == "__main__":
    main()
