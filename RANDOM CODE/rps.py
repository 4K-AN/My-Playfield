import random

def get_user_choice():
    """
    Prompt the user to enter their choice and validate it.
    Returns:
        str: The choice made by the user ('batu', 'gunting', 'kertas').
    """
    choices = ['batu', 'gunting', 'kertas']
    while True:
        user_input = input("Masukkan pilihan Anda (batu, gunting, kertas) atau 'exit' untuk keluar: ").lower()
        if user_input == 'exit':
            return 'exit'
        if user_input in choices:
            return user_input
        print("Pilihan tidak valid. Silakan coba lagi.")

def get_computer_choice():
    """
    Randomly select a choice for the computer.
    Returns:
        str: The computer's choice ('batu', 'gunting', 'kertas').
    """
    return random.choice(['batu', 'gunting', 'kertas'])

def determine_winner(user, computer):
    """
    Determine the winner between the user and the computer.
    Args:
        user (str): The user's choice.
        computer (str): The computer's choice.
    Returns:
        str: 'User', 'Computer', or 'Draw'.
    """
    if user == computer:
        return 'Draw'
    
    wins = {
        'batu': 'gunting',   # Batu menghancurkan gunting
        'gunting': 'kertas', # Gunting memotong kertas
        'kertas': 'batu'     # Kertas membungkus batu
    }
    
    if wins[user] == computer:
        return 'User'
    else:
        return 'Computer'

def play_game():
    print("=== Selamat datang di permainan Batu-Gunting-Kertas ===\n")
    while True:
        user_choice = get_user_choice()
        if user_choice == 'exit':
            print("Terima kasih telah bermain! Sampai jumpa.")
            break
        
        computer_choice = get_computer_choice()
        print(f"Komputer memilih: {computer_choice}")
        
        result = determine_winner(user_choice, computer_choice)
        if result == 'Draw':
            print("Hasil: Seri!\n")
        elif result == 'User':
            print("Hasil: Anda menang!\n")
        else:
            print("Hasil: Komputer menang!\n")

if __name__ == "__main__":
    play_game()
