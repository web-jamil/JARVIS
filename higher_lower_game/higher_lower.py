import random

from higher_lower_art import logo, vs
from game_data import data
import os

def clear():
    """Clear the terminal screen"""
    # For Windows
    if os.name == 'nt':
        os.system('cls')
    # For Mac and Linux (posix)
    else:
        os.system('clear')





def get_random_account():
    """Get data from random account"""
    return random.choice(data)

def format_data(account):
    """Format account into printable format: name, description, and country"""
    name = account["name"]
    description = account["description"]
    country = account["country"]
    return f"{name}, a {description}, from {country}"

def check_answer(guess, a_followers, b_followers):
    """Checks followers against user's guess 
    and returns True if they got it right.
    Or False if they got it wrong.""" 
    if a_followers > b_followers:
        return guess == "a"
    else:
        return guess == "b"

def game():
    print(logo)  # Prints the logo from the art module.
    score = 0
    game_should_continue = True
    account_a = get_random_account()
    account_b = get_random_account()

    while game_should_continue:
        account_a = account_b
        account_b = get_random_account()

        while account_a == account_b:  # Ensure account_a and account_b are not the same.
            account_b = get_random_account()

        print(f"Compare A: {format_data(account_a)}.")
        print(vs)  # Prints the 'vs' art from the art module.
        print(f"Against B: {format_data(account_b)}.")
        
        guess = input("Who has more followers? Type 'A' or 'B': ").lower()
        a_follower_count = account_a["follower_count"]
        b_follower_count = account_b["follower_count"]
        is_correct = check_answer(guess, a_follower_count, b_follower_count)
        clear()
        # Clear the screen for the next round.
        print(logo)  # Prints the logo again after the screen is cleared.
        
        if is_correct:
            score += 1
            print(f"You're right! Current score: {score}.")
        else:
            game_should_continue = False
            print(f"Sorry, that's wrong. Final score: {score}")

game()
