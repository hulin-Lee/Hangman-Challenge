import numpy as np
import pandas as pd
import utils
from tqdm import tqdm

def policy(obs, excluded_letters):
    # Filter known words by length first (this reduces the search space significantly)
    mask = known_words['length'] == len(obs)
    
    # Further filter by each known character
    for i, s in enumerate(obs):
        if s != '_':
            mask &= known_words[str(i + 1)] == s  # Filter based on exact match of each character
    search_known = known_words[mask]

    # If there are no possible valid words, return the next best letter that is not excluded
    if search_known.empty:
        for letter in SORTED_LETTERS_FULL:
            if letter not in excluded_letters:
                return letter
    else:
        # Calculate the frequency of each letter in the remaining known words
        letter_counts = search_known[letters].sum().values
        
        # Filter valid letters based on exclusion list
        valid_indices = [i for i in range(26) if letters[i] not in excluded_letters]
        valid_letters = letters[valid_indices]
        valid_counts = letter_counts[valid_indices]

        # If no letter has a count greater than 0, choose a letter from SORTED_LETTERS_FULL
        if np.max(valid_counts) < 1:
            for letter in SORTED_LETTERS_FULL:
                if letter not in excluded_letters:
                    return letter
        else:
            # Sort valid letters based on their counts and return the one with the highest count
            sorted_indices = np.argsort(valid_counts)[::-1]
            for i in sorted_indices:
                if valid_letters[i] not in excluded_letters:
                    return valid_letters[i]

if __name__ == "__main__":
    # Load known words and prepare for sorting letters
    known_words = pd.read_csv('known_words.csv')
    SORTED_LETTERS_FULL = np.array(['e', 'i', 'a', 's', 'o', 'r', 't', 'n', 'l', 'c', 'u', 'm', 'p',
       'd', 'h', 'g', 'y', 'b', 'f', 'v', 'k', 'w', 'x', 'z', 'q', 'j'])  # frequency from high to low in known words
    letters = np.array([chr(i) for i in range(ord('a'), ord('z') + 1)])

    # Initialize Hangman challenge
    hangman_challenge = utils.hangman_challenge()
    
    success_count = 0
    # Iterate through all the words in the dictionary
    for i in tqdm(range(len(hangman_challenge.dictionary)), desc='Hangman Challenge'):
        hangman_challenge.reset(i)
        excluded_letters = set()  # To keep track of excluded letters

        # Start the guessing loop
        while hangman_challenge.left_chance > 0:
            guss = policy(hangman_challenge.observation, excluded_letters)  # Get the next guess
            state = hangman_challenge.test_policy(guss)  # Test if the guess was successful

            if state:
                if hangman_challenge.success:
                    success_count += 1
                break
            
            # Update the excluded letters based on successful or failed guesses
            excluded_letters.update(hangman_challenge.successful_gass)
            excluded_letters.update(hangman_challenge.failed_gass)

    # Calculate and print success rate
    print(f'Success rate: {success_count / len(hangman_challenge.dictionary):.4f}')
