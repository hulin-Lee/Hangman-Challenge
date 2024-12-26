import numpy as np
import pandas as pd
from torch import nn
import string

class hangman_challenge():
    def __init__(self, words=None):
        if words is None or len(words) == 0:
            try:
                validation_words = pd.read_csv('validation_words.csv')
                self.dictionary = validation_words['word'].values
            except FileNotFoundError:
                 print("Error: 'validation_words.csv' not found. Please provide a valid word list.")
        else:
            self.dictionary = words
        self.target = np.random.choice(self.dictionary)
        self.observation = ''.join(['_'] * len(self.target))
        self.left_chance = 6
        self.successful_gass = set()  
        self.failed_gass = set() 
        self.success = False
        self.good_gass = False
        self.finished_fraction = 0

    def reset(self, word_id=None):
        if word_id is None:
            self.target = np.random.choice(self.dictionary)
        else:
            self.target = self.dictionary[word_id]
        self.observation = ''.join(['_'] * len(self.target))
        self.left_chance = 6
        self.successful_gass = set()
        self.failed_gass = set()
        self.success = False
        self.good_gass = False
        self.finished_fraction = 0

    def play(self, gass):
        if len(gass) != 1 or not gass.isalpha():
            print("Invalid input! Please guess a single letter.")
            print(f"Current word: {self.observation}", f" Finished: {self.finished_fraction * 100:.1f}%")
            print(f"Chances left: {self.left_chance}")
            print(f"Successful gass: {self.successful_gass}")
            print(f"Failed gass: {self.failed_gass}")
            return False
        gass = gass.lower() 

        if gass in self.successful_gass or gass in self.failed_gass:
            print(f"You have already guessed the letter: {gass}. This guess does not count, try a different letter!")
            return False

        if gass in self.target:
            self.successful_gass.add(gass)
            self.observation = ''.join([c if c in self.successful_gass else '_' for c in self.target])
            finished = np.sum([1 if c in self.successful_gass else 0 for c in self.target])
            self.finished_fraction = finished / len(self.target)
        else:
            self.failed_gass.add(gass)
            self.left_chance -= 1

        print(f"Current word: {self.observation}", f" Finished: {self.finished_fraction * 100:.1f}%" )
        print(f"Chances left: {self.left_chance}")
        print(f"Successful gass: {self.successful_gass}")
        print(f"Failed gass: {self.failed_gass}")
        
        if self.observation == self.target:
            print("Congratulations! You've guessed the word!")
            return True  
        elif self.left_chance <= 0:
            print(f"Game over! The word was: {self.target}")
            return True 
        return False  

    def test_policy(self, gass):
        self.good_gass = False
        if gass in self.target:
            self.good_gass = True
            self.successful_gass.add(gass)
            self.observation = ''.join([c if c in self.successful_gass else '_' for c in self.target])
            finished = np.sum([1 if c in self.successful_gass else 0 for c in self.target])
            self.finished_fraction = finished / len(self.target)
        else:
            self.failed_gass.add(gass)
            self.left_chance -= 1
        if self.observation == self.target:
            self.success = True
            return True  
        elif self.left_chance <= 0:
            return True
        # print(f"Current word: {self.observation}", f" Finished: {self.finished_fraction * 100:.1f}%" )
        # print(f"Chances left: {self.left_chance}")
        # print(f"Successful gass: {self.successful_gass}")
        # print(f"Failed gass: {self.failed_gass}")
        
        return False

class know_words_dataset(nn.Module):
    def __init__(self, words=None, padding_index=0, split='train', train_ratio=0.9):
        super().__init__()
        if words is None or len(words) == 0:
            try:
                know_words = pd.read_csv('known_words.csv')
                self.dictionary = know_words['word'].values
            except FileNotFoundError:
                 print("Error: 'known_words.csv' not found. Please provide a valid word list.")
        else:
            self.dictionary = words
        self.padding_idx = padding_index
        letters = string.ascii_lowercase
        self.v2i = {char: idx + 1 for idx, char in enumerate(letters)} 
        self.v2i['<PAD>'] = self.padding_idx  
        self.i2v = {idx: char for char, idx in self.v2i.items()}
        self.n_vocab = len(self.v2i)
        
        self.length = np.array([len(word) for word in self.dictionary])
        self.max_len = self.length.max()
        self.x = []
        for word in self.dictionary:
            seq = []
            for l in word:
                seq.append(self.v2i[l])
            padding = [self.padding_idx] * (self.max_len - len(seq))
            seq.extend(padding)
            self.x.append(seq)
        self.x = np.array(self.x)
        total_len = len(self.x)
        train_end = int(total_len * train_ratio)
        if split == 'train':
            self.x = self.x[: train_end]
            self.length = self.length[: train_end]
        elif split == 'test':
            self.x = self.x[train_end:]
          
            self.length = self.length[train_end:]
        else:
            raise ValueError("split should be 'train' or 'test'")
   
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.length[index]
        
    
        