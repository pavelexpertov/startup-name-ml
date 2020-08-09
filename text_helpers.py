"""Helper functions for text data"""

import unicodedata
import random
import string

import torch
import torch.nn as nn

# The ' (single quote) is for the words from the unix dictionary 'words'
ALL_LETTERS = string.ascii_letters + "'"
LETTERS_TOTAL = len(ALL_LETTERS) + 1
with open('/usr/share/dict/words') as fo:
    words = fo.read().split('\n')

def unicodeToAscii(s):
    '''Return "normalised" string for ASCII format'''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS
    )

WORDS = [unicodeToAscii(word) for word in words if word]

def getInputWordTensor(word):
    '''Return a tensor representing a word in terms of multi-dimensional ndarrays of 0s and 1s'''
    # Each letter is represented as an array of 0s and 1 represents index of the letter.
    # Keep in mind that the '1' in the function creates a tensor to contain the array,
    # because pytorch accepts input as batchers rather than 'actual' values.
    tensor = torch.zeros(len(word), 1, LETTERS_TOTAL)
    for letter_tensor, letter in zip(tensor, word):
        letter_tensor[0][ALL_LETTERS.find(letter)] = 1
    return tensor

def getTargetWordTensor(word):
    '''Return a tensor of type Long that represents letters in terms of index positions including EOS'''
    return torch.LongTensor([ALL_LETTERS.find(letter) for letter in word] + [LETTERS_TOTAL - 1])

def getRandomTraningTensorSet(words):
    '''Return an input tensor and a target tensor for a randomly selected word'''
    random_word = random.choice(words)
    return getInputWordTensor(random_word), getTargetWordTensor(random_word)
