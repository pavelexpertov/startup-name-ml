# In [ ]
import random

import torch
import torch.nn as nn

import text_helpers as th

# <markdown>
# Ok, so I will borrow the model from previous notebook.
# This is because the model needs to have a short term memory (i.e. hidden state).
# Furthermore, I will make the use of lowercased letters as to make the model to predict
# better since it's a small set of characters to associate subsequent ones with.and
# maximum length of words is *five* since I want the model to generate only five
# characters that sounds like a word.

# In [ ]
# Creating a set of small words with up to 5 characters
WORDS = [word.lower() for word in th.WORDS]
LOWER_UP_TO_5_CHAR_WORDS = [word for word in WORDS if len(word) <= 5] + [word[:5] for word in WORDS if len(word) > 5]
LOWER_UP_TO_5_CHAR_WORDS = list(set(LOWER_UP_TO_5_CHAR_WORDS))
len(LOWER_UP_TO_5_CHAR_WORDS)

# In [ ]
class firstRNN(nn.Module):
    '''My first ever Recurrent Neural Network'''

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(output_size + hidden_size, output_size)
        # used to be 0.1
        self.dropout = nn.Dropout(0.8)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        '''Forward function for training'''
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        combined_output = torch.cat((output, hidden), 1)
        output = self.o2o(combined_output)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def getInitialisedHidden(self):
        return torch.ones([1, self.hidden_size])

# In [ ]
criterion = nn.NLLLoss()

def train(input_word_tensor, target_word_tensor, rnn, learning_rate=0.0005):
    '''Train a given RNN model with an input and target tensors that represent a word'''
    # unsqueeze function makes each value in 1d tensor to be placed within
    # its own tensor. (e.g. tensor of shape (4) turns into shape (4, 1))
    target_word_tensor.unsqueeze_(-1)
    hidden = rnn.getInitialisedHidden()

    rnn.zero_grad()

    loss = 0

    for input_letter_tensor, target_letter_tensor in zip(input_word_tensor, target_word_tensor):
        #output, hidden = rnn(input_letter_tensor, hidden)
        output, hidden = rnn(input_letter_tensor, hidden)
        # Add up losses for each letter prediction
        loss += criterion(output, target_letter_tensor)

    loss.backward()

    for parameter in rnn.parameters():
        parameter.data.add_(parameter.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_word_tensor.size(0)

# In [ ]
import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# In [ ]
def train_loop(rnn, word_sample, train_func, n_iters=100000, print_every=10, plot_every=500, learning_rate=0.0005):
    '''Train a neural network through loop iterations'''
    all_losses = []
    total_loss = 0 # Reset every plot_every iters

    start = time.time()

    for iter in range(1, n_iters + 1):
        input_tensor, target_tensor = th.getRandomTraningTensorSet(word_sample)
        output, loss = train_func(input_tensor, target_tensor, rnn)
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

# In [ ]
rnn = firstRNN(th.LETTERS_TOTAL, 128, th.LETTERS_TOTAL)
# train_loop(rnn, LOWER_UP_TO_5_CHAR_WORDS, train, n_iters=15000)

# In [ ]
def evaluate(input_letter, rnn):
    '''Gimme me a word'''
    with torch.no_grad():
        letters = [input_letter]

        input_letter_tensor = th.getInputWordTensor(input_letter)
        hidden = rnn.getInitialisedHidden()

        counter = 0
        while(counter < 20):
            output, hidden = rnn(input_letter_tensor[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == th.LETTERS_TOTAL - 1:
                break
            else:
                letter = th.ALL_LETTERS[topi]
                letters.append(letter)
                input_letter_tensor = th.getInputWordTensor(letter)
            counter += 1

        return ''.join(letters)

# In [ ]
# evaluate('o', rnn)

# <markdown>
## Observations
# - It looks like if I remove the dropout layer, it will just print a constant list of strings. Thus the layer is necessary sincie
# it provides a 'random' capability to the model due to its job of turning of certain numberes/bits in tensors.
# - It looks like the drop out layer affects the randomness of words as well as **word**-like generated samples:
    # - If the default values (i.e. dropout at 0.1 and learning_rate at 0.0005), the random generated text doesn't read like a word.
    # - If dropout is increased either to 0.4 or 0.8, the generated sequences are much more like words and even include capital letters despite samples not having them at all.
    # It shows that the dropout makes the model more generative by randomly turning off certain bits in the array.
        # - The learning rate seems to be kinda effective but it seems the models still give out consistently word-like sequences.
# I will try to amend training data to improve generation of word-like sequences.

# In [ ]
# Will make a dataset where it only consists of first two characters.
TWO_CHAR_WORDS = list(set([word[:2] for word in LOWER_UP_TO_5_CHAR_WORDS if len(word) > 1]))
print('length', len(TWO_CHAR_WORDS))
for choice in [word for word in random.choices(TWO_CHAR_WORDS, k=15)]:
    print(choice)

# In [ ]
rnn = firstRNN(th.LETTERS_TOTAL, 128, th.LETTERS_TOTAL)
train_loop(rnn, TWO_CHAR_WORDS, train, n_iters=250)
for l in ['f', 'p', 'a', 's', 't', 'z', 'y']:
    for i in range(7):
        print(f'Letter {l}, iter: {i}', evaluate(l, rnn))

# In [ ]
rnn = firstRNN(th.LETTERS_TOTAL, 128, th.LETTERS_TOTAL)
train_loop(rnn, TWO_CHAR_WORDS, train, n_iters=int(len(TWO_CHAR_WORDS)*1.5))
for l in ['f', 'p', 'a', 's', 't', 'z', 'y']:
    for i in range(7):
        print(f'Letter {l}, iter: {i}', evaluate(l, rnn))

# In [ ]
rnn = firstRNN(th.LETTERS_TOTAL, 128, th.LETTERS_TOTAL)
train_loop(rnn, TWO_CHAR_WORDS, train, n_iters=int(len(TWO_CHAR_WORDS)*1000), print_every=1000, plot_every=5000)
for l in ['f', 'p', 'a', 's', 't', 'z', 'y']:
    for i in range(7):
        print(f'Letter {l}, iter: {i}', evaluate(l, rnn))

# <markdown>
#Ok, results looks gibberish but it does have few sequences where it may sound like a word. Gonna try just three characters.

# In [ ]
# Will make a dataset where it only consists of first three characters.
THREE_CHAR_WORDS = list(set([word[:3] for word in LOWER_UP_TO_5_CHAR_WORDS if len(word) > 2]))
print('length', len(THREE_CHAR_WORDS))
for choice in [word for word in random.choices(THREE_CHAR_WORDS, k=15)]:
    print(choice)

# In [ ]
rnn = firstRNN(th.LETTERS_TOTAL, 128, th.LETTERS_TOTAL)
train_loop(rnn, THREE_CHAR_WORDS, train, n_iters=5500)
for l in ['f', 'p', 'a', 's', 't', 'z', 'y']:
    for i in range(7):
        print(f'Letter {l}, iter: {i}', evaluate(l, rnn))

# In [ ]
rnn = firstRNN(th.LETTERS_TOTAL, 128, th.LETTERS_TOTAL)
train_loop(rnn, THREE_CHAR_WORDS, train, n_iters=int(len(THREE_CHAR_WORDS)*1.3))
for l in ['f', 'p', 'a', 's', 't', 'z', 'y']:
    for i in range(7):
        print(f'Letter {l}, iter: {i}', evaluate(l, rnn))

# In [ ]
rnn = firstRNN(th.LETTERS_TOTAL, 128, th.LETTERS_TOTAL)
train_loop(rnn, THREE_CHAR_WORDS, train, n_iters=int(len(THREE_CHAR_WORDS)*10), print_every=1000, plot_every=5000)
for l in ['f', 'p', 'a', 's', 't', 'z', 'y']:
    for i in range(7):
        print(f'Letter {l}, iter: {i}', evaluate(l, rnn))

# <markdown>
#Not bad but it still produces gibberish.

#I did little research about words and I realised words are made up of syllables.
#Thus, it's possible that instead of just generating random character sequences,
#probably I should generate random sequences of **syllables**. Hopefully it won't
#be expensive to train (which I bet it will).

#Ok, looking at spacy, it has a plugin pip package that allows you to
#get a list of syllables, which is what I need --> https://spacy.io/universe/project/spacy_syllables#gatsby-noscript

# <markdown>
## Testing different settings of the algorithm
#Settings to test:
#- Size of the hidden layer: 128, 256, 512
#- Learning rate: 0.0005, 0.001, 0.0015
#- Learning Iterations: 100, 1000, 10000, 100000

#Then, use a mutliprocess (i.e. futures) pool in order to train models with specific settings in
#concurrency and then have a look at results for 'a' and 'f' character.

#Furthermore, there will be varied datasets that will be used for training.

# In []
# Concurrency Training functions
import concurrent.futures
import multiprocessing

import pandas as pd

import utils

def training_function(hidden_layer_size, learning_rate, learning_iterations, dropout_rate, training_set):
    '''Training function for a concurrent process.'''
    print('hidden_layer_size', hidden_layer_size)
    print('learning_rate', learning_rate)
    print('learning_iterations', learning_iterations)
    print('dropout_rate', dropout_rate)
    rnn = firstRNN(th.LETTERS_TOTAL, hidden_layer_size, th.LETTERS_TOTAL, dropout_rate)
    train_loop(rnn, training_set, train, n_iters=learning_iterations,
               print_every=int(learning_iterations/10), plot_every=int(learning_iterations/2))
    a_samples = [evaluate('a', rnn) for n in range(3)]
    f_samples = [evaluate('f', rnn) for n in range(3)]
    output_set = {
        "hidden_layer_size": hidden_layer_size,
        "learning_rate": learning_rate,
        "learning_iterations": learning_iterations,
        "dropout_rate": dropout_rate,
        "a_samples": ' '.join(a_samples),
        "f_samples": ' '.join(f_samples)
    }
    return output_set

def train_settings_permutation(training_set):
    '''Return a dataframe containing results from different permutations of settings'''
    df = pd.DataFrame()
    with concurrent.futures.ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'), max_workers=2) as executor:
        settings = [
            [128, 256, 512], # Size of a hidden layer
            [0.0005, 0.001, 0.0015], # Learning rate
            [100, 1000, 10000, 100000] # learning iterations
        ]
        futures_dict = {}
        for permutation_settings in utils.get_argument_permutations(*settings):
            future = executor.submit(training_function, *permutation_settings, training_set)
            futures_dict[future] = permutation_settings
        done, not_done = concurrent.futures.wait(futures_dict)
    return done


def train_settings_permutation_mp(training_set):
    '''Return a dataframe containing results from different permutations of settings

    This function uses multiprocessing module from the standard library.
    '''
    df = pd.DataFrame()
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(3) as pool:
        settings = [
            [128, 256, 512], # Size of a hidden layer
            [0.0005, 0.001, 0.0015], # Learning rate
            [100, 1000, 10000, 100000], # learning iterations
            [0.1, 0.5, 0.8], # dropout rate
        ]
        futures_dict = {}

        permutations = [(*permutation_settings, training_set)
                         for permutation_settings in utils.get_argument_permutations(*settings)]

        pool.map(training_function, permutations)

# In [ ]
# LOWER_UP_TO_5_CHAR_WORDS
# UPDATE: Due to multi-threaded models of Pytorch, I wasn't able to make the process run as expected.
# Thus, I shall run it in a loop with less testing values. I will test extreme values to compare
# how widly setting values affect the examples.

# In [ ]
def train_settings_permutation_via_loop(training_set):
    settings = [
        [128, 512], # Size of a hidden layer
        [0.0005, 0.0015], # Learning rate
        [10000, 100000], # learning iterations
        [0.1, 0.5, 0.8], # dropout rate
        [training_set]
    ]
    results = []
    for permutation in utils.get_argument_permutations(*settings):
        results.append(training_function(*permutation))
    return results

class firstRNN(nn.Module):
    '''My first ever Recurrent Neural Network

    This one has an additional parameter of `Dropout rate` for the constructior
    '''

    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(output_size + hidden_size, output_size)
        # used to be 0.1
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        '''Forward function for training'''
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        combined_output = torch.cat((output, hidden), 1)
        output = self.o2o(combined_output)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def getInitialisedHidden(self):
        return torch.ones([1, self.hidden_size])


# In [ ]
# LOWER_UP_TO_5_CHAR_WORDS training set
# import pickle
# results = train_settings_permutation_via_loop(LOWER_UP_TO_5_CHAR_WORDS)
# with open('mah_results.pickle', 'wb') as file_obj:
#     p = pickle.Pickler(file_obj)
#     p.dump(results)


# In [ ]
# Functions to autmoate this stuff
import pickle

def train_stuff(name, training_set):
    results = train_settings_permutation_via_loop(LOWER_UP_TO_5_CHAR_WORDS)
    with open(f'{name}_results.pickle', 'wb') as file_obj:
        p = pickle.Pickler(file_obj)
        p.dump(results)

# In [ ]
# training_sets = [
#     ['LOWER_UP_TO_5_CHAR_WORDS', LOWER_UP_TO_5_CHAR_WORDS],
#     ['TWO_CHAR_WORDS', TWO_CHAR_WORDS],
#     ['THREE_CHAR_WORDS', THREE_CHAR_WORDS]
# ]
# for name, training_set in training_sets:
#     train_stuff(name, training_set)

# In [ ]
def unpickle_results(training_set_name):
    '''Return an unpickled result'''
    with open(f'{training_set_name}_results.pickle', 'rb') as file_obj:
        p = pickle.Unpickler(file_obj)
        serialised_results = p.load()
    return serialised_results

def get_dataframe_from_pickled_resutls(training_set_name):
    results = unpickle_results(training_set_name)
    if len(results) == 0:
        raise Exception(f'Unserialised result of {training_set_name} is empty for some reason')
    expected_keys = tuple(results[0].keys())
    results_keys_list = [tuple(result.keys()) for result in results]
    if not all([expected_keys == result_keys for result_keys in results_keys_list]):
        raise Exception(f'At least one result has inconsitent number of keys in {training_set_name} dataset')
    initialiser_dict = {key : [result[key] for result in results] for key in expected_keys}
    return pd.DataFrame(initialiser_dict)

# <markdown>
# Analysing data
I will use the following criteria to get an idea of which parameters affect the text generation:
1.

# In [ ]
lutfcw_df = get_dataframe_from_pickled_resutls('LOWER_UP_TO_5_CHAR_WORDS')
lutfcw_df.columns

g = lutfcw_df.groupby(['hidden_layer_size', 'learning_rate'])
g['learning_rate']
