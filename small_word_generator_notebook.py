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
Ok, results looks gibberish but it does have few sequences where it may sound like a word. Gonna try just three characters.

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
Not bad but it still produces gibberish.

I did little research about words and I realised words are made up of syllables.
Thus, it's possible that instead of just generating random character sequences,
probably I should generate random sequences of **syllables**. Hopefully it won't
be expensive to train (which I bet it will).

Ok, looking at spacy, it has a plugin pip package that allows you to
get a list of syllables, which is what I need --> https://spacy.io/universe/project/spacy_syllables#gatsby-noscript

# <markdown>
## Testing different settings of the algorithm
Settings to test:
- Size of the hidden layer: 128, 256, 512
- Learning rate: 0.0005, 0.001, 0.0015
- Iterations: 100, 1000, 10000, 100000
