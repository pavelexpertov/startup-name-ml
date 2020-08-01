# StartUp Name Generator

## Intro
The idea is that a model generates a very 'techy' name for a future startup.

But what data do I have at hand?
I have the following:
- [Pitchards](https://starthouse.xyz/) -- a list of pitchcards from famous start ups.
- [A kaggle dataset](https://www.kaggle.com/govlab/open-data-500-companies) -- a dataset that represents companies using U.S. government's open data.
- [Y combinator companies](https://data.world/adamhelsinger/y-combinator-companies) 
- [Inc 5000 companies](https://data.world/albert/inc-5000-2016-the-full-list)

Found an extra [dataset](https://www.kaggle.com/theworldbank/doing-business?) that might help with external factors but not sure.

My expectations on how I will be going about to create such a model are:
1. Pull an existing built model that 'understands' sentences or (even better) words.
2. Tune the model to my list of picked up start up names.
3. Hope that it will just generate the rest of a name when you provide a part of it.
    - Also it gotta sound techy as well (e.g. tumblr, pongware, etc.)
    
## Further thoughts on implementing Generative RNN from pytorch tutorial.
Ok, the model I made up was messed up. It works but it doesn't work as expected since it was spitting gibberish.

Even though if I overtrained it on few words, it managed to print them out in the middle of the gibberish but it didn't
see the outputted word from its previous predictions.

So, I have been thinking, observing similar tutorials and their implementations: instead of
trying to generate a general language model (LM from now on) for varied sequences of words and then
tuning them to startup names (like I observed from fastai tutorial for text classification),
I would have to make several models with specific purpose.

Before explaining what models are, I need to bring myself back to the question: How to *generate* a **startup** name
for a given user *input*?

There are two places that needs to be looked at: user-point and model-point. Keep in mind that user-point is a web application
and model-point is a backend service with pre-trained models.

**User Point**

So the user wants to generate a name but there are three most likely possibilities that the user can do and what can be done:
1. User can input nothing
    - Add a random generator for a beginning letter. Unless user puts some text, each iteration it will just generate a letter and send a new request.
2. User can input at least one letter
    - A specific model for generating few letters that make word-like sequence is applied.
3. User can input a word or two (joined together since most startup names don't have separate words)
    - A specific model for appending specific "endings" for the input. Or
    - A specific model for removing a particular letter from the input. It is somewhat related to the above model but explain later. Or
    - A specific model for appending a word that will rhyme with the user input (that's still for decision whether to make it or not but it's still on the table).
    
**Model Point**

As for models described above, the reason there are multiple is because a trained model is only 'taught' to do one thing only.
In other words, relying on a model that's built to predict completed words is pointless if I tuned it to make predict startup names later since
it looks like it may not be able to generate a random name based on general and startup 'observations' due to nature of the LM to be probabilistic
rather than 'intelligent'. Even if it does have an architecture for a short-term memory, it would not probably help to generate one anyway.

So I can see developing few small but specific models that will correct an expectation of 'One model rules them all' (even though it may be possibe, don't know).

So, the models:
1. small-word-generator -- a model that generates a sequence of letters based on letter.
    - It is inspired by the tutorial I did previously
    - The dataset will be simple words with up to 3, 4, or even 5 characters long.
    - Expectation of model it will generate word-like sequences from a single letter (or a short sequence with a helping hidden state).
2. techy-ending-generator -- a model to generate a techy-sounding ending for a given input.
    - The model can be just like a generative model where you provide a single character and using a hidden state to generate a sequence of predictions. Or
    - it can be a model that does use a character for prediction, but instead of 'generating', it predicts what the ending will be by using a vector space of observed endings.
    - The reason is that there seems to be a pattern for startup names to use particular endings to make them sound like a start up.
3. techy-rhymer-generator -- a model to generate/predict a rhymable name to a user input (still up for consideration).
    - It basically predicts a word suitable for a given input from a user. To make it sound catchy, I guess.
    - It will use a vector space of observed words and use last 3, 4, 5 characters as input parameters.

The models are described based on anecdotal observations of startup names from sources but will do a statistical analysis to see
whether they can be used as model parameters.