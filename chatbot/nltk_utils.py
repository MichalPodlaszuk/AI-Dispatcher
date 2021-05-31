import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
from functools import lru_cache
import spacy


stemmer = PorterStemmer()


@lru_cache()
def loader():
    nltk.download('punkt')


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word)


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


@lru_cache()
def remove_stop(sentence):
    nlp = spacy.load('en_core_web_trf')
    stopwords = [',', '.', '?', '!']
    stopwords.extend(spacy.lang.en.stop_words.STOP_WORDS)
    sent_without_stop = [w for w in tokenize(sentence) if w not in stopwords]
    return sent_without_stop
