import json
from nltk_utils import remove_stop
import spacy
from BackTranslation import BackTranslation
from PyDictionary import PyDictionary
from functools import lru_cache
import random


@lru_cache()
def load():
    with open('../data/data_clean/intents.json', 'r') as f:
        intents = json.load(f)

    trans = BackTranslation(url=[
          'translate.google.com',
          'translate.google.pl',
        ], proxies={'http': '127.0.0.1:1234', 'http://host.name': '127.0.0.1:4012'})

    dictionary = PyDictionary()

    nlp = spacy.load('en_core_web_trf')

    all_words = []
    tags = []
    xy = []
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']: # BACK TRANSLATION HERE (on patterns with BackTranslation)
            btp = trans.translate(pattern, src='en', tmp='zh-tw')
            w = remove_stop(pattern)
            w1 = remove_stop(btp.result_text)
            thesaurus_w = []
            thesaurus_w1 = []
            for word in w:
                synonyms = dictionary.synonym(word)
                if str(type(synonyms)) == "<class 'NoneType'>":
                    thesaurus_w.append(word)
                else:
                    thesaurus_w.append(random.choice(synonyms))
            for word in w1:
                synonyms = dictionary.synonym(word)
                if str(type(synonyms)) == "<class 'NoneType'>":
                    thesaurus_w1.append(word)
                else:
                    thesaurus_w1.append(random.choice(synonyms))
            all_words.extend(w)
            all_words.extend(w1)
            all_words.extend(thesaurus_w)
            all_words.extend(thesaurus_w1)
            xy.append((w, tag))
            xy.append((w1, tag))
            xy.append((thesaurus_w, tag))
            xy.append((thesaurus_w1, tag))
            print(xy)
    return all_words, tags, xy
