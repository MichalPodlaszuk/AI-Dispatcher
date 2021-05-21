import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
syn = wordnet.synsets('teaching')
print(syn)