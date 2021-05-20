import os
from pdf2text import pdf2text

PATH = '../data/data_raw/pdf'

for root, dirs, files in os.walk(PATH):
    n = 0
    for file in files:
        n += 1
        pdf2text(file, n)