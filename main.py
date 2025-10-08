import csv
from pprint import pprint

import datetime

from gensim import corpora
from gensim import models
from pprint import pprint  # pretty-printer
from gensim import similarities

import re

from nltk.corpus import stopwords
from nltk import PorterStemmer

def get_documents_and_topics(csv):
    rows = []
    with open('news2.csv', mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)

        # Skip header (optional)
        next(reader)

        # Loop through each row
        for row in reader:
            rows.append(row)

    documents = [r[3] for r in rows]
    topics = [r[1].split(",") for r in rows]
    return documents, topics

def get_bow():
    porter = PorterStemmer()

    # remove common words and tokenize
    stoplist = stopwords.words('english')
    texts = [
        [porter.stem(word) for word in document.lower().split() if word not in stoplist]
        for document in documents
    ]

    print("Tokens of each document:")
    pprint(texts)

    # create mapping keyword-id
    dictionary = corpora.Dictionary(texts)

    # create the vector for each doc
    model_bow = [dictionary.doc2bow(text) for text in texts]
