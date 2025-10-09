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
from math import sqrt
import numpy as np


def get_all_articles_in_section(sec, documents, sections):
    ret = []
    for i, top in enumerate(sections):
        if sec in top:
            ret.append(documents[i])

    return ret

def get_documents_and_topics(csv_file):
    rows = []
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)

        # Skip header (optional)
        next(reader)

        # Loop through each row
        for row in reader:
            rows.append(row)

    documents = [r[3] for r in rows]
    sections = [r[2] for r in rows]
    tags = [r[1].split(",") for r in rows]
    headers = [r[0] for r in rows]

    return documents, tags, sections, headers

def get_bow(documents):
    porter = PorterStemmer()

    # remove common words and tokenize
    stoplist = stopwords.words('english')
    texts = [
        [porter.stem(word) for word in document.lower().split() if word not in stoplist]
        for document in documents
    ]

    #print("Tokens of each document:")
    #pprint(texts)

    # create mapping keyword-id
    dictionary = corpora.Dictionary(texts)

    # create the vector for each doc
    model_bow = [dictionary.doc2bow(text) for text in texts]

    dictionary = corpora.Dictionary(texts)


    return model_bow, dictionary

def ex123(method = "tfidf", topic="Food & Drink"):
    init_t: datetime = datetime.datetime.now()

    print(f"starting {method} (topic: {topic})...")
    docs, tags, sections, headers = get_documents_and_topics("news2.csv")
    food_and_drink_corpus = get_all_articles_in_section(topic, docs, sections)

    corpus_bow,dictionary = get_bow(docs)

    if method == "tfidf":
        model = models.TfidfModel(corpus_bow)
        vectors = model[corpus_bow]

        id2token = dict(dictionary.items())


        def convert(match):
            return dictionary.id2token[int(match.group(0)[0:-1])]



        matrix = similarities.MatrixSimilarity(vectors)
    
    elif method == "lda":
        model = models.LdaModel(corpus_bow, num_topics=30, id2word=dictionary, random_state=30, passes=2)
        vectors = []
        for v in corpus_bow:
            vectors.append(model[v])

        matrix = similarities.MatrixSimilarity(vectors)

    end_creation_model_t: datetime = datetime.datetime.now()


    porter = PorterStemmer()
    stoplist = stopwords.words('english')
    total_goods = 0

    for doc in food_and_drink_corpus:
        doc_s = [porter.stem(word) for word in doc.lower().split() if word not in stoplist]

        vec_bow = dictionary.doc2bow(doc_s)
        vec_tfidf = model[vec_bow]

        # calculate similarities between doc and each doc of texts using tfidf vectors and cosine
        sims = matrix[vec_tfidf]

        # sort similarities in descending order
        sims = sorted(enumerate(sims), key=lambda item: -item[1])

        """print()
        print("Given the doc: " + doc)
        print("whose tfidf vector is: " + re.sub("[0-9]+,", convert, str(vec_tfidf)))
        print()
        print("The Similarities between this doc and the documents of the corpus are:")
        for doc_position, doc_score in sims[1:11]:
            print(doc_score, docs[doc_position])"""
        
        best_matches = sims[1:11]

        for pos, score in best_matches:
            section = sections[pos]
            if topic in section:
                total_goods += 1

    ratio_quality = total_goods/(len(food_and_drink_corpus)*10)
    print(f"{method} for category {topic}: ratio quality: {ratio_quality}")
    end_t: datetime = datetime.datetime.now()
    elapsed_time_model_creation: datetime = end_creation_model_t - init_t
    elapsed_time_comparison: datetime = end_t - end_creation_model_t
    print(f'Execution time {method} model:', elapsed_time_model_creation, 'seconds')
    print(f'Execution time {method} comparison:', elapsed_time_comparison, 'seconds')
    print()

def ex4(verbose = False):
    # create a similarity matrix based on tags:

    def distance(tags_a, tags_b):
        a_in_b = 0
        for i, tag in enumerate(tags_a):
            if tag in tags_b:
                a_in_b += 1
        
        b_in_a = 0
        for i, tag in enumerate(tags_b):
            if tag in tags_a:
                b_in_a += 1

        # geometric mean as distance measure
        d = sqrt(a_in_b/len(tags_a)*b_in_a/len(tags_b))
        if not d == 0 and verbose:
            print(d)
            pprint(tags_a)
            pprint(tags_b)
            print()
        return d
    
    docs, tags, sections, headers = get_documents_and_topics("news2.csv")

    matrix = np.ones((len(tags), len(tags)))
    for i, tags_a in enumerate(tags):
        for j, tags_b in enumerate(tags):
            matrix[i, j] = distance(tags_a, tags_b)
    return matrix

print("\033[31mEX 01\033[0m")

ex123("tfidf", "Food & Drink")

print("\033[31mEX 02\033[0m")

ex123("lda", "Food & Drink")

print("\033[31mEX 03\033[0m")

ex123("tfidf", "Sports")
ex123("lda", "Sports")

print("\033[31mEX 04 (Head of distance Matrix\033[0m")

# pretty print for numpy (stolen from chatgpt...)
np.set_printoptions(precision=2, suppress=True, linewidth=100)

pprint(ex4()[:10, :10])
