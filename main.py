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

init_t: datetime = datetime.datetime.now()


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
    topics = [r[1].split(",") for r in rows]
    headers = [r[0] for r in rows]

    return documents, topics, sections, headers

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



def ex1(method = "tfidf"):
    docs, topics, sections, headers = get_documents_and_topics("news2.csv")
    food_and_drink_corpus = get_all_articles_in_section("Food & Drink", docs, sections)

    corpus_bow,dictionary = get_bow(docs)

    if method == "tfidf":
        tfidf = models.TfidfModel(corpus_bow)
        tfidf_vectors = tfidf[corpus_bow]

        id2token = dict(dictionary.items())


        def convert(match):
            return dictionary.id2token[int(match.group(0)[0:-1])]



        matrix_tfidf = similarities.MatrixSimilarity(tfidf_vectors)
    
    elif method == "lda":
        lda = models.LdaModel(corpus_bow, num_topics=30, id2word=dictionary, random_state=30, passes=2)
        lda_vectors = []
        for v in corpus_bow:
            lda_vectors.append(lda[v])

        matrix_lda = similarities.MatrixSimilarity(lda_vectors)

    end_creation_model_t: datetime = datetime.datetime.now()


    porter = PorterStemmer()
    stoplist = stopwords.words('english')
    total_goods = 0

    for doc in food_and_drink_corpus:
        doc_s = [porter.stem(word) for word in doc.lower().split() if word not in stoplist]

        vec_bow = dictionary.doc2bow(doc_s)
        vec_tfidf = tfidf[vec_bow]

        # calculate similarities between doc and each doc of texts using tfidf vectors and cosine
        sims = matrix_tfidf[vec_tfidf]

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
            if "Food & Drink" in section:
                total_goods += 1

    ratio_quality = total_goods/(len(food_and_drink_corpus)*10)
    print(ratio_quality)
    end_t: datetime = datetime.datetime.now()
    elapsed_time_model_creation: datetime = end_creation_model_t - init_t
    elapsed_time_comparison: datetime = end_t - end_creation_model_t
    print()
    print(f'Execution time {method} model:', elapsed_time_model_creation, 'seconds')
    print(f'Execution time {method} comparison:', elapsed_time_comparison, 'seconds')


ex1("tfidf")
ex1("lda")