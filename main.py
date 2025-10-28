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
from collections import Counter


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
    topic_corpus = get_all_articles_in_section(topic, docs, sections)

    corpus_bow,dictionary = get_bow(docs)

    if method == "tfidf":
        model = models.TfidfModel(corpus_bow)
        vectors = model[corpus_bow]

        id2token = dict(dictionary.items())


        def convert(match):
            return dictionary.id2token[int(match.group(0)[0:-1])]



        matrix = similarities.MatrixSimilarity(vectors)
    
    elif method == "lda":
        # using 30 topics as specified in the assignment requirements
        model = models.LdaModel(corpus_bow, num_topics=30, id2word=dictionary, random_state=30, passes=2)
        vectors = []
        for v in corpus_bow:
            vectors.append(model[v])

        matrix = similarities.MatrixSimilarity(vectors)

    end_creation_model_t: datetime = datetime.datetime.now()


    porter = PorterStemmer()
    stoplist = stopwords.words('english')
    total_goods = 0

    for doc in topic_corpus:
        doc_s = [porter.stem(word) for word in doc.lower().split() if word not in stoplist]

        vec_bow = dictionary.doc2bow(doc_s)
        vec_model = model[vec_bow]

        # calculate similarities between doc and each doc of texts using tfidf vectors and cosine
        sims = matrix[vec_model]

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

    ratio_quality = total_goods/(len(topic_corpus)*10)
    print(f"{method} for category {topic}: ratio quality: {ratio_quality}")
    end_t: datetime = datetime.datetime.now()
    elapsed_time_model_creation: datetime = end_creation_model_t - init_t
    elapsed_time_comparison: datetime = end_t - end_creation_model_t
    print(f'Execution time {method} model:', elapsed_time_model_creation, 'seconds')
    print(f'Execution time {method} comparison:', elapsed_time_comparison, 'seconds')
    print()

# -----------------------------
# helper functions for exercise 4
# -----------------------------

def _normalize_tag(t: str, porter: PorterStemmer) -> str:
    t = (t or '').strip().lower()
    # keep alphanumerics, whitespace and hyphens/underscores; drop other punctuation
    t = re.sub(r"[^\w\s-]", "", t)
    return porter.stem(t) if t else t


def _prepare_tags(tags, normalize: bool):
    """Return a list of per-document tag lists, optionally normalized and deduplicated."""
    if not normalize:
        # deduplicate while keeping order
        return [[t for i, t in enumerate(taglist) if t and t.strip() and t not in taglist[:i]] for taglist in tags]
    porter = PorterStemmer()
    normed = []
    for taglist in tags:
        cleaned = []
        seen = set()
        for t in taglist:
            nt = _normalize_tag(t, porter)
            if nt and nt not in seen:
                seen.add(nt)
                cleaned.append(nt)
        normed.append(cleaned)
    return normed

# similarity measures functions

def _sim_geometric(a, b):
    # geometric mean of coverage proportions
    if not a or not b:
        return 0.0
    a_in_b = sum(1 for t in a if t in b)
    b_in_a = sum(1 for t in b if t in a)
    return sqrt((a_in_b / len(a)) * (b_in_a / len(b))) if a_in_b and b_in_a else 0.0


def _sim_jaccard(a, b):
    if not a and not b:
        return 0.0
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    inter = len(sa & sb)
    return (inter / union) if union else 0.0


def _sim_overlap(a, b):
    # overlap coefficient
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    denom = min(len(sa), len(sb))
    return (inter / denom) if denom else 0.0


def _build_pairwise_matrix(tag_lists, sim_func):
    n = len(tag_lists)
    m = np.ones((n, n))
    for i, a in enumerate(tag_lists):
        for j, b in enumerate(tag_lists):
            if j < i:
                # symmetry
                m[i, j] = m[j, i]
            else:
                m[i, j] = sim_func(a, b)
    return m


def _build_tfidf_matrix(tag_lists):
    """Compute cosine similarity matrix from TF-IDF of tags using gensim."""
    # gensim expects list of tokens per document
    dictionary = corpora.Dictionary(tag_lists)
    corpus_bow = [dictionary.doc2bow(tags) for tags in tag_lists]
    model = models.TfidfModel(corpus_bow)
    vectors = list(model[corpus_bow])
    index = similarities.MatrixSimilarity(vectors, num_features=len(dictionary))

    n = len(tag_lists)
    mat = np.zeros((n, n))
    for i, vec in enumerate(vectors):
        sims = index[vec]
        mat[i, :] = sims
    return mat


def ex4_distance_matrix(measure: str = 'geometric', normalize: bool = False, verbose: bool = False):
    """Create a similarity matrix based on tags with configurable measure and normalization.

    measure: 'geometric' | 'jaccard' | 'overlap' | 'tfidf'
    normalize: apply lowercasing + punctuation cleanup + stemming to tags
    """
    _, tags, _, _ = get_documents_and_topics("news2.csv")
    tag_lists = _prepare_tags(tags, normalize=normalize)

    if measure == 'geometric':
        matrix = _build_pairwise_matrix(tag_lists, _sim_geometric)
    elif measure == 'jaccard':
        matrix = _build_pairwise_matrix(tag_lists, _sim_jaccard)
    elif measure == 'overlap':
        matrix = _build_pairwise_matrix(tag_lists, _sim_overlap)
    elif measure == 'tfidf':
        matrix = _build_tfidf_matrix(tag_lists)
    else:
        raise ValueError(f"Unknown measure: {measure}")

    if verbose:
        print(f"Built similarity matrix with measure={measure}, normalize={normalize}")
    return matrix


def _same_section(sa: str, sb: str) -> bool:
    if not sa or not sb:
        return False
    return (sa == sb) or (sa in sb) or (sb in sa)


def evaluate_ex4_approaches(csv_file: str = "news2.csv", top_k: int = 10, section_filter: str | None = None):
    """Evaluate different tag-based similarities and normalization settings.

    If section_filter is None: evaluate across the whole corpus (default behavior).
    If section_filter is provided (e.g., "Food & Drink"), only documents from that
    section are used as queries, while the candidate set remains the whole corpus.
    We report the overall precision@K under the chosen setting.
    """
    docs, tags, sections, headers = get_documents_and_topics(csv_file)
    settings = [
        ('geometric', False),
        ('geometric', True),
        ('jaccard', False),
        ('jaccard', True),
        ('overlap', False),
        ('overlap', True),
        ('tfidf', False),
        ('tfidf', True),
    ]

    results = []
    for measure, normalize in settings:
        mat = ex4_distance_matrix(measure=measure, normalize=normalize, verbose=False)
        total_goods = 0

        if section_filter:
            # use only docs from the chosen section as queries
            query_indices = [i for i, sec in enumerate(sections) if _same_section(section_filter, sec)]
            n_queries = len(query_indices)
            for i in query_indices:
                sims = list(enumerate(mat[i, :]))
                sims = sorted(sims, key=lambda kv: -kv[1])
                # drop self
                sims = [(j, s) for j, s in sims if j != i]
                top = sims[:top_k]
                sec_i = sections[i]
                total_goods += sum(1 for j, _ in top if _same_section(sec_i, sections[j]))
            denom = (n_queries * top_k) if n_queries and top_k else 0
            ratio_quality = (total_goods / denom) if denom else 0.0
        else:
            # evaluate over the whole corpus (all docs as queries)
            n = len(docs)
            for i in range(n):
                sims = list(enumerate(mat[i, :]))
                sims = sorted(sims, key=lambda kv: -kv[1])
                # drop self
                sims = [(j, s) for j, s in sims if j != i]
                top = sims[:top_k]
                sec_i = sections[i]
                total_goods += sum(1 for j, _ in top if _same_section(sec_i, sections[j]))
            ratio_quality = total_goods / (n * top_k) if n and top_k else 0.0
        results.append({
            'measure': measure,
            'normalize': normalize,
            'score': ratio_quality,
        })

    # sort by score desc, then by measure name
    results.sort(key=lambda r: (-r['score'], r['measure'], r['normalize']))

    # Pretty print table
    if section_filter:
        print("\nTag-similarity evaluation for section '{}' (precision with k={}):".format(section_filter, top_k))
    else:
        print("\nTag-similarity evaluation (precision with k={}):".format(top_k))
    print("{:<12}  {:<11}  {:>8}".format("measure", "normalize", "score"))
    print("-" * 36)
    for r in results:
        print("{:<12}  {:<11}  {:>8.4f}".format(r['measure'], str(r['normalize']), r['score']))

    if results:
        best = results[0]
        print("\nBest approach: {} (normalize={}) with score {:.4f}".format(best['measure'], best['normalize'], best['score']))
    print()

def print_tags_summary(csv_file: str = "news2.csv", normalize: bool = True, limit: int = 25):
    """
    Count how many times each tag appears, print sorted by frequency desc, cut after `limit`.
    Also prints a simple histogram of tag frequency distribution.
    """
    _, tags, _, _ = get_documents_and_topics(csv_file)
    def norm(t: str) -> str:
        t = t.strip()
        return t.lower() if normalize else t
    counts = Counter(norm(t) for taglist in tags for t in taglist if t and t.strip())
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    print(f"Unique tags: {len(items)}")
    to_show = items if limit is None else items[:limit]
    for tag, cnt in to_show:
        print(f"- {tag}: {cnt}")
    if limit is not None and len(items) > limit:
        print(f"... (+{len(items) - limit} more)")

    # simple histogram to show how many tags are low-frequency
    if counts:
        freq_counts = Counter(counts.values())  # maps frequency -> number of tags with that frequency

        buckets = [
            ("1", lambda f: f == 1),
            ("2", lambda f: f == 2),
            ("3-5", lambda f: 3 <= f <= 5),
            ("6-10", lambda f: 6 <= f <= 10),
            ("11-20", lambda f: 11 <= f <= 20),
            ("21+", lambda f: f >= 21),
        ]
        bucket_vals = []
        for label, pred in buckets:
            total = sum(n for freq, n in freq_counts.items() if pred(freq))
            bucket_vals.append((label, total))
        max_val = max((v for _, v in bucket_vals), default=1) or 1
        scale = 30  # bar width
        print("\nTag frequency distribution (tags per bucket):")
        for label, v in bucket_vals:
            bar_len = int((v / max_val) * scale) if max_val else 0
            bar = "#" * bar_len
            print(f"{label:>5}: {v:>5} {bar}")



print("\033[31mEX 01\033[0m")
ex123("tfidf", "Food & Drink")

print("\033[31mEX 02\033[0m")
ex123("lda", "Food & Drink")

print("\033[31mEX 03\033[0m")
ex123("tfidf", "Sports")
ex123("lda", "Sports")

print("\033[31mExtra analysis for exercise 4: tags frequency\033[0m")
print_tags_summary("news2.csv", normalize=True, limit=10)

print("\033[31mEX 04 — approach comparison (normalization vs. measure)\033[0m")
evaluate_ex4_approaches("news2.csv", top_k=10, section_filter="Food & Drink")

# print("Distance matrix (geometric, raw tags):")
# np.set_printoptions(precision=2, suppress=True, linewidth=200)
# pprint(ex4_distance_matrix(measure='geometric', normalize=False)[:20, :20])
