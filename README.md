# AdaptiveSystems Practical Assignment 1: Content-Based Recommender Systems

UPM - Adaptive Systems (2025/2026)
Buonaccorsi Emanuele, Rachinger Linus

## Exercise 1: TF-IDF for "Food & Drink"

The first task involved calculating the `ratio_quality` for the "Food & Drink" topic using TF-IDF vectors and measuring the execution times for model creation and similarity comparison.

**Results (from EX 01):**
* **Ratio Quality:** **0.5992**
* **Execution Time (Model Creation):** 29.58 seconds
* **Execution Time (Similarity Comparison):** 1 minute 3.21 seconds

**Discussion:**
A `ratio_quality` of 0.5992 indicates that for an average article in the "Food & Drink" category, approximately 6 of its 10 most similar articles (found within the *entire* corpus) actually belonged to the "Food & Drink" category. This shows a reasonably good topic cohesion using the TF-IDF (lexical) approach.

The model creation time was quick, but the similarity comparison (implementing the pseudocode) was significantly slower. This is probably because finding the top-10 most similar items for *each* article requires a large number of comparisons against the entire corpus, which is computationally expensive with sparse TF-IDF vectors.



## Exercise 2: LDA for "Food & Drink"

This task repeated the first, substituting TF-IDF with an LDA (Latent Dirichlet Allocation) model configured with 30 topics and two passes.

**Results (from EX 02):**
* **Ratio Quality:** **0.5254**
* **Execution Time (Model Creation):** 45.19 seconds
* **Execution Time (Similarity Comparison):** 0.996 seconds

**Comparison with TF-IDF:**

| Metric | TF-IDF (Food & Drink) | LDA (Food & Drink) | Winner |
| :--- | :--- | :--- | :--- |
| **Ratio Quality** | **0.5992** | 0.5254 | **TF-IDF** |
| **Model Time** | **29.58 s** | 45.19 s | **TF-IDF** |
| **Comparison Time** | 63.21 s | **0.996 s** | **LDA** |

**Explanation of Differences:**

**Quality:** TF-IDF produced a higher quality ratio (0.599) than LDA (0.525). This suggests that "Food & Drink" articles share a specific, distinct vocabulary (e.g., "recipe," "saute," "vintage," "restaurant") that TF-IDF, a lexical model, captures very well. LDA, being a *semantic* model, groups articles into broader topics. With only 30 topics for the entire news corpus, "Food & Drink" might have been merged into a general "Lifestyle" or "Health" topic, reducing its distinctiveness.
**Execution Time:** LDA took longer to train (45.19s vs 29.58s) because it's a more complex, probabilistic model. However, the similarity comparison was **dramatically faster** (under 1 second vs. 63 seconds for TF-IDF). This is because LDA produces dense, low-dimensional vectors (e.g., a vector of 30 probabilities). Calculating similarity between dense vectors is computationally far more efficient than with high-dimensional, sparse TF-IDF vectors.


## Exercise 3: TF-IDF and LDA for "Sports" Topic

This task repeated the previous experiments for the "Sports" topic.

**Results (from EX 03):**
* **TF-IDF (Sports):**
    * Ratio Quality: **0.3873**
    * Model Time: 29.34 seconds
    * Comparison Time: 56.62 seconds
* **LDA (Sports):**
    * Ratio Quality: **0.0545** (extremely low)
    * Model Time: 45.18 seconds
    * Comparison Time: 0.97 seconds

**Comparison with "Food & Drink":**

The quality for "Sports" was **significantly worse** than for "Food & Drink" across both models.
* TF-IDF quality dropped from 0.599 to 0.387.
* LDA quality went from 0.525 to 0.055.

**Why are the "Sports" results worse?**

"Sports" is an extremely broad and diverse category. For instance, an article about football will use completely different vocabulary than an article about marathon running or chess. However, every food and drink article will use words like "food", "cooking" or "taste". This lexical diversity makes it hard for TF-IDF to find strong similarities.

The extremely low LDA score (0.055) indicates that this model performs really bad for this topic. 
We observed that the corpus is heavily skewed towards topics like "Politics" (1014 articles) and "Entertainment" (504 articles), while "Sports" only has 110 articles. This imbalance likely caused LDA to allocate very few topics to "Sports," representing it poorly in the topic space.


## Exercise 4: Improving Results with Tagging


As tagging is done manually, the information contained in tags is expected to be much more reliable, concise and of lower redundancy.
A helper function `print_tags_summary` has been used to better understand the tags frequency and distribution across articles.

```
Tag frequency distribution (tags per bucket):
    1:  9445 ##############################
    2:  1556 ####
  3-5:  1171 ###
 6-10:   391 #
11-20:   217 
  21+:   160 
```

We observed that the majority of tags are low-frequency, with a significant number appearing only once. This indicates a long-tail distribution, where a few tags are very common while many others are rare. By manually looking at the tags, we also observe that there can be a lot of redundancy (e.g., "food", "foods", "foodie", "foodies").

For this reason, we tried to apply tag normalization and evaluated several different tag-based similarity measures, in order to find the best approach for article similarity based on tags.

Similarity approaches tested:
- Jaccard similarity: |A ∩ B| / |A ∪ B|
- Overlap coefficient: |A ∩ B| / min(|A|, |B|)
- TF-IDF on tags
- Geometric-mean

These are the results obtained (precision with k=10):

```
Tag-similarity evaluation for section 'Food & Drink' (precision with k=10):
measure       normalize       score
------------------------------------
jaccard       True           0.6287
geometric     True           0.6213
jaccard       False          0.6197
overlap       True           0.6164
geometric     False          0.6131
overlap       False          0.6082
tfidf         True           0.5697
tfidf         False          0.5656
```

This indicates that the Jaccard similarity measure with normalization is the most effective approach for this specific section.
When comparing with the results obtained in the previous exercises, using TFIDF and LDA vectors on the article description, we can see that tag-based similarity results in a slightly higher precision (0.6287 vs 0.5992 and 0.5189 respectively).

These are the previous results for comparison:
```
EX 01
starting tfidf (topic: Food & Drink)...
tfidf for category Food & Drink: ratio quality: 0.5991803278688524

EX 02
starting lda (topic: Food & Drink)...
lda for category Food & Drink: ratio quality: 0.5188524590163934
```

Note that these results are specific to the "Food & Drink" section; other sections may result in different outcomes. We chose to focus on this section in order to compare with the results from exercises 1 and 2.





## Output 
Here is a complete printout of the final results from a run of the code:
```
EX 01
starting tfidf (topic: Food & Drink)...
tfidf for category Food & Drink: ratio quality: 0.5991803278688524
Execution time tfidf model: 0:00:29.576168 seconds
Execution time tfidf comparison: 0:01:03.212218 seconds

EX 02
starting lda (topic: Food & Drink)...
lda for category Food & Drink: ratio quality: 0.5254098360655738
Execution time lda model: 0:00:45.190032 seconds
Execution time lda comparison: 0:00:00.995950 seconds

EX 03
starting tfidf (topic: Sports)...
tfidf for category Sports: ratio quality: 0.38727272727272727
Execution time tfidf model: 0:00:29.338944 seconds
Execution time tfidf comparison: 0:00:56.617522 seconds

starting lda (topic: Sports)...
lda for category Sports: ratio quality: 0.05454545454545454
Execution time lda model: 0:00:45.182735 seconds
Execution time lda comparison: 0:00:00.968834 seconds

Extra analysis for exercise 4: tags frequency
Unique tags: 12940
- donald trump: 523
- politics: 422
- business: 282
- u.s. news: 237
- healthy living: 203
- entertainment: 183
- voices: 148
- hillary clinton: 133
- parents: 132
- women: 117
... (+12930 more)

Tag frequency distribution (tags per bucket):
    1:  9445 ##############################
    2:  1556 ####
  3-5:  1171 ###
 6-10:   391 #
11-20:   217 
  21+:   160 
EX 04 — approach comparison (normalization vs. measure)

Tag-similarity evaluation for section 'Food & Drink' (precision with k=10):
measure       normalize       score
------------------------------------
jaccard       True           0.6287
geometric     True           0.6213
jaccard       False          0.6197
overlap       True           0.6164
geometric     False          0.6131
overlap       False          0.6082
tfidf         True           0.5697
tfidf         False          0.5656

Best approach: jaccard (normalize=True) with score 0.6287
```












## Assignment Instructions

This assignment can be completed in teams of two or three students.

Given this [data set](https://drive.google.com/file/d/1WbRoAu6db7EokWkHbyXsfUX1IZu0sBYn/view?usp=sharing) including news, the assignment consists of the following tasks:

1. Implement the following pseudocode to calculate the variable ratio_quality using the TFIDF vectors:

```
total_goods = 0

For every article (a) on topic "Food and Drink":

   Obtain the top-10 most similar articles (top-10) in Corpus to a # (NOT only in topic "Food and Drink", but in the whole corpus) also, excluding a itself (we do not want to compare a with itself)

   Count how many articles in top-10 are related to topic "Food and Drink" (goods)

   total_goods = total_goods + goods

ratio_quality = total_goods/(num_articles_food_and_drink*10)
```

And measure the execution times separately for the following two subprocesses:
- Creating the model (from the program begin to the call similarities.MatrixSimilarity(tfidf_vectors))
- Implementation of the pseudocode above.


2. Repeat the previous task with LDA vectors and compare the results. Explain the differences in the results. Use 30 topics, two passes, and a random state parameter.

3. Repeat the previous two tasks but with the topic "Sports" and compare the results. Why do you think that the quality of the results is worse than the ones obtained with the topic "Food and Drink"?

4. Explain how you can get better results comparing articles by resorting to tagging. Propose a method to calculate the similarity between two articles using their associated tags. Note that the articles in the data set are already tagged.

5. (Only for students of HCID master) Explain how you would implement a content-based RS for news using the techniques mentioned in this statement. Note that you would need to work with a user profile/model, so explain briefly which kind of information you would store in this model.

To do this assignment, you must use the python libraries Gensim and NLTK, which provide implementations for the CB algorithms addressed in the course.

You will have to submit a zip/rar file containing a document explaining the results obtained in the proposed tasks, the discussion of these results, and the source code. 

Only one member of each team must submit the assignment in Moodle.