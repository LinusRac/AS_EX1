# AdaptiveSys

# Assignment 1: Content-Based Recommender Systems

## Explanation ex 3

Sports is a topic that is much more broad than food an drinks. For instance, an article about football will use completely different vocabulary than an article about marathon running or chess. However, every food and drink article will use words like "food", "cooking" or "taste". Maybe comparing "food and drink" to "volleyball" would be more representative.

## Explanation ex 4

As tagging is done manually, the information contained in tags is much more reliable, concise and of lower redundancy.

In our code, we determine which percentage of the tags for article 1 are also tags of article and 2 and vice versa. From these two numbers we calculate the geometric mean in order to get a distance measure between 1 (same tags) and 0 (completely disjunct sets of tags).

## Instructions

This assignment can be completed in teams of two or three students.

Given this data set including news, the assignment consists of the following tasks:

1. Implement the following pseudocode to calculate the variable ratio_quality using the TFIDF vectors:

total_goods = 0

For every article (a) on topic "Food and Drink":

   Obtain the top-10 most similar articles (top-10) in Corpus to a # (NOT only in topic "Food and Drink", but in the whole corpus) also, excluding a itself (we do not want to compare a with itself)

   Count how many articles in top-10 are related to topic "Food and Drink" (goods)

   total_goods = total_goods + goods

ratio_quality = total_goods/(num_articles_food_and_drink*10)

 

And measure the execution times separately for the following two subprocesses: 

Creating the model (from the program begin to the call similarities.MatrixSimilarity(tfidf_vectors))
Implementation of the pseudocode above.
 

2. Repeat the previous task with LDA vectors and compare the results. Explain the differences in the results. Use 30 topics, two passes, and a random state parameter.

3. Repeat the previous two tasks but with the topic "Sports" and compare the results. Why do you think that the quality of the results is worse than the ones obtained with the topic "Food and Drink"?

4. Explain how you can get better results comparing articles by resorting to tagging. Propose a method to calculate the similarity between two articles using their associated tags. Note that the articles in the data set are already tagged.

5. (Only for students of HCID master) Explain how you would implement a content-based RS for news using the techniques mentioned in this statement. Note that you would need to work with a user profile/model, so explain briefly which kind of information you would store in this model.

To do this assignment, you must use the python libraries Gensim and NLTK, which provide implementations for the CB algorithms addressed in the course.

You will have to submit a zip/rar file containing a document explaining the results obtained in the proposed tasks, the discussion of these results, and the source code. 

Only one member of each team must submit the assignment in Moodle.