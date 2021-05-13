"""
CountVectorizer creates a sparse matrix of words appearing in documents. i.e. each new word is assigned an index
and number of occurrences per sentence and position of occurance is stored in the matrix.
Tf-idf transformer calculates the Term Frequency- Inverse Document Frequency of terms (more common terms are
scored lower as they tend to be less useful in NLP).
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

count = CountVectorizer()
# ngram_range parameter sets the token length (1, 1) means one word, (2, 2) means two
docs = np.array(['The Sun is shining', 'The weather is sweet', 'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print("Raw Frequency:\n",bag.toarray())

tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
print("\nTerm Frequency-Inverse Document Frequency (TF-IDF):\n", tfidf.fit_transform(bag).toarray())
