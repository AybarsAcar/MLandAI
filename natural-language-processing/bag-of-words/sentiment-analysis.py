import numpy as np
import pandas as pd
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# tab separated file
# ignore the quoting double quotes (")
dataset = pd.read_csv('../../data/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# clean the text
english_stopwords = stopwords.words('english')
english_stopwords.remove('not')

cleaned_data = []
for i in range(0, len(dataset)):
  # replace anything other than letter with ' '
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()

  # apply stemming
  ps = PorterStemmer()
  review_char = []
  for word in review:
    if not word in set(english_stopwords):
      review_char.append(ps.stem(word))

  review = ' '.join(review_char)
  cleaned_data.append(review)

# Create the Bag of Words model
# Apply tokenisation - so unrelated words will be eliminated which are the less frequent ones
# get the most 1500 frequent words
cv = CountVectorizer(max_features=1500)

# create the sparse matrix
X = cv.fit_transform(cleaned_data).toarray()
y = dataset.iloc[:, -1].values

print("Number of words:", len(X[0]))

# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the Naive Bayes model
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Accuracy Score:", accuracy_score(y_test, y_pred))


def review_predicter(review, stopwords, cv, model):
  review = re.sub('[^a-zA-Z]', ' ', review)
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review if not word in stopwords]
  review = " ".join(review)
  corpus = [review]
  X_test = cv.transform(corpus).toarray()
  y_pred = model.predict(X_test)
  return y_pred


positive_prediction = review_predicter("I love this restaurant so much", english_stopwords, cv, classifier)
print(positive_prediction)
negative_prediction = review_predicter("I hate this restaurant so much", english_stopwords, cv, classifier)
print(negative_prediction)

print(review_predicter("this place is disgusting", english_stopwords, cv, classifier))
