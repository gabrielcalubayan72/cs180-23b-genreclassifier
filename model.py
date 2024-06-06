
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk as nltk

# Load the CSV file
data = pd.read_csv('processed_lyrics.csv')

# Extract the column names
column_names = data.columns.tolist()

# Print the column names
print("Column Names:", column_names)

# As seen in the value counts for each genre below, song entries with `tag = "pop"` *dominate* the dataset. Hence, to improve the accuracy of the model, we used stratification in splitting the training and testing datasets.
data.tag.value_counts()

# We also numerically label each entry using the mapping below for processing later.
data['label_num'] = data.tag.map({'country':0, 'misc':1, 'pop':2, 'rap':3, 'rb':4, 'rock':5})
data.drop('views', axis=1, inplace=True)
data.drop('language_cld3', axis=1, inplace=True)
data.drop('language_ft', axis=1, inplace=True)

data.head(10)

# We label the dataset at this point into its x-axis `lyrics`, and y-axis `tag`, and split it into a training and testing dataset.
lyrics = data['lyrics']
genre = data['tag']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(lyrics, genre, test_size=0.2, stratify=genre, random_state=42)

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the training data
vectorizer.fit(X_train)

# Examine the fitted vocabulary
vectorizer.get_feature_names_out()

# Fit and transform training data into a 'document-term matrix'
X_train_dtm = vectorizer.fit_transform(X_train)

# Examine the vocabulary and document-term matrix together
pd.DataFrame(X_train_dtm.toarray(), columns = vectorizer.get_feature_names_out())

# Transform testing data into a document-term matrix (using existing vocabulary)
X_test_dtm = vectorizer.transform(X_test)
X_test_dtm.toarray()

# Examine the vocabulary and document-term matrix together
pd.DataFrame(X_test_dtm.toarray(), columns=vectorizer.get_feature_names_out())

# Initialize the Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()

# Train the classifier
nb_classifier.fit(X_train_dtm, y_train)

# Make predictions on the test data
y_pred_class = nb_classifier.predict(X_test_dtm)

# Calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)
metrics.confusion_matrix(y_test, y_pred_class)

# Print the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_class))

# Store the vocabulary of X_train
X_train_tokens = vectorizer.get_feature_names_out()
len(X_train_tokens)

# Examine the first 50 tokens
print(X_train_tokens[0:50])

# Examine the last 50 tokens
print(X_train_tokens[-50:])

# Naive Bayes counts the number of times each token appears in each class
nb_classifier.feature_count_

# Rows represent classes, columns represent tokens
nb_classifier.feature_count_.shape

country_token_count = nb_classifier.feature_count_[0, :]
misc_token_count = nb_classifier.feature_count_[1, :]
pop_token_count = nb_classifier.feature_count_[2, :]
rap_token_count = nb_classifier.feature_count_[3, :]
rb_token_count = nb_classifier.feature_count_[4, :]
rock_token_count = nb_classifier.feature_count_[5, :]

# Create a DataFrame of tokens with their separate ham and spam counts
tokens = pd.DataFrame({'token':X_train_tokens, 'country':country_token_count, 'misc':misc_token_count, 'pop':pop_token_count, 'rap':rap_token_count, 'rb':rb_token_count, 'rock':rock_token_count}).set_index('token')
tokens.head()

# Examine 5 random DataFrame rows
tokens.sample(5, random_state=427)

# Add 1 to tag counts to avoid dividing by 0 (1 point)
tokens[['country', 'misc', 'pop', 'rap', 'rb', 'rock']] = tokens[['country', 'misc', 'pop', 'rap', 'rb', 'rock']].apply(lambda x: x + 1)

# Convert the tag counts into frequencies
tokens['country'] = tokens['country'] / nb_classifier.class_count_[0]
tokens['misc'] = tokens['misc'] / nb_classifier.class_count_[1]
tokens['pop'] = tokens['pop'] / nb_classifier.class_count_[2]
tokens['rap'] = tokens['rap'] / nb_classifier.class_count_[3]
tokens['rb'] = tokens['rb'] / nb_classifier.class_count_[4]
tokens['rock'] = tokens['rock'] / nb_classifier.class_count_[5]

tokens.sample(5, random_state=427)

# Calculate the ratio of each genre to all other genres for each token
for genre in ['country', 'misc', 'pop', 'rap', 'rb', 'rock']:
    other_genres = [col for col in tokens.columns if col != genre]
    tokens[f'{genre}_ratio'] = tokens[genre] / tokens[other_genres].sum(axis=1)

# Sample 5 rows from the DataFrame
sampled_tokens = tokens.sample(5, random_state=427)

# Print the sampled tokens with the calculated ratios
print(sampled_tokens)

# Examine the DataFrame sorted by spam_ratio
# note: use sort() instead of sort_values() for pandas 0.16.2 and earlier
tokens.sort_values('pop_ratio', ascending=False)

# Show default parameters for CountVectorizer
vectorizer
metrics.accuracy_score(y_test, y_pred_class)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Define the pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB())
])

# Define the parameters to search
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],  # Test unigrams and bigrams
    'vect__max_df': [0.5, 0.75, 1.0],       # Test different maximum document frequencies
    'vect__min_df': [1, 2, 5],               # Test different minimum document frequencies
    'vect__stop_words': [None, 'english'],    # Test with and without stopwords
    'clf__alpha': [0.1, 0.5, 1.0]            # Test different alpha values for Laplace smoothing
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best Parameters:", grid_search.best_params_)

# Print the best accuracy found
print("Best Accuracy:", grid_search.best_score_)

# Extract the best parameters found by grid search
best_params = grid_search.best_params_
best_params

# Initialize CountVectorizer with the best parameters
vectorizer = CountVectorizer(ngram_range=best_params['vect__ngram_range'],
                             max_df=best_params['vect__max_df'],
                             min_df=best_params['vect__min_df'],
                             stop_words=best_params['vect__stop_words'])

# Fit the vectorizer to the entire training data
X_train_vectorized = vectorizer.fit_transform(X_train)

# Initialize and train the Multinomial Naive Bayes classifier
clf = MultinomialNB(alpha=best_params["clf__alpha"])
clf.fit(X_train_vectorized, y_train)

# Predict on the test data
X_test_vectorized = vectorizer.transform(X_test)
y_pred_class = clf.predict(X_test_vectorized)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred_class)
print("Accuracy:", accuracy)

import pickle
pickle.dump(vectorizer, open('vectorizer.pkl','wb'))
pickle.dump(clf, open('model.pkl','wb'))