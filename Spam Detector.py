# Project to implement a spam classifier

#Importing packages
import os

# Data wrangling and data visualistion 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Processing text
import nltk
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

# Others
import numpy as np
from collections import Counter
import time
from statistics import mean

# !----First, EDA(Exploratory Data Analysis) must be done----!

data = pd.read_csv("spam.csv", encoding= "latin-1") ## to read the data file
data = data.dropna(how = "any", axis= 1)            ## to drop the null value
##(how = any means if anything is null, delete row or column; axis = 1 means drop the column with null)

data.columns = ['label', 'body_text']               ## to label the columns
data.head()                                         ## to select top 5 data of the columns


# let's use more visual representation of our data
total = len(data)                              ## get the total length of data
plt.figure(figsize = (5,5))                    ## give the size of the plot
plt.title("Number of spam vs ham messages")    ## give the title of the plot
ax = sns.countplot(x= "label", data= data)     ## plotting the data using seaborn

for p in ax.patches:                           ## iteration to show the visualization in percentage
    percentage = '{0:.0f}%'.format(p.get_height()/total * 100)
    x = p.get_x() + p.get_width() / 2
    y = p.get_height() + 20
    ax.annotate(percentage, (x,y), ha= 'center')

plt.show()                                     ## show the plot



# !----Feature Engineering----!
# body length
data["body_len"] = data.body_text.apply(lambda x: len(x) - x.count(" "))    ## appending a new column for the new method

# percentage of punctuation in the body text

def count_punct(text):                                                      ## function for counting the punctuation perc(%)
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3) * 100 

data["punct%"] = data.body_text.apply(lambda x: count_punct(x))             ## appending a new column for the function

# using new features(body_len) to explore the distribution

bins = np.linspace(0, 200, 40)
data.loc[data.label == 'spam', 'body_len'].plot(kind = "hist", bins =
bins, alpha = 0.5, density = True, label = 'spam')

data.loc[data.label == 'ham', 'body_len'].plot(kind = "hist", bins = 
bins, alpha = 0.5, density = True, label = 'ham')

plt.legend(loc = "best")
plt.xlabel("body_len")
plt.title("Body length ham vs spam")
plt.show()

# using new features(punct%) to explore the distribution 

bins = np.linspace(0, 50, 40)
data.loc[data.label == 'spam', 'punct%'].plot(kind = 'hist', bins = 
bins, alpha = 0.5, density = True, label = 'spam')

data.loc[data.label == 'ham', 'punct%'].plot(kind = 'hist', bins = 
bins, alpha = 0.5, density = True, label = 'ham')

plt.legend(loc= 'best')
plt.xlabel('punct%')
plt.title("Punctuation percentage ham vs spam")
plt.show()


# !---- This is main part of the process(Cleaning Text) ----!


# Create function for text cleaning 
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.findall('\S+', text)
    text = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words("English")]
    return text
# Apply function to body_text 
data['cleaned_text'] = data['body_text'].apply(lambda x: clean_text(x))
data[['body_text', 'cleaned_text']].head(10)





# !---- Vectorisation ----!
# TfidfVectorizer
# There are 3 methods to do vectorising but I am
# -selecting TF- IDF vectorizer.
# Checking the vectorizer with an example

corpus = ['I love bananas', 'Bananas are so amazing!', 
            'Bananas go so well with pancakes']
tfidf_vect = TfidfVectorizer()
corpus = tfidf_vect.fit_transform(corpus)
pd.DataFrame(corpus.toarray(), columns= tfidf_vect.get_feature_names())



# !---- Modelling ----!

# Train-test-split Approach
X_train, X_test, Y_train, Y_test = train_test_split(data[['body_text', 'body_len', 'punct%']], data.label, random_state = 42, test_size = 0.2)
# Instantiate and fit TfidfVectorizer
tfidf_vect = TfidfVectorizer(analyzer = clean_text)
tfidf_vect_fit = tfidf_vect.fit(X_train['body_text'])
# Use fitted TfidfVectorizer to transform body text in X_train and X_test
tfidf_train = tfidf_vect.transform(X_train['body_text'])
tfidf_test = tfidf_vect.transform(X_test['body_text'])
# Recombine transformed body text with body_len and punct% features
X_train = pd.concat([X_train[['body_len', 'punct%']].reset_index(drop = True), pd.DataFrame(tfidf_train.toarray())], axis = 1)
X_test = pd.concat([X_test[['body_len', 'punct%']].reset_index(drop = True), pd.DataFrame(tfidf_test.toarray())], axis = 1)


# Random Forest Method
# Instantiate randomforest Classifier
def explore_rf_params(n_est, depth):
    rf = RandomForestClassifier(n_estimators = n_est, max_depth = depth, n_jobs = -1, random_state = 42)
    rf_model = rf.fit(X_train, Y_train)
    Y_pred = rf_model.predict(X_test)
    precision, recall, fscore, support = score(Y_test, Y_pred, pos_label = 'spam', average = 'binary')
    print(f"Est: {n_est} / Depth: {depth} ---- Precision: {round(precision, 3)} / Recall: {round(recall, 3)} / Accuracy: {round((Y_pred==Y_test).sum() / len(Y_pred), 3)}")
    
for n_est in [50, 100, 150]:
    for depth in [10, 20, 30, None]:
        explore_rf_params(n_est, depth)


# Instantiate RandomForestClassifier with optimal set of hyperparameters 
rf = RandomForestClassifier(n_estimators = 100, max_depth = None, random_state = 42, n_jobs = -1)
# Fit model
start = time.time()
rf_model = rf.fit(X_train, Y_train)
end = time.time()
fit_time = end - start
# Predict 
start = time.time()
Y_pred = rf_model.predict(X_test)
end = time.time()
pred_time = end - start
# Time and prediction results
precision, recall, fscore, support = score(Y_test, Y_pred, pos_label = 'spam', average = 'binary')
print(f"Fit time: {round(fit_time, 3)} / Predict time: {round(pred_time, 3)}")
print(f"Precision: {round(precision, 3)} / Recall: {round(recall, 3)} / Accuracy: {round((Y_pred==Y_test).sum() / len(Y_pred), 3)}")

# Confusion matrix for RandomForestClassifier
matrix = confusion_matrix(Y_test, Y_pred)
sns.heatmap(matrix, annot = True, fmt = 'd')

# Gradient Boosting Classifier Method
# Instantiate GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state = 42)
# Fit model
start = time.time()
gb_model = gb.fit(X_train, Y_train)
end = time.time()
fit_time = end - start
# Predict 
start = time.time()
Y_pred = gb_model.predict(X_test)
end = time.time()
pred_time = end - start
# Time and prediction results
precision, recall, fscore, support = score(Y_test, Y_pred, pos_label = 'spam', average = 'binary')
print(f"Fit time: {round(fit_time, 3)} / Predict time: {round(pred_time, 3)}")
print(f"Precision: {round(precision, 3)} / Recall: {round(recall, 3)} / Accuracy: {round((Y_pred==Y_test).sum() / len(Y_pred), 3)}")



# Pipeline
# Instantiate TfidfVectorizer, RandomForestClassifier and GradientBoostingClassifier 
tfidf_vect = TfidfVectorizer(analyzer = clean_text)
rf = RandomForestClassifier(random_state = 42, n_jobs = -1)
gb = GradientBoostingClassifier(random_state = 42)
# Make columns transformer
transformer = make_column_transformer((tfidf_vect, 'body_text'), remainder = 'passthrough')
# Build two separate pipelines for RandomForestClassifier and GradientBoostingClassifier 
rf_pipeline = make_pipeline(transformer, rf)
gb_pipeline = make_pipeline(transformer, gb)
# Perform 5-fold cross validation and compute mean score 
rf_score = cross_val_score(rf_pipeline, data[['body_text', 'body_len', 'punct%']], data.label, cv = 5, scoring = 'accuracy', n_jobs = -1)
gb_score = cross_val_score(gb_pipeline, data[['body_text', 'body_len', 'punct%']], data.label, cv = 5, scoring = 'accuracy', n_jobs = -1)
print(f"Random forest score: {round(mean(rf_score), 3)}")
print(f"Gradient boosting score: {round(mean(gb_score), 3)}")
