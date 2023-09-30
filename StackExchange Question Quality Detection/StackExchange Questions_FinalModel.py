# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:37:00 2023

@author: Aditi
"""

import xml.etree.cElementTree as ET
import pandas as pd
import spacy
import re
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
import seaborn as sns
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import enchant
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import importlib
import my_module

# Use my_module here...

# If you want to reload my_module:
importlib.reload(my_module)
importlib.reload(scipy)
# Use my_module again...
# Download stop words and WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

tree = ET.parse("C:/Users/Aditi/Documents/XML data/Posts.xml")
root = tree.getroot()

data = []
for post in root.findall("./row[@PostTypeId='1']"):
    view_count = int(post.get("ViewCount", 0))
    text_length = len(post.get("Body", ""))
    title_length = len(post.get("Title", ""))
    comment_count = int(post.get("CommentCount", 0))
    favorite_count = int(post.get("FavoriteCount", 0))
    score = int(post.get("Score", 0))
    answer_count = int(post.get("AnswerCount", 0))
    body = post.get("Body", "")
    title = post.get("Title", "")
    tags = post.get("Tags", "")

    if score > 5 and answer_count > 0:
        quality = "Good"
    elif score >= 0 and score <= 5 and answer_count == 0:
        quality = "Low"
    elif score < 0:
        quality = "Very-Low"
    else:
        continue

    data.append([post.get("Id"), view_count, text_length, title_length, comment_count, favorite_count, body, title, tags, quality])

columns = ["ID", "ViewCount", "TextLength", "TitleLength", "CommentCount", "FavoriteCount", "Body", "Title", "Tags", "Quality"]
df = pd.DataFrame(data, columns=columns)
print(df.head())
print(df['Tags'])
print(df.isnull().sum())
print(df['FavoriteCount'])
# Describe the dataset to get the statistics
print(df.describe())

# Create a correlation matrix
corr_matrix = df.corr()

# Print the correlation matrix
print(corr_matrix)
sns.boxplot(data=df)

# Remove stop words
stopwords_list = stopwords.words('english')

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))
def remove_stopwords(text):
    words = text.split()
    stopwords_list = stopwords.words('english')
    filtered_words = [word for word in words if word.lower() not in stopwords_list]
    return ' '.join(filtered_words)
def remove_special_chars_nums(text):
    # Remove non-alphabetic characters and numbers
    cleaned_text = re.sub('[^a-zA-Z]+', ' ', text)
    return cleaned_text
# Create an instance of the English dictionary from PyEnchant
english_dict = enchant.Dict("en_US")
# define the pattern for nouns
noun_pattern = re.compile(r'\b[A-Za-z]+\b')

# function to extract nouns from a text string
def extract_nouns(text):
    nouns = []
    for word in noun_pattern.findall(text):
        if len(word) > 1:
            nouns.append(word)
    return nouns

# Define a function to extract noun phrases
def extract_phrases(text):
    blob = TextBlob(text)
    return blob.noun_phrases



df['Body'] = df['Body'].apply(remove_html_tags)
df['Body'] = df['Body'].apply(remove_punctuation)
df['Body'] = df['Body'].apply(remove_stopwords)
df['Body'] = df['Body'].apply(remove_special_chars_nums)
# apply the extract_nouns function to the Body feature
df['Body_nouns'] = df['Body'].apply(lambda x: extract_nouns(x))
# Apply the function to the Body feature
df['Body_phrases'] = df['Body'].apply(extract_phrases)

df['Title'] = df['Title'].apply(remove_html_tags)
df['Title'] = df['Title'].apply(remove_punctuation)
df['Title'] = df['Title'].apply(remove_stopwords)
df['Title'] = df['Title'].apply(remove_special_chars_nums)
df['Title_nouns'] = df['Title'].apply(lambda x: extract_nouns(x))
# Apply the function to the Body feature
df['Title_phrases'] = df['Title'].apply(extract_phrases)
df['Tags'] = df['Tags'].apply(remove_punctuation)
df['Tags'] = df['Tags'].apply(remove_stopwords)
df['Tags'] = df['Tags'].apply(remove_special_chars_nums)
df['Tag_nouns'] = df['Tags'].apply(lambda x: extract_nouns(x))
print(df['Tags'])
print(df['Tag_nouns'])

print(df['Body'])
print(df['Body_nouns'])
print(df['Title_nouns'])

print(df['Body_phrases'])
print(df['Title_phrases'])


print(df['Quality'])

quality_mapping = {"Very-Low": 0, "Low": 1, "Good": 2}
df["quality_label"] = df["Quality"].map(quality_mapping)

encoder = LabelEncoder()
df["quality_encoded"] = encoder.fit_transform(df["quality_label"])
print(df['quality_encoded'])


vectorizer = CountVectorizer(max_features=1000)
documents_nounb = [' '.join(words) for words in df['Body_nouns']]
documents_phraseb = [' '.join(words) for words in df['Body_phrases']]
documents_nount = [' '.join(words) for words in df['Title_nouns']]
documents_phraset = [' '.join(words) for words in df['Title_phrases']]
documents_tagnoun = [' '.join(words) for words in df['Tag_nouns']]
bow_matrix_nounb = vectorizer.fit_transform(documents_nounb)
bow_matrix_phraseb = vectorizer.fit_transform(documents_phraseb)
bow_matrix_nount = vectorizer.fit_transform(documents_nount)
bow_matrix_phraset = vectorizer.fit_transform(documents_phraset)
bow_matrix_tagnoun = vectorizer.fit_transform(documents_tagnoun)

print(df)
df.dropna(subset=['ViewCount', 'TextLength', 'TitleLength', 'FavoriteCount', 'CommentCount'], inplace=True)
numerical_features = pd.concat([df['ViewCount'], df['TextLength'], df['TitleLength'], df['FavoriteCount'], df['CommentCount']], axis=1)
print(numerical_features.shape)
print(numerical_features)

print(bow_matrix_nounb.shape, bow_matrix_nounb.dtype)
print(bow_matrix_phraseb.shape, bow_matrix_phraseb.dtype)
print(bow_matrix_nount.shape, bow_matrix_nount.dtype)
print(bow_matrix_phraset.shape, bow_matrix_phraset.dtype)
print(bow_matrix_tagnoun.shape, bow_matrix_tagnoun.dtype)


# Select the relevant features and target variable
features2 = ['ViewCount', 'TextLength', 'TitleLength', 'FavoriteCount', 'CommentCount', 'Body_nouns', 'Body_phrases', 'Title_nouns', 'Title_phrases', 'Tag_nouns']
target2 = 'quality_encoded'

# Compute the correlation matrix
corr_matrix = df[features2 + [target2]].corr()

# Get the correlation coefficients for the target variable
correlations = corr_matrix[target2]

# Sort the correlations in descending order
correlations = correlations.sort_values(ascending=False)

# Print the correlations
print(correlations)



# Concatenate the matrix features horizontally

#X = np.hstack((numerical_features, bow_matrix_nounb, bow_matrix_phraseb, bow_matrix_nount, bow_matrix_phraset, bow_matrix_tagnoun))

#X = ((bow_matrix_nounb, bow_matrix_phraseb, bow_matrix_nount, bow_matrix_phraset, bow_matrix_tagnoun, numerical_features))
#X = df[['ViewCount', 'TextLength', 'TitleLength', 'FavoriteCount', 'CommentCount']]
#X = np.hstack((bow_matrix_nounb.toarray(),
#               bow_matrix_phraseb.toarray(),
#               bow_matrix_nount.toarray(),
#               bow_matrix_phraset.toarray(),
#               bow_matrix_tagnoun.toarray()))

# Initialize a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the data and transform the data
scaled_features = scaler.fit_transform(df[['ViewCount', 'TextLength', 'FavoriteCount']])

# Replace the original columns with the scaled values
df[['ViewCount', 'TextLength', 'FavoriteCount']] = scaled_features

#scaler = MinMaxScaler()
# Fit the scaler to the data and transform the data
#scaled_features = scaler.fit_transform(df[['ViewCount', 'TextLength', 'FavoriteCount']])

# Replace the original columns with the scaled values
#df[['ViewCount', 'TextLength', 'FavoriteCount']] = scaled_features

# Add 1 to each value to avoid taking the log of zero
#log_features = np.log(df[['ViewCount', 'TextLength', 'FavoriteCount']] + 1)

# Replace the original columns with the log-transformed values
#df[['ViewCount', 'TextLength', 'FavoriteCount']] = log_features

X = np.hstack((df[['ViewCount', 'TextLength', 'FavoriteCount']].values, 
               bow_matrix_nounb.toarray(), 
               bow_matrix_phraseb.toarray(), 
               bow_matrix_tagnoun.toarray()))
#X[X < 0] = 0

# Print the shape of the concatenated matrix

# create target variable
y = df['quality_encoded'].values
print(y.shape)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
print('Logistic Regression train accuracy:', logreg.score(X_train, y_train))
print('Logistic Regression test accuracy:', logreg.score(X_test, y_test))

# Train and evaluate Multinomial Naive Bayes
#mnb = MultinomialNB()
#mnb.fit(X_train, y_train)
#print('Multinomial Naive Bayes train accuracy:', mnb.score(X_train,y_train))
#print('Multinomial Naive Bayes test accuracy:', mnb.score(X_test, y_test))

# Train and evaluate Random Forests
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print('Random Forests train accuracy:', rf.score(X_train, y_train))
print('Random Forests test accuracy:', rf.score(X_test, y_test))

