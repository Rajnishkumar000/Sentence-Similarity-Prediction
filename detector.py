import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from bs4 import BeautifulSoup
import warnings
from bs4 import MarkupResemblesLocatorWarning
warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
df=pd.read_csv('train.csv')
newdf=df.sample(30000,random_state=2)
nltk.download()
#PREPROCESSING
def preprocess(q):
    q = str(q).lower().strip()
    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')
    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]
        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    # Removing HTML tags
    q = BeautifulSoup(q, "html.parser").get_text()
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()
    return q

newdf['question1'] = newdf['question1'].apply(preprocess)
newdf['question2'] = newdf['question2'].apply(preprocess)
quesdf=newdf[['question1','question2']]
# FEATURE ENGINEERING
newdf['q1len']=newdf['question1'].str.len()
newdf['q2len']=newdf['question2'].str.len()
newdf['ques1numwords']=newdf['question1'].apply(lambda row:len(row.split(" ")))
newdf['ques2numwords']=newdf['question2'].apply(lambda row:len(row.split(" ")))

def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1 & w2)
newdf['word_common'] = newdf.apply(common_words, axis=1)

def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return (len(w1) + len(w2))
newdf['word_total'] = newdf.apply(total_words, axis=1)
newdf['wordshare']=round(newdf['word_common']/newdf['word_total'],2)
from nltk.corpus import stopwords
def fetch_token_features(row):
    q1 = row['question1']
    q2 = row['question2']
    SAFE_DIV = 0.0001
    STOP_WORDS = stopwords.words("english")
    token_features = [0.0] * 8
    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    # Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    return token_features

token_features = newdf.apply(fetch_token_features, axis=1)
newdf["cwc_min"] = list(map(lambda x: x[0], token_features))
newdf["cwc_max"] = list(map(lambda x: x[1], token_features))
newdf["csc_min"] = list(map(lambda x: x[2], token_features))
newdf["csc_max"] = list(map(lambda x: x[3], token_features))
newdf["ctc_min"] = list(map(lambda x: x[4], token_features))
newdf["ctc_max"] = list(map(lambda x: x[5], token_features))
newdf["last_word_eq"] = list(map(lambda x: x[6], token_features))
newdf["first_word_eq"] = list(map(lambda x: x[7], token_features))

#length based features
import distance
def fetch_length_features(row):
    q1 = row['question1']
    q2 = row['question2']
    length_features = [0.0] * 3
    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    # Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    # Average Token Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2
    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    return length_features
length_features = newdf.apply(fetch_length_features, axis=1)
newdf['abs_len_diff'] = list(map(lambda x: x[0], length_features))
newdf['mean_len'] = list(map(lambda x: x[1], length_features))
newdf['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))

# Fuzzy Features
def fetch_fuzzy_features(row):
    q1 = row['question1']
    q2 = row['question2']
    fuzzy_features = [0.0] * 4
    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    return fuzzy_features

fuzzy_features = newdf.apply(fetch_fuzzy_features, axis=1)
# Creating new feature columns for fuzzy features
newdf['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
newdf['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
newdf['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
newdf['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))
final_df = newdf.drop(columns=['id','qid1','qid2','question1','question2'])
print("finaldf shape is ",final_df.shape)

from sklearn.feature_extraction.text import CountVectorizer
#MERGED TEXT
questions=list(quesdf['question1'])+list(quesdf['question2'])
cv=CountVectorizer(max_features=3000)
q1rr,q2rr=np.vsplit(cv.fit_transform(questions).toarray(),2)
tempdf1=pd.DataFrame(q1rr,index=quesdf.index)
tempdf2=pd.DataFrame(q2rr,index=quesdf.index)
tempdf=pd.concat([tempdf1,tempdf2],axis=1)
print("tempdf shape is",tempdf.shape)
final_df=pd.concat([final_df,tempdf],axis=1)
print(final_df.shape)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(final_df.iloc[:,1:].values,final_df.iloc[:,0].values,test_size=0.2,random_state=1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
ac=accuracy_score(y_test,y_pred)
print("Accuracy of Random Forest is ",ac)

def query_point_creator(q1, q2):
    input_query = []
    # preprocess
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    # fetch basic features
    # character count
    input_query.append(len(q1))
    input_query.append(len(q2))
    # word count
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))
    p = {'question1': q1, 'question2': q2}
    input_query.append(common_words(p))
    input_query.append(total_words(p))
    input_query.append(round(common_words(p) / total_words(p), 2))
    # fetch token features
    token_features = fetch_token_features({'question1': q1, 'question2': q2})
    input_query.extend(token_features)
    # fetch length based features
    length_features = fetch_length_features({'question1': q1, 'question2': q2})
    input_query.extend(length_features)
    # fetch fuzzy features
    fuzzy_features = fetch_fuzzy_features({'question1': q1, 'question2': q2})
    input_query.extend(fuzzy_features)
    # bow feature for q1
    q1_bow = cv.transform([q1]).toarray()
    # bow feature for q2
    q2_bow = cv.transform([q2]).toarray()
    return np.hstack((np.array(input_query).reshape(1,22),q1_bow,q2_bow))

import pickle
pickle.dump(rf,open('model.pkl','wb'))
pickle.dump(cv,open('cv.pkl','wb'))
q1="raj"
q2="raj"
print(rf.predict(query_point_creator(q1, q2)))