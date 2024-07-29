
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
df=pd.read_csv('train.csv')

#TAKING ONLY 30000 ROW
newdf=df.sample(30000,random_state=2)

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

    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
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
    q= BeautifulSoup(q, features="html.parser")
    q = q.get_text()

    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    return q

newdf['question1'] = newdf['question1'].apply(preprocess)
newdf['question2'] = newdf['question2'].apply(preprocess)


quesdf=newdf[['question1','question2']]
print(quesdf.head())


#REPEATED QUESTIONS
qid=pd.Series(newdf['qid1'].tolist()+newdf['qid2'].tolist())
print("no of unique questions",np.unique(qid).shape[0])
x=qid.value_counts()>1
print("number of questions getting repeated",x[x].shape[0])

# REPEATED QUSTION HISTOGRAM
# plt.hist(qid.value_counts().values,bins=160)
# plt.yscale('log')
# plt.show()




# FEATURE ENGINEERING
print("\n\nFEATURE ENGINEERING\n\n")
newdf['q1len']=newdf['question1'].str.len()
newdf['q2len']=newdf['question2'].str.len()

newdf['ques1numwords']=newdf['question1'].apply(lambda row:len(row.split(" ")))
newdf['ques2numwords']=newdf['question2'].apply(lambda row:len(row.split(" ")))
print(newdf.head())


def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1 & w2)
newdf['word_common'] = newdf.apply(common_words, axis=1)
print(newdf.head())



def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return (len(w1) + len(w2))


newdf['word_total'] = newdf.apply(total_words, axis=1)

newdf['wordshare']=round(newdf['word_common']/newdf['word_total'],2)
print(newdf.head())


# Advanced Features

#Token features
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

    # Get the non-stopwords in Questions
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
print(newdf.shape)

#
# sns.pairplot(newdf[['ctc_min', 'cwc_min', 'csc_min', 'is_duplicate']],hue='is_duplicate')
#
# sns.pairplot(newdf[['ctc_max', 'cwc_max', 'csc_max', 'is_duplicate']],hue='is_duplicate')
#
# sns.pairplot(newdf[['last_word_eq', 'first_word_eq', 'is_duplicate']],hue='is_duplicate')
#
# sns.pairplot(newdf[['mean_len', 'abs_len_diff','longest_substr_ratio', 'is_duplicate']],hue='is_duplicate')
#
#
# sns.pairplot(newdf[['fuzz_ratio', 'fuzz_partial_ratio','token_sort_ratio','token_set_ratio', 'is_duplicate']],hue='is_duplicate')
#
# plt.show()







# Using TSNE for Dimentionality reduction for 15 Features(Generated after cleaning the data) to 3 dimention
#
# from sklearn.preprocessing import MinMaxScaler
#
# X = MinMaxScaler().fit_transform(newdf[['cwc_min', 'cwc_max', 'csc_min', 'csc_max' , 'ctc_min' , 'ctc_max' , 'last_word_eq', 'first_word_eq' , 'abs_len_diff' , 'mean_len' , 'token_set_ratio' , 'token_sort_ratio' ,  'fuzz_ratio' , 'fuzz_partial_ratio' , 'longest_substr_ratio']])
# y = newdf['is_duplicate'].values
# from sklearn.manifold import TSNE
#
# tsne2d = TSNE(
#     n_components=2,
#     init='random', # pca
#     random_state=101,
#     method='barnes_hut',
#     n_iter=1000,
#     verbose=2,
#     angle=0.5
# ).fit_transform(X)
#
#
# x_df = pd.DataFrame({'x':tsne2d[:,0], 'y':tsne2d[:,1] ,'label':y})
#
# # draw the plot in appropriate place in the grid
# sns.lmplot(data=x_df, x='x', y='y', hue='label', fit_reg=False,palette="Set1",markers=['s','o'])
#
# plt.show()
#
# tsne3d = TSNE(
#     n_components=3,
#     init='random', # pca
#     random_state=101,
#     method='barnes_hut',
#     n_iter=1000,
#     verbose=2,
#     angle=0.5
# ).fit_transform(X)
#
# import plotly.graph_objs as go
# import plotly.tools as tls
# import plotly.offline as py
#
#
# trace1 = go.Scatter3d(
#     x=tsne3d[:,0],
#     y=tsne3d[:,1],
#     z=tsne3d[:,2],
#     mode='markers',
#     marker=dict(
#         sizemode='diameter',
#         color = y,
#         colorscale = 'Portland',
#         colorbar = dict(title = 'duplicate'),
#         line=dict(color='rgb(255, 255, 255)'),
#         opacity=0.75
#     )
# )
#
# data=[trace1]
# layout=dict(height=800, width=800, title='3d embedding with engineered features')
# fig=dict(data=data, layout=layout)
# go.Figure(fig).show()
# plt.show()
#
#
#
#



# print("\n\n\n\n\n\n\nDATA ANALYSING NOW\n")
# #DATA ANALYSIS
# sns.displot(newdf['q1len'])
# print('minimum characters',newdf['q1len'].min())
# print('maximum characters',newdf['q1len'].max())
# print('average num of characters',int(newdf['q1len'].mean()))
#
# sns.displot(newdf['q2len'])
# print('minimum characters',newdf['q2len'].min())
# print('maximum characters',newdf['q2len'].max())
# print('average num of characters',int(newdf['q2len'].mean()))
#
#
#
# sns.displot(newdf['ques1numwords'])
# print('minimum words',newdf['ques1numwords'].min())
# print('maximum words',newdf['ques1numwords'].max())
# print('average num of words',int(newdf['ques1numwords'].mean()))
#
# plt.show()

#COMMON WORDS

# sns.distplot(newdf[newdf['is_duplicate'] == 0]['word_common'],label='non_duplicate')
# sns.distplot(newdf[newdf['is_duplicate'] == 1]['word_common'],label='duplicate')

#DISTPLOT IS DEPRECIATED AND ITS NEW VERSION IS HISTPLOT WITH PARAMETER KDE=TRUE
# sns.histplot(newdf[newdf['is_duplicate'] == 0]['word_common'],label='non_duplicate',kde=True)
# sns.histplot(newdf[newdf['is_duplicate'] == 1]['word_common'],label='duplicate',kde=True)
# plt.legend()
# plt.show()
#
# # total words
# sns.histplot(newdf[newdf['is_duplicate'] == 0]['word_total'],label='non_duplicate',kde=True)
# sns.histplot(newdf[newdf['is_duplicate'] == 1]['word_total'],label='duplicate',kde=True)
# plt.legend()
# plt.show()
#
# # word share
# sns.histplot(newdf[newdf['is_duplicate'] == 0]['wordshare'],label='non_duplicate',kde=True)
# sns.histplot(newdf[newdf['is_duplicate'] == 1]['wordshare'],label='duplicate',kde=True)
# plt.legend()
# plt.show()


quesdf = newdf[['question1','question2']]
print(quesdf.head())


print("\n\n\n\n\n\n")
#
from sklearn.feature_extraction.text import CountVectorizer
#MERGED TEXT
questions=list(quesdf['question1'])+list(quesdf['question2'])
cv=CountVectorizer(max_features=3000)

q1rr,q2rr=np.vsplit(cv.fit_transform(questions).toarray(),2)
print(q1rr)


tempdf1=pd.DataFrame(q1rr,index=quesdf.index)
tempdf2=pd.DataFrame(q2rr,index=quesdf.index)

tempdf=pd.concat([tempdf1,tempdf2],axis=1)
print(tempdf)
print(tempdf.shape)


quesdf = newdf[['question1','question2']]
print(quesdf.head())

final_df = newdf.drop(columns=['id','qid1','qid2','question1','question2'])
print(final_df.shape)
print(final_df)



final_df=pd.concat([final_df,tempdf],axis=1)

print(final_df.shape)
print(final_df.head())






from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(final_df.iloc[:,1:].values,final_df.iloc[:,0].values,test_size=0.2,random_state=1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
ac=accuracy_score(y_test,y_pred)
print("Accuracy of Random Forest is ",ac)

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
ab=accuracy_score(y_test,y_pred)
print("Accuracy of Xgboost is ",ab)


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

    input_query.append(common_words(q1, q2))
    input_query.append(total_words(q1, q2))
    input_query.append(round(common_words(q1, q2) / total_words(q1, q2), 2))

    # fetch token features
    token_features = fetch_token_features(q1, q2)
    input_query.extend(token_features)

    # fetch length based features
    length_features = fetch_length_features(q1, q2)
    input_query.extend(length_features)

    # fetch fuzzy features
    fuzzy_features = fetch_fuzzy_features(q1, q2)
    input_query.extend(fuzzy_features)

    # bow feature for q1
    q1_bow = cv.transform([q1]).toarray()

    # bow feature for q2
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1, 22), q1_bow, q2_bow))




import pickle

pickle.dump(rf,open('model.pkl','wb'))
pickle.dump(cv,open('cv.pkl','wb'))

