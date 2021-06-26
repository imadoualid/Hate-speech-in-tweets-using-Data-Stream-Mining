# coding: utf-8
import pandas as pd
from nltk.corpus import stopwords
import re
import spacy
import en_core_web_sm
import string
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

nlp = en_core_web_sm.load()
set_stopWord=('btw','rt','da','dat','ya','tho')
nlp.Defaults.stop_words.add(set_stopWord)
for elem in set_stopWord:
    nlp.vocab[elem].is_stop=True



def lower(s):
    return s.lower()
def remove_urls(s):
    return re.sub(r'\b(([\w-]+://?|www[.])[^\s()<>]+(?:([\w\d]+)|([^[:punct:]\s]|/)))\b','',s)
def remove_emails(s):
    return re.sub(r'(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])','',s)
def remove_tags(s):
    return re.sub(r'@\S+','',s)
def remove_special_characters(s):
    return re.sub(r'([^\w\s\']|\s\'|\'\s)',' ',s)
def remove_numbers(s):
    return re.sub(r'\b\d+\b',' ',s)
def remove_extra_spaces(s):
    return re.sub(r'\s+',' ',s).strip()
def tokenize(s):

    return nlp(s)
def remove_stop_words(tokens):
    return [t for t in tokens if not nlp.vocab[t.text].is_stop and t.pos_ not in ["SPACE"]]
def lemmatize(tokens):
    return [t.lemma_ for t in tokens if len(t.text) > 2]
def remove_duplicate_words(tokens):
    return list(set(tokens))
def remove_empty_words(tokens):
    return [word for word in tokens if word]
def preprocessing(s):
    s = lower(s)
    s = remove_urls(s)
    s = remove_emails(s)
    s = remove_tags(s)
    s = remove_special_characters(s)
    s = remove_numbers(s)
    s = remove_extra_spaces(s)
    t = tokenize(s)
    t = remove_stop_words(t)
    t = lemmatize(t)
    t = remove_duplicate_words(t)
    t = remove_empty_words(t)
    return t

def word2vec(model,words):
    return model.infer_vector(words)


"""
df=pd.read_csv('labeled_data.csv')
df['label']=df['class'].apply(lambda x : 1 if x==0 or x==1 else 0)
df = df.drop(["Unnamed: 0","count", "hate_speech", "offensive_language","neither","class"], axis=1)


df['tweet']=df['tweet'].apply(preprocessing).apply(lambda x : str(x))

df = pd.read_csv("clean_labeled_data.csv")
tweets = [TaggedDocument(doc, [i]) for i, doc in enumerate(df['tweet'].apply(lambda x : eval(x)))]
print("here2")
vec_size = 50
"""

model = Doc2Vec.load("word2vec.model")
"""
#model = Doc2Vec(tweets,vector_size=50, window=2, min_count=1, workers=4)
#model.train(tweets,total_examples=len(tweets),epochs=20)
tweets = [model.infer_vector(tweet[0]) for tweet in tweets]
tweets = [model.docvecs[i] for i in range(len(df['tweet']))]

df = pd.concat([pd.DataFrame(tweets),df["label"]],axis=1)

df.to_csv("final_data.csv",index=False)
#model.save("word2vec.model")
"""

#print(word2vec(model,['hi','how','are','youuuuuuuuuuuuuuuuuuuuuuuuuuuu']))
