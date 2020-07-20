#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


import nltk
import re
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string


# In[3]:


nltk.download('wordnet')


# In[4]:


df_test = pd.read_csv("stanfordtrainingandtestdata/testdata.manual.2009.06.14.csv", encoding = "latin-1", names=["polarity", "tweet ID", "date","query","username","text"])
df_train = pd.read_csv("stanfordtrainingandtestdata/training.1600000.processed.noemoticon.csv", encoding = "latin-1", names=["polarity", "tweet ID", "date","query","username","text"])


# In[5]:


df_train.head(15)


# In[6]:


sns.countplot(df_test['polarity'])


# In[7]:


df= df_train.append(df_test, ignore_index=True)
df=df.sample(frac=0.5)


# In[8]:


df.head(15)


# In[9]:


df.info()


# In[10]:


df=df[df['polarity']!=2]
df.info()
sns.countplot(df['polarity'])


# In[11]:


df =df[['polarity','text']]
df.head()


# In[12]:


from nltk.corpus import stopwords


# In[13]:


def text_pr(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    nopunc = [c for c in text if c not in string.punctuation]
    text= "".join(nopunc)
    
    return [word.lower() for word in text.split() if word.lower() not in stopwords.words('english') and len(word)>3]


# In[14]:


df['cleaned_text']=df['text'].apply(text_pr)


# In[15]:


df.head(15)


# In[20]:


df1=df.copy()


# In[183]:


#ps=PorterStemmer()
#df['cleaned_text'] = df['cleaned_text'].apply(lambda x: [ps.stem(i) for i in x])


# In[18]:


lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    text = [word.lower() for word in text]
    return [lemmatizer.lemmatize(word, pos='a') for word in text]


# In[19]:


df['lemma_text'] = df['cleaned_text'].apply(lemmatize)


# In[20]:


df.head()


# In[21]:


df['cleaned_text']=df['cleaned_text'].apply(" ".join)


# In[22]:


df.head()


# In[23]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'],df['polarity'], test_size=0.33, random_state=42)


# In[26]:


X_train.head()


# In[27]:


from sklearn.feature_extraction.text import CountVectorizer


# In[28]:


bow_transformer = CountVectorizer()
bow_transformer.fit(X_train)


# In[29]:


print(len(bow_transformer.vocabulary_))


# In[30]:


X_train_bow = bow_transformer.transform(X_train)
X_test_bow = bow_transformer.transform(X_test)


# In[31]:


X_train_bow


# In[35]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[36]:


tfidf_tr=TfidfTransformer()
tfidf_tr.fit(X_train_bow)


# In[37]:


X_train_tfidf=tfidf_tr.transform(X_train_bow)
X_test_tfidf= tfidf_tr.transform(X_test_bow)


# In[51]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[39]:


nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)


# In[41]:


pred1=nb_model.predict(X_train_tfidf)


# In[43]:


nb_predictions = nb_model.predict(X_test_tfidf)


# In[44]:


from sklearn.metrics import classification_report


# In[45]:


print(classification_report(pred1,y_train))


# In[47]:


print(classification_report(nb_predictions,y_test))


# In[49]:


from sklearn.metrics import roc_auc_score
print('AUC: ',roc_auc_score(y_test,nb_predictions))


# In[53]:


Linear_svc =SVC(kernel='linear')
Linear_svc.fit(X_train_tfidf, y_train)
svm_prediction = Linear_svc.predict(X_test_tfidf)


# In[55]:


print('AUC: ',roc_auc_score(y_test,svm_prediction))


# In[56]:


rfc_model = RandomForestClassifier(n_estimators = 100)


# In[58]:


rfc_model.fit(X_train_tfidf,y_train)


# In[59]:


rfc_predictions = rfc_model.predict(X_test_tfidf)


# In[60]:


print('AUC: ',roc_auc_score(y_test,rfc_predictions))


# In[ ]:




