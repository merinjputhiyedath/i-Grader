#!/usr/bin/env python
# coding: utf-8

# # Merin Joseph
# ## CPSC 597-project: i-Grader ( Faculty Assist System: autograde essay)
# ### CSUF - CS Grad
# ### CWID: 885869974
# 

# In[1]:


import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#nltk.download('stopwords')
#nltk.download('punkt')


# In[3]:


df = pd.read_csv(r"C:\Users\ponny\Desktop\Dataset\training_set_rel3.tsv", sep='\t', encoding='ISO-8859-1');
df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df[df['essay_set']==7]


# In[9]:


df.dropna(axis=1,inplace=True)


# In[10]:


df.shape


# In[11]:


df.head()


# In[12]:


df['essay_set'].value_counts()


# In[13]:


df.drop(columns=['rater1_domain1','rater2_domain1'],inplace=True,axis=1)


# In[14]:


df.shape


# In[15]:


df.head()


# In[16]:


min(df['domain1_score'])


# In[17]:


max(df['domain1_score'])


# In[18]:


df.skew()


# In[19]:


df.kurtosis()


# In[20]:


#Check Skewness
sns.distplot(df['domain1_score'])


# In[21]:


df[df['essay_set']==2]['domain1_score']


# In[22]:


min_range = [2,1,0,0,0,0,0,0]
max_range = [12,6,3,3,4,4,30,60]

def normalize(x,mi,ma):
    #print("Before Normalization: "+str(x))
    x = (x-mi)/(ma-mi)
    #print("After Normalization : "+str(x))
    return round(x*10)

df['final_score']=df.apply(lambda x:normalize(x['domain1_score'],min_range[x['essay_set']-1],max_range[x['essay_set']-1]),axis=1)


# In[23]:


df['final_score']


# In[24]:


sns.distplot(df['final_score'])


# In[25]:


df.skew()


# In[26]:


df.head()


# In[27]:


df.describe()


# In[28]:


df.drop('domain1_score',axis=1,inplace=True)


# In[29]:


df.head()


#  

# **PRE_PROCESSING**

# In[30]:


def clean_essay(essay):
    x=[]
    for i in essay.split():
        if i.startswith("@"):
            continue
        else:
            x.append(i)
    return ' '.join(x)

df['essay'] = df['essay'].apply(lambda x:clean_essay(x))


# In[32]:


import nltk
nltk.download('stopwords')


# In[33]:


stop_words = set(stopwords.words('english')) 
def remove_stop_words(essay):
    word_tokens = word_tokenize(essay) 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w)
    return ' '.join(filtered_sentence)

df['clean_essay'] = df['essay'].apply(lambda x:remove_stop_words(x))


# In[34]:


def remove_puncs(essay):
    essay = re.sub("[^A-Za-z ]","",essay)
    return essay

df['clean_essay'] = df['clean_essay'].apply(lambda x:remove_puncs(x))


# In[35]:


df.head()


# In[36]:


def sent2word(x):
    x=re.sub("[^A-Za-z0-9]"," ",x)
    words=nltk.word_tokenize(x)
    return words

def essay2word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw = tokenizer.tokenize(essay)
    final_words=[]
    for i in raw:
        if(len(i)>0):
            final_words.append(sent2word(i))
    return final_words
        

def noOfWords(essay):
    count=0
    for i in essay2word(essay):
        count=count+len(i)
    return count

def noOfChar(essay):
    count=0
    for i in essay2word(essay):
        for j in i:
            count=count+len(j)
    return count

def avg_word_len(essay):
    return noOfChar(essay)/noOfWords(essay)

def noOfSent(essay):
    return len(essay2word(essay))

def count_pos(essay):
    sentences = essay2word(essay)
    noun_count=0
    adj_count=0
    verb_count=0
    adverb_count=0
    for i in sentences:
        pos_sentence = nltk.pos_tag(i)
        for j in pos_sentence:
            pos_tag = j[1]
            if(pos_tag[0]=='N'):
                noun_count+=1
            elif(pos_tag[0]=='V'):
                verb_count+=1
            elif(pos_tag[0]=='J'):
                adj_count+=1
            elif(pos_tag[0]=='R'):
                adverb_count+=1
    return noun_count,verb_count,adj_count,adverb_count

data = open('big.txt').read()
words = re.findall('[a-z]+', data.lower())

def check_spell_error(essay):
    essay=essay.lower()
    new_essay = re.sub("[^A-Za-z0-9]"," ",essay)
    new_essay = re.sub("[0-9]","",new_essay)
    count=0
    all_words = new_essay.split()
    for i in all_words:
        if i not in words:
            count+=1
    return count
    
    


# In[37]:


df.head()


#  

# **Prep for ML**

# In[39]:


vectorizer = CountVectorizer(max_features = 10000, ngram_range=(1, 3), stop_words='english')
count_vectors = vectorizer.fit_transform(df['clean_essay'])
feature_names = vectorizer.get_feature_names()
data = df[['essay_set','clean_essay','final_score']].copy()
X = count_vectors.toarray()
y = data['final_score'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


#  

# **Machine Learning ALGO's without Pre-processing steps**

# Linear Regression

# In[42]:


#Save Trained Model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
pickle.dump(linear_regressor,open(r"C:\Users\ponny\Desktop\final_proj\SavedModels\LR_without_pp",'wb'))

#Use Saved Model
model = pickle.load(open(r"C:\Users\ponny\Desktop\final_proj\SavedModels\LR_without_pp",'rb'))
y_pred = model.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))


# Since linear regression is giving really bad results, we move on to models which do not map the features linearly like Support Vector Machines or Random Forests 

# SVR

# In[44]:


#Save Trained Model
clf = SVR(C=1.0, epsilon=0.2)
clf.fit(X_train, y_train)
pickle.dump(clf,open(r"C:\Users\ponny\Desktop\final_proj\SavedModels\SVR_without_pp",'wb'))

#Use Saved Model
clf = pickle.load(open(r"C:\Users\ponny\Desktop\final_proj\SavedModels\SVR_without_pp", 'rb'))
y_pred=clf.predict(X_test)
print("Mean squared error:%.2f"%mean_squared_error(y_test,y_pred))


# Random Forest

# In[45]:


#Save Trained Model
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
pickle.dump(rf, open(r"C:\Users\ponny\Desktop\final_proj\SavedModels\RF_without_PP", 'wb'))

#Use Saved Model
rf = pickle.load(open(r"C:\Users\ponny\Desktop\final_proj\SavedModels\RF_without_PP", 'rb'))
predictions = rf.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))


#  

# **Machine Learning ALGO's with Pre-processing steps**

# In[47]:


import nltk
nltk.download('averaged_perceptron_tagger')


# In[48]:


pro_data = df.copy()
pro_data['char_count'] = pro_data['essay'].apply(noOfChar)
pro_data['word_count'] = pro_data['essay'].apply(noOfWords)
pro_data['sent_count'] = pro_data['essay'].apply(noOfSent)
pro_data['avg_word_len'] = pro_data['essay'].apply(avg_word_len)
pro_data['spell_err_count'] = pro_data['essay'].apply(check_spell_error)
pro_data['noun_count'], pro_data['adj_count'], pro_data['verb_count'], pro_data['adv_count'] = zip(*pro_data['essay'].map(count_pos))
pro_data.to_csv(r"C:\Users\ponny\Desktop\final_proj\Processed_data1.csv")


# In[49]:


prep_df = pd.read_csv(r"C:\Users\ponny\Desktop\final_proj\Processed_data1.csv")
prep_df.drop('Unnamed: 0',inplace=True,axis=1)
prep_df.head()


# In[50]:


prep_df.shape


# In[54]:


vectorizer = CountVectorizer(max_features = 10000, ngram_range=(1, 3), stop_words='english')
count_vectors = vectorizer.fit_transform(prep_df['clean_essay'])
feature_names = vectorizer.get_feature_names()
X = count_vectors.toarray()
X_full = np.concatenate((prep_df.iloc[:, 5:].values, X), axis = 1)
y_full = prep_df['final_score'].values
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size = 0.3)


# Linear Regression

# In[75]:


#Save Trained Model
# linear_regressor = LinearRegression()
# linear_regressor.fit(X_train, y_train)
# pickle.dump(linear_regressor,open(r"C:\Users\ponny\Desktop\final_proj\SavedModels\LR_with_pp",'wb'))

#Use Saved Model
model = pickle.load(open(r"C:\Users\ponny\Desktop\final_proj\SavedModels\LR_with_pp",'rb'))
y_pred = model.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))


# In[76]:


from sklearn.metrics import r2_score

r_squared = r2_score(y_test, y_pred)
r_squared_percentage = r_squared * 100
print(r_squared_percentage)


# SVR

# In[77]:


#Save Trained Model
# clf = SVR(C=1.0, epsilon=0.2)
# clf.fit(X_train, y_train)
# pickle.dump(clf,open(r"C:\Users\ponny\Desktop\final_proj\SavedModels\SVR_with_pp",'wb'))

#Use Saved Model
clf = pickle.load(open(r"C:\Users\ponny\Desktop\final_proj\SavedModels\SVR_with_pp", 'rb'))
y_pred=clf.predict(X_test)
print("Mean squared error:%.2f"%mean_squared_error(y_test,y_pred))


# In[78]:


from sklearn.metrics import r2_score

r_squared = r2_score(y_test, y_pred)
r_squared_percentage = r_squared * 100
print(r_squared_percentage)


# Random Forest

# In[79]:


#Save Trained Model
# rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# rf.fit(X_train, y_train)
# pickle.dump(rf, open(r"C:\Users\ponny\Desktop\final_proj\SavedModels\RF_with_PP", 'wb'))

#Use Saved Model
rf = pickle.load(open(r"C:\Users\ponny\Desktop\final_proj\SavedModels\RF_with_PP", 'rb'))
y_pred = rf.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))


# In[80]:


from sklearn.metrics import r2_score

r_squared = r2_score(y_test, y_pred)
r_squared_percentage = r_squared * 100
print(r_squared_percentage)


# ## Thank you
