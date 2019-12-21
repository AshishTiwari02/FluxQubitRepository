
# coding: utf-8

# In[1]:


#---------------------------------------------------
# Importing Required Library and Version
#--------------------------------------------------

import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn


print('Python Version : {}'.format(sys.version))
print('Numpy Version : {}'.format(numpy.__version__))
print('Pandas Version : {}'.format(pandas.__version__))
print('Matplotlib Version : {}'.format(matplotlib.__version__))
print('SeaBorn Version : {}'.format(seaborn.__version__))
print('Scipy Version : {}'.format(scipy.__version__))
print('Sklearn Version : {}'.format(sklearn.__version__))


# In[2]:


#---------------------------------------------------
# Library with Alias
#--------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


#---------------------------------------------------
# Import Data sets
#---------------------------------------------------
#data = pd.read_csv(r"C:\Users\SONY\ARIMA\Sales_Records.csv")
review_train = pd.read_csv(r"C:\Users\SONY\Amazon_Topic_Modelling\train.csv")
print(review_train)


# In[31]:


import re
def remove_name(txt,pattern):
    r = re.findall(pattern,txt)
    for i in r:
        txt = re.sub(i,'',txt)
    return txt

review_train['Updated Reviews'] = np.vectorize(remove_name)(review_train['Review Text'], "@â€™")


# In[32]:


print(review_train)


# In[38]:


review_train['Updated Reviews'] = review_train['Updated Reviews'].str.replace("[^a-zA-Z#]"," ")
review_train['Updated Reviews'] = review_train['Updated Reviews'].apply(lambda x:' '.join([w for w in x.split() if len(w)>1]))


# In[39]:


print(review_train)


# In[42]:


import re

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

review_train['Updated Reviews'] = preprocess_reviews(review_train['Updated Reviews'])


# In[44]:


print(review_train)


# In[45]:


from nltk.stem.porter import *
stemmer = PorterStemmer()

tokens = review_train['Updated Reviews'].apply(lambda x : x.split())
tokens = tokens.apply(lambda x : [stemmer.stem(iterator) for iterator in x ])
#tokens.head()
for i in range(len(tokens)):
    tokens[i] = ' '.join(tokens[i])
review_train['Updated Reviews']= tokens
review_train.head()


# In[46]:


from wordcloud import WordCloud 
#allwords = ' '.join([text for text in review_train['Review Text']]) 
allwordschanged = ' '.join([text for text in review_train['Updated Reviews']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(allwordschanged)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

