
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


# In[10]:


#---------------------------------------------------
# Library with Alias
#--------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk import FreqDist
nltk.download('stopwords')


# In[11]:


#---------------------------------------------------
# Import Data sets
#---------------------------------------------------
#data = pd.read_csv(r"C:\Users\SONY\ARIMA\Sales_Records.csv")
review_train = pd.read_csv(r"C:\Users\SONY\Amazon_Topic_Modelling\train.csv")
print(review_train)
review_test = pd.read_csv(r"C:\Users\SONY\Amazon_Topic_Modelling\test.csv")


# In[12]:


def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()
  
  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
  
  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()
    
freq_words(review_train['Review Text'])


# In[13]:


# replace "n't" with " not"
review_train['Review Text'] = review_train['Review Text'].str.replace("n\'t", " not")

# remove unwanted characters, numbers and symbols
review_train['Review Text'] = review_train['Review Text'].str.replace("[^a-zA-Z#]", " ")


# In[14]:


freq_words(review_train['Review Text'])


# In[15]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[16]:


def remove_stopwords(rev):
  rev_new = " ".join([i for i in rev if i not in stop_words])
  return rev_new


# In[17]:


# remove short words (length < 3)
review_train['Review Text'] = review_train['Review Text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

# remove stopwords from the text
reviews = [remove_stopwords(r.split()) for r in review_train['Review Text']]

# make entire text lowercase
reviews = [r.lower() for r in reviews]


# In[18]:


freq_words(reviews, 35)


# In[36]:


import spacy


# In[40]:


from spacy.cli.download import download


# In[43]:


nlp = spacy.load('en', disable=['parser', 'ner'])


# In[44]:


def lemmatization(texts, tags=['NOUN', 'ADJ']):
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output

tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
print(tokenized_reviews[1])


# In[45]:


print(tokenized_reviews[1])
len(tokenized_reviews[1])


# In[46]:


reviews_2 = lemmatization(tokenized_reviews)
print(reviews_2[1])


# In[47]:


print(reviews_2[1])
len(reviews_2[1])


# In[48]:


set(tokenized_reviews[1]) - set(reviews_2[1])


# In[50]:


reviews_3 = []
for i in range(len(reviews_2)):
    reviews_3.append(' '.join(reviews_2[i]))
    
review_train['Review Text'] = reviews_3


# In[51]:


freq_words(review_train['Review Text'], 35)


# In[53]:


from gensim import corpora


# In[54]:


# Create the term dictionary of our corpus, where every unique term is assigned an index
dictionary = corpora.Dictionary(reviews_2)


# In[55]:


doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]


# In[58]:


import pyLDAvis
import pyLDAvis.gensim
import gensim


# In[59]:


LDA = gensim.models.ldamodel.LdaModel


# In[64]:


lda_model = LDA(corpus=doc_term_matrix,
                id2word=dictionary,
                num_topics=21, 
                random_state=100,
                chunksize=1000,
                passes=50)


# In[65]:


lda_model.print_topics()


# In[66]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
vis


# In[67]:


lda_model.print_topics()

