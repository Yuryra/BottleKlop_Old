
from minisom import MiniSom  
import bs4
import stop_words

#import numpy as np
#import matplotlib.pyplot as plt

#som = MiniSom(6, 6, 150, sigma=0.5, learning_rate=0.5)
#som.train_random(data, 100)


#!/usr/bin/env python
# coding: utf-8

# Poems analysis
# ----
# 
# In this notebook we will use Minisom to cluster poems from three different authors.
# 
# Requirements:
# - Glove vectors, https://nlp.stanford.edu/projects/glove/ glove.6B.50d.txt
# - Beautiful soup
# - An internet connection as the poems will be downlaoded from www.poemhunter.com

# In[1]:

#https://stackoverflow.com/questions/32538758/nameerror-name-get-ipython-is-not-defined



import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


from urllib.request import urlopen
html = urlopen("http://www.google.com/")
print(html)


# Retrieving the poems from poemhunter.com
# ----
# 
# ***Warning***: this may take a while.

# In[7]:


#from urllib import urlopen
from bs4 import BeautifulSoup


#-------------------------------------------------------------------------------

def scrape_poem(poem_url):
    poem_page = urlopen(poem_url).read()
    soup = BeautifulSoup(poem_page)
    poem = ''
    poem_string = soup.find_all("div", 
                                {"class": "KonaBody" })[0].find_all('p')[0]
    poem_string = str(poem_string)[3:-4].replace('<br/>', ' ')
    return unicodeOf(poem_string)
    #return unicode(poem_string, errors='ignore') - this is python2

def scrape_poems_index(poems_index_url):
    poems_index = urlopen(poems_index_url).read()    
    soup = BeautifulSoup(poems_index)
    pages = soup.find_all("div", {"class": "pagination"})
    if len(pages) == 0:
        return get_all_links(soup)
    
    pages = pages[0].find_all('a')
    result = {}
    cnt = 0
    for page in pages:
        page_link = 'https://www.poemhunter.com/'+page['href']
        try:
          page_soup = BeautifulSoup(urlopen(page_link))
        except Exception as e:
          continue
        
        result.update(get_all_links(page_soup))
    return result

def get_all_links(page_soup):
    result = {}
    for link in page_soup.find_all('table')[0].find_all('a'):
        result[link.text] = 'https://www.poemhunter.com/'+link['href']
    return result

def get_poems(poems_index, max_poems=None):
    poems = {}
    for i, (title, poem_url) in enumerate(poems_index.items()):
        print ('fetching', title, '...'),
        try:
            poems[title] = scrape_poem(poem_url)
            print('OK')
        except Exception as e:
            #print ('impossible to fetch')
            print(e)
        if i == max_poems-1:
            return poems
        if i > 2: break #########################################!!!!
    return poems

def unicodeOf(unicode_or_str):
  if isinstance(unicode_or_str, str):
    text = unicode_or_str
    decoded = False
  else:
    text = unicode_or_str.decode(encoding)
    decoded = True
  return text
# In[13]:


poems_index_neruda = scrape_poems_index('https://www.poemhunter.com/pablo-neruda/poems/')
poems_index_bukowski = scrape_poems_index('https://www.poemhunter.com/charles-bukowski/poems/')
poems_index_poe = scrape_poems_index('https://www.poemhunter.com/edgar-allan-poe/poems/')


# In[ ]:


poems_neruda = get_poems(poems_index_neruda, max_poems=60)
poems_bukowski = get_poems(poems_index_bukowski, max_poems=60)
poems_poe = get_poems(poems_index_poe, max_poems=60)


# In[ ]:

#Clearly you're passing in d.keys() to your shuffle function. Probably this was written with python2.x (when d.keys() returned a list). With python3.x, d.keys() returns a dict_keys object which behaves a lot more like a set than a list. As such, it can't be indexed.

all_poems = [poems_neruda, poems_bukowski, poems_poe]
titles = np.concatenate([list(title_list.keys()) for title_list in all_poems])
y = np.concatenate([[i]*len(p) for i, p in enumerate(all_poems)])
all_poems = np.concatenate([list(p.values()) for p in all_poems])


# Preprocessing of the poems
# ---
# 
# The following operations are applied:
# 
# 1. stopwords removal
# 2. tokenization
# 3. conversion in Glove vectors

# In[7]:


from string import punctuation
import stop_words
stopwords = stop_words.get_stop_words('english')

def tokenize_poem(poem):
    poem = poem.lower().replace('\n', ' ')
    for sign in punctuation:
        poem = poem.replace(sign, '')
    tokens = poem.split()
    tokens = [t for t in tokens if t not in stopwords and t != '']
    return tokens

tokenized_poems = [tokenize_poem(poem) for poem in all_poems]


# In[8]:


def gimme_glove():
    #with open('glove.6B/glove.6B.50d.txt') as glove_raw:
    with open('c:/ml_space/glove.6B.50d.txt', encoding="utf8") as glove_raw:
        for line in glove_raw.readlines():
            splitted = line.split(' ')
            yield splitted[0], np.array(splitted[1:], dtype=np.float)
            
glove = {w: x for w, x in gimme_glove()}

def closest_word(in_vector, top_n=1):
    vectors = glove.values()
    idx = np.argsort([np.linalg.norm(vec-in_vector) for vec in vectors])
    return [glove.keys()[i] for i in idx[:top_n]]


# In[9]:


def poem_to_vec(tokens):
    words = [w for w in np.unique(tokens) if w in glove]
    return np.array([glove[w] for w in words])

W = [poem_to_vec(tokenized).mean(axis=0) for tokenized in tokenized_poems]
W = np.array(W)


# Running minisom and visualizing the result
# ----
# ***Warning***: This may take a while.

# In[16]:


from minisom import MiniSom
map_dim = 16
som = MiniSom(map_dim, map_dim, 50, sigma=1.0, random_seed=1)
#som.random_weights_init(W)
som.train_batch(W, len(W)*500)


# In[17]:


author_to_color = {0: 'chocolate',
                   1: 'steelblue',
                   2: 'dimgray'}
color = [author_to_color[yy] for yy in y]


# In[18]:


plt.figure(figsize=(14, 14))
for i, (t, c, vec) in enumerate(zip(titles, color, W)):
    winnin_position = som.winner(vec)
    plt.text(winnin_position[0], 
             winnin_position[1]+np.random.rand()*.9, 
             t,
             color=c)

plt.xticks(range(map_dim))
plt.yticks(range(map_dim))
plt.grid()
plt.xlim([0, map_dim])
plt.ylim([0, map_dim])
###plt.plot()
plt.show()

x = 1
# In[ ]:



