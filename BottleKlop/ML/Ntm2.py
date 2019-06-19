import os
import glob #file system lib

import xml.etree.ElementTree as ET  



St2Dict = { # inlcudes guam and such territories ..
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}
St2List = list(St2Dict.keys())

#items = ['a', 'b', 'c']
#items.index('a') # gives 0
#items[0]         # gives 'a'


class NtmDclustReader: #gets list of domain in cluster
    def __init__(self, dclustEl):
        try:

          #self.dc = dclustEl.attrib["dnms"].split(";")
                    
          self.dc = list()
          for dmn in dclustEl.attrib["dnms"].split(";"):
            self.dc.append(dmn)

        except Exception as e:
            raise

class NtmFileReaderZZZZ:
    """description of class"""
    def __init__(self, fpath, dbg = set()):
        try:
            self.fpath = fpath
            self.dbg = dbg
            
        except Exception as e:
            raise
        x = 1
    
    def test(self, pp = None):
        self.tree = ET.parse(self.fpath)  
        self.root = self.tree.getroot()
        self.itemEls = self.root.findall("dclust")
        #self.items = []
        i = 0
        for itemEl in self.itemEls:
          #xxx= yyy   
          try:

            if 'fhead' in self.dbg: 
                if i > 2: break # prints only 3 first items
            i = i + 1

            if 'dclust' in self.dbg:
              #return NtmDclustReader(itemEl)

              item = NtmDclustReader(itemEl)
              p= item.dc
            #  #if 'pprint' in self.dbg: print("::      paragraph ::" + str(j) + " " +  p)
              yield p
        


          except Exception as e:
              #exc_type, exc_obj, exc_tb = sys.exc_info()
              raise

        return None


    def yield_paragraphs(self, pp = None):
        #return ["aaa","bbb","ccc"]

        self.tree = ET.parse(self.fpath)  
        self.root = self.tree.getroot()
        self.itemEls = self.root.findall("dclust")
        #self.items = []
        i = 0
        for itemEl in self.itemEls:
                
          try:

            if 'fhead' in self.dbg: 
                if i > 2: break # prints only 3 first items
            i = i + 1

            if 'dclust' in self.dbg:
              item = NtmDclustReader(itemEl)
              p= item.dc
              #if 'pprint' in self.dbg: print("::      paragraph ::" + str(j) + " " +  p)
              yield p
        


          except Exception as e:
              #exc_type, exc_obj, exc_tb = sys.exc_info()
              raise

          return None
#---------------------------------------------------


class NtmItemReaderGG: #gets list of gzips
    def __init__(self, itemEl):
        try:

          ns = {'seaglex': "http://www.seaglex.com/ns",
                'geo': "http://www.w3.org/2003/01/geo/wgs84_pos#"}

          # a typical entry has sonething like this :
          # ...
          #<seaglex:ggZip prb="1.00">Westerville 43082, OH</seaglex:ggZip>
          #<seaglex:ggZip prb="1.00">Westerville, OH</seaglex:ggZip>
          #<seaglex:ggZip prb="1.00">Columbus, OH</seaglex:ggZip>
          #<seaglex:ggZip prb="1.00">Delaware County, OH</seaglex:ggZip>
          #<seaglex:ggZip prb="1.00">Polaris, Columbus, OH</seaglex:ggZip>
          #<seaglex:ggZip prb="1.00">Columbus 43240, OH</seaglex:ggZip>
          #<seaglex:ggZip prb="1.00">Columbus City Township, OH</seaglex:ggZip>
          #<seaglex:ggZip prb="1.00">Lewis Center 43035, OH</seaglex:ggZip>
          #<seaglex:ggZip prb="1.00">Orange Township, OH</seaglex:ggZip>
          #<seaglex:ggZip prb="1.00">OH</seaglex:ggZip>
          #self.st2_index = 
          

          self.ggZips = list()
          for g in itemEl.findall("seaglex:ggZip", ns):
            self.ggZips.append(g.text)
            if g.text.find(',') == -1 : 
              self.st2_index  = St2List.index(g.text)
              if self.st2_index >= 50:
                pass
        except Exception as e:
            raise


class NtmItemReader:
    def __init__(self, itemEl, dbg = set(), use_snap=False):
        try:
            self.dbg = dbg
            self.title = itemEl.find("title").text
            self.url = itemEl.find("link").text
            dscrEl = itemEl.find("description")
            self.catList = []
            for catEl in itemEl.findall("category"):
              self.catList.append(catEl.text)

            #get all classifications here .. later
            ###dscrEl = itemEl.find("description")
            textEl = None
            #description = dscrEl.text
            

            if use_snap:
              ns = {'seaglex': "http://www.seaglex.com/ns",
                'geo': "http://www.w3.org/2003/01/geo/wgs84_pos#"}
              textEl = itemEl.find("seaglex:snap", ns)
            else:
              textEl = dscrEl


            self.paragraphs = textEl.text.split("\n")
            x = 1
        except Exception as e:
            raise

    def yield_paragraphs(self, pp = None):
        try:                     
          if 'fprint' in self.dbg: print("::  title ::" + str(i) + " " +  item.title)

          title = self.title
          if pp is not None :
              title = pp (title)
          yield title
                
          j = 0
          for p in self.paragraphs:
              if pp is not None :
                  p = pp (p)
              else:
                  if 'pprint' in self.dbg: print("::      paragraph ::" + str(j) + " " +  p)

              yield p #"p " + str(j) #p
              j = j + 1
        except Exception as e:
            raise


class NtmReaderXml:
    """description of class"""
    def __init__(self, xmlTree, dbg = set()):
      try:
        self.tree = xmlTree
        self.dbg = dbg
            
      except Exception as e:
        raise

    @classmethod #a different constructor
    def fromXml(cls, xmlString, dbg = set()):
      try:
        root = ET.fromstring(xmlString)     
        tree = ET.ElementTree(root)
        return cls(tree, dbg)
            
      except Exception as e:
        print(e)
        raise
     
    @classmethod #a different constructor
    def fromFile(cls, fpath, dbg = set()):
      try:
        tree = ET.parse(fpath)  
        return cls(tree, dbg)
            
      except Exception as e:
        print(e)
        raise


    def getItems(self):
      try:
        #self.tree = ET.parse(self.fpath)  
        self.root = self.tree.getroot()
        xpath = "channel/item"
        if 'dclust' in self.dbg:
          xpath = "dclust"
        self.itemEls = self.root.findall(xpath)
        #self.items = []
        i = 0
      except Exception as e:
        #exc_type, exc_obj, exc_tb = sys.exc_info()
        print(e)
        raise

    def yield_paragraphs(self, pp = None):
        try:

            self.getItems()

            i = 0
            for itemEl in self.itemEls:
                


                if 'fhead' in self.dbg: 
                    if i > 2: break # prints only 3 first items
                i = i + 1


                if 'dclust' in self.dbg:
                  item = NtmDclustReader(itemEl)
                  p= item.dc
                  if 'pprint' in self.dbg: print("::      paragraph ::" + str(j) + " " +  p)
                  yield p

                elif 'gg' in self.dbg:
                  item = NtmItemReaderGG(itemEl)
                  p= item.ggZips
                  if 'pprint' in self.dbg: print("::      paragraph ::" + str(j) + " " +  p)
                  yield p

                else:

                  item = NtmItemReader(itemEl)
                  if 'fprint' in self.dbg: print("::  title ::" + str(i) + " " +  item.title)

                  title = item.title
                  if pp is not None :
                      title = pp (title)
                  yield title
                
                  j = 0
                  for p in item.paragraphs:
                      if pp is not None :
                          p = pp (p)
                      else:
                          if 'pprint' in self.dbg: print("::      paragraph ::" + str(j) + " " +  p)

                      yield p #"p " + str(j) #p
                      j = j + 1

        except Exception as e:
            #exc_type, exc_obj, exc_tb = sys.exc_info()
            print(e)
            raise

        return None


#import module1

class NtmDirectoryReaderY:
    def __init__(self, dpath, dbg = set()):
        try:
            self.dpath = dpath
            self.dbg = dbg
         
        except Exception as e:
            raise
        x = 1

    def yield_paragraphs(self, pp = None):
        try:
            for fpath in glob.glob(os.path.join(self.dpath, "*.xml")):
                if 'dprint' in self.dbg: print(":: fpath ::" + fpath)
                if False:
                    fry = NtmReaderXml.fromFile(fpath,self.dbg) #NtmFileReaderY(fpath,self.dbg)
                    yield fry.yield_paragraphs(pp) #.__next__()
                    #return fpath #fry.yield_paragraphs()

                elif 'dclustXXX' in self.dbg: 
                    fry = NtmFileReaderZZZZ(fpath,self.dbg)
                    yield fry.test()
                    #yield fry.yield_paragraphs(pp) #.__next__()
                    

                elif True:
                    fry = NtmReaderXml.fromFile(fpath,self.dbg) #NtmFileReaderY(fpath,self.dbg)
                    for x in fry.yield_paragraphs(pp):
                        yield x
                    #return fpath #fry.yield_paragraphs()
                else:
                    yield fpath
        except Exception as e:
            raise


# --------------------------- useful np stuff -------------------
import numpy as np
from mxnet import nd
#for doc i will return a normalized vector (https://stackoverflow.com/questions/30795944/how-can-a-sentence-or-a-document-be-converted-to-a-vector)
def docs_vectorizer(doc, model, size):
    #size = model.vector_size
    doc_vec = np.zeros(size, dtype = np.float32) # 400 ?-yyy should it be precizely the model 'size=150?'
    #return doc_vec
    numw = 0
    for w in doc:
        try:
            #x = model[w]
            #continue
            doc_vec = np.add(doc_vec, model[w])
            numw+=1
        except:
            pass
    return 0
    #x = np.sqrt(np.dot(doc_vec, doc_vec.T))
    #return doc_vec / x
    #return doc_vec / np.sqrt(np.dot(doc_vec, doc_vec.T))
    return doc_vec / np.sqrt(doc_vec.dot(doc_vec))
    

def My_n_grams(str, n = None):
  grams_1 = gensim.utils.simple_preprocess(str) #somehow and simingly strips html tags like <p> .. </p> off
  l = list()
  i = 0
  for gram1 in grams_1:
    l.append(gram1)
    if i>0 :
      l.append(grams_1[i-1] + " " + gram1)
    i = i + 1
  return l


def softmax1(vector):
    exp = nd.exp(vector-nd.max(vector))
    return exp / nd.nansum(exp)

# ------------------------ debuging stuff ----------------
import gensim 
def docs_forTestGet():
  docs = []
  #docs.append(gensim.utils.simple_preprocess("I want to test Marcus Millichap for real estate"))
  #docs.append(gensim.utils.simple_preprocess("I want to test CBRE for real estate"))
  #docs.append(gensim.utils.simple_preprocess("I want to test Ngkf for real estate"))
  #docs.append(gensim.utils.simple_preprocess("but why  Marcus Millichap do not correlate"))
  #docs.append(gensim.utils.simple_preprocess("they should be similar Marcus Millichap"))
  #docs.append(gensim.utils.simple_preprocess("humpty dumpty sat on Marcus Millichap"))
  #docs.append(gensim.utils.simple_preprocess("Marcus Millichap somehow are not together"))

  #docs.append(gensim.utils.simple_preprocess("they should be similar CBRE"))
  #docs.append(gensim.utils.simple_preprocess("humpty dumpty sat on CBRE"))
  #docs.append(gensim.utils.simple_preprocess("CBRE somehow are not together"))

  docs.append(gensim.utils.simple_preprocess("aaa bbb xxx ccc ddd"))
  
  docs.append(gensim.utils.simple_preprocess("aaa bbb yyy ccc ddd"))
  docs.append(gensim.utils.simple_preprocess("aaa bbb yyy ccc ddd"))
  docs.append(gensim.utils.simple_preprocess("aaa bbb yyy ccc ddd"))

  docs.append(gensim.utils.simple_preprocess("aaa bbb zzz ccc ddd"))
  docs.append(gensim.utils.simple_preprocess("aaa bbb zzz ccc ddd"))
  docs.append(gensim.utils.simple_preprocess("aaa bbb zzz ccc ddd"))
  
  docs.append(gensim.utils.simple_preprocess("aaa bbb ttt ccc ddd"))
  return docs