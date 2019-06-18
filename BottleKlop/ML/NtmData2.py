import os
import glob #file system lib
#import tarfile
import gc # #https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python

import gensim 
from gensim.models import KeyedVectors
#from gensim.test.utils import get_tmpfile

import sklearn 
#from sklearn import preprocessing

import numpy
import xml.etree.ElementTree as ET  
import Ntm2
from Ntm2 import *


import mxnet as mx
from mxnet import nd

import globals # from this project

class NtmData:#(object):
    #"""
    #  this is an example of docstring
    #  \ml_space
    #    \models
    #      \model1
    #        ...savedParams .. c:\temp\params.bin
    #    \wordvecs .. C:\Temp\Word2Vec_From_Out_2_150
    #    \traindata .. C:\Temp\Word2Vec_From_Out_2_150\train_ge_cat_2019-2-04.npy
    #    \rawdata - sample_files_out and such .. c : \ users \ yury\GitClonesHere\NtmData\FromProxy\sample_files_out
    #"""
    def __init__(self, dbg = set()):
        try:
            self.rawdata = globals.Ml_Space_Root + "Rawdata"
            self.traindata = globals.Ml_Space_Root + "Traindata" 
            self.wordvecs = globals.Ml_Space_Root + "TrainSaved\Wordvecs"#r"C:\Temp\Word2Vec_From_Out_2_150"

            self.ntmFilesDpath =  self.rawdata + "\\" + r"sample_files_out" #r"C:\Users\yury\GitClonesHere\NtmData\FromProxy\sample_files_out"
            self.ntmDclustDpath = self.rawdata + "\\" + r"sample_files_urls" #r"C:\Users\yury\GitClonesHere\NtmData\FromProxy\sample_files_urls"
            self.ntmNoclsDpath= self.rawdata + "\\" + r"sample_files_nocls"
            

            self.textVectors = "w2vecNEW.bin"

            self.domainVectors = "w2vecDomains.bin"
            
            self.geoVectors = "w2vecGG.bin"


            self.kvText = None
            self.kvDomain = None
            self.kvGeo = None
            
            self.kvText_size = 150
            self.kvDomain_size = 20
            self.kvGeo_size = 20


            self.ntmTrainFpath = self.traindata + "\\" + r"train_ge_cat_2019-2-04.npy" #self.vectorsDpath + "\\" + "train_100x100.npy" #"train_2019-2-4.npy" #"train.npy"
            self.ntmTrainFpath_Nocls = self.traindata + "\\" + r"train_noclsBlahBlah.npy" #self.vectorsDpath + "\\" + "train_100x100.npy" #"train_2019-2-4.npy" #"train.npy"
            #self.ntmTestFapth = self.vectorsDpath + "\\" + "test.npy"

            self.modelParamsSaved = globals.Ml_Space_Root +  "TrainSaved\Models" + "\\" + "mlp_scratch" + "\\" + "params.bin"

            self.dbg = dbg

            #self.loadVectors()
         
                  #this is really a list of themes or something.. fo now:
            self.catList = ["Office","Industrial","Retail Real Estate","Multifamily","Hospitality","Self-storage","Senior housing","Mixed-use"]
            self.catDict = {}
            for i, cat in enumerate(self.catList): 
              hot = numpy.zeros(len(self.catList))
              hot[i] = 1  
              self.catDict[cat.lower()] = hot 
 
        except Exception as e:
            print(e)
            raise
        x = 1

    #@classmethod
    #def classmethod(cls):
    #    return 'class method called', cls

    #@staticmethod
    #def staticmethod():
    #    return 'static method called'

    def loadVectors(self):

      if self.kvText is None :
        from Ntm2 import docs_vectorizer

        self.kvText = KeyedVectors.load(self.wordvecs + "\\" + self.textVectors, mmap="r")
        self.kvDomain = KeyedVectors.load(self.wordvecs + "\\" + self.domainVectors, mmap="r")
        self.kvGeo = KeyedVectors.load(self.wordvecs + "\\" + self.geoVectors, mmap="r")

        #dbgSet = set(['dclust', 'dprint']) #'set(['dprint', 'fprint', 'fhead'])
        #dry = NtmDirectoryReaderY(dpath, dbgSet)
        #documents = list(dry.yield_paragraphs(gensim.utils.simple_preprocess))
        
    def play(self, dbg = None):

      #https://gluon.mxnet.io/chapter01_crashcourse/ndarray.html#Converting-from-MXNet-NDArray-to-NumPy
      #a = x.asnumpy()
      #y = nd.array(a)



      self.loadVectors()
      x = self.kvText['jll']
      y = self.kvText['cushman']   #https://stackoverflow.com/questions/21979970/how-to-use-word2vec-to-calculate-the-similarity-distance-by-giving-2-words 
      cosine_similarity = numpy.dot(x, y)/(numpy.linalg.norm(x)* numpy.linalg.norm(y))

      d = self.kvText.distance(npJll, npJll)
      print(d)

    def probTo(self, npVector):
      #make sure this are all positive 
      npVector = (1 + npVector / numpy.dot(npVector, npVector)) / 2
      npVector = npVector / numpy.sum(npVector)
      return npVector



    def select_by_label(self, data = None, labels = None, good_labels = None):
      #clear data
      good_labels = np.array(good_labels)
      y_index = 3
      badRowIndexList = []
      for i, label in enumerate(labels):
        x  = 1
        overlap = np.intersect1d(label, good_labels)
        if overlap.size == 0 :
          badRowIndexList.append(i)

      badRowIndexes = numpy.array(badRowIndexList)


      ddd = numpy.delete(data, badRowIndexes, 0)
      lll = numpy.delete(labels, badRowIndexes, 0)
      return ddd, lll
    

    def loadSomWrap(self, num_examples = None, starting_index = 0,  rescaling = None):
        
      W, Y = self.loadTextAndCls(num_examples = num_examples, starting_index = starting_index, rescaling = rescaling)
      labels = np.zeros(len(Y), dtype=int)
      for i, label in enumerate(Y):
        labels[i] = np.argmax(label)
      del Y

      gc.collect()

      W = np.vstack(W).astype(np.float32)

      if rescaling == "aaa":
        uuu = numpy.ones((W.shape))
        W = numpy.add(W,uuu) / 2
    
        x = numpy.sqrt(numpy.dot(W[0], W[0].T))
        W = sklearn.preprocessing.normalize(W)
        x = numpy.sqrt(numpy.dot(W[0], W[0].T))
        pass

      return W, labels
    
    TrainDict = { 'txt': 0 #     txt_index = 0
      , 'dmn' : 1 #dmn_index = 1
      , 'geo' : 2 # geo_index = 2
      , 'cls' : 3 #cls_index = 3
      , 'st2' : 4 #st2_index = 4
    }

    def loadVersus(self, num_examples = None, starting_index = 0,  dbg = None, fpath= None, rescaling = None
                   , train_from = None, train_to = None):

      txt_index = NtmData.TrainDict['dmn']

      #dmn_index = 1
      #geo_index = 2
      #cls_index = 3
      #st2_index = 4

      #data_index = 0 #1 #2
      #label_index = 4

      data_index = NtmData.TrainDict[train_from]
      label_index = NtmData.TrainDict[train_to]



      if fpath is None: 
        fpath = self.ntmTrainFpath
        fpath = "c:\\temp\\ml\\gg_temp.npy"

      npTrain = numpy.load(fpath)
      


      if num_examples is None: 
        num_examples = len(npTrain) - starting_index

      data = npTrain[starting_index:starting_index + num_examples,data_index]
      labels = npTrain[starting_index:starting_index + num_examples,label_index]

      
      del npTrain
      gc.collect()

      data = np.vstack(data).astype(np.float32)

      if train_to == 'cls':
        ll = np.zeros((len(labels)),np.int)
        for i, l in enumerate(labels):
          ll[i] = np.argmax(l)

        labels = ll
      else:
        labels = np.vstack(labels).astype(np.int)
      





      return data, labels
    
    def loadTextAndCls(self, num_examples = None, starting_index = 0,  dbg = None, fpath= None, rescaling = None):
      if fpath is None: 
        fpath = self.ntmTrainFpath
        fpath = r"C:\Ml_space\traindata\train_ge_cat_2019-2-04.npy"

      npTrain = numpy.load(fpath)
      
      y_index = 3

      #clear data
      badRowIndexList = []
      for i, row in enumerate(npTrain):
        if not row[y_index].any() :
          badRowIndexList.append(i)

      badRowIndexes = numpy.array(badRowIndexList)


      cleared = numpy.delete(npTrain, badRowIndexes, 0)

      if num_examples is None: 
        num_examples = len(cleared) - starting_index

      txt = cleared[starting_index:starting_index + num_examples,0]
      labels = cleared[starting_index:starting_index + num_examples,y_index]



      del badRowIndexList
      del badRowIndexes
      del npTrain
      gc.collect()

      return txt, labels

    def loadData(self, batch_size, number_batches, shuffle = False, dbg = None):
      fpath = self.ntmTrainFpath
      fpath = r"C:\Temp\Word2Vec_From_Out_2_150\train_ge_cat_2019-2-04.npy"
      npTrain = numpy.load(fpath)
      
      isTraining = True
      l = len(npTrain)

      num_inputs = self.kvText_size
      num_outputs = self.kvGeo_size
      num_examples = batch_size * number_batches #int(l * 0.8)

      #for now kill all 
      
      y_index = 3


      #xxx = numpy.count_nonzero(npTrain,axis = y_index) 


      #badRowIndexes = numpy.where(numpy.count_nonzero(npTrain[:,y_index]) == 0)
      #badRowIndexes = numpy.where(numpy.count_nonzero(npTrain[:,y_index]) == 0)
      
           

      badRowIndexList = []
      for i, row in enumerate(npTrain):
        if not row[y_index].any() :
          badRowIndexList.append(i)

      badRowIndexes = numpy.array(badRowIndexList)


      cleared = numpy.delete(npTrain, badRowIndexes, 0)

      train_dataset = mx.gluon.data.dataset.ArrayDataset(cleared[:num_examples,0], cleared[:num_examples,y_index])
      train_data_loader = mx.gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = shuffle) #, num_workers=CPU_COUNT)

      num_tests = int(num_examples / 4)
      test_dataset = mx.gluon.data.dataset.ArrayDataset(cleared[num_examples:num_examples + num_tests, 0]
                                                    , cleared[num_examples:num_examples + num_tests,y_index])
      test_data_loader = mx.gluon.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = shuffle) #, num_workers=CPU_COUNT)
      

      return train_data_loader, test_data_loader, num_inputs, num_outputs, num_examples, batch_size




      ndX = nd.empty((num_examples,num_inputs))
      ndY = nd.empty((num_examples,self.kvGeo_size))



      offset = 0
      for i, row in enumerate(npTrain[0:20000,]):
        npDescriptionVector = row[0]
        ndX[i - offset] = nd.array(npDescriptionVector)
        npGeoVector = row[2]
        
        # if will make it look as if probabilities ..
        #ndGeo = nd.array(self.probTo(npGeoVector))

        ndGeo = nd.array(npGeoVector)
        #make sure its normed
        ndGeo /= nd.norm(ndGeo)


        ndY[i - offset] = ndGeo
        
        
        if i == num_examples - 1 and isTraining:
          dataset = mx.gluon.data.dataset.ArrayDataset(ndX, ndY)
          train_data_loader = mx.gluon.data.DataLoader(dataset, batch_size=batch_size) #, num_workers=CPU_COUNT)

          break

          offset = i
          dataset = None
          ndX = nd.empty((l - i,num_inputs))
          ndY = nd.empty((l - i,num_outputs))
          isTraining = False

      dataset = mx.gluon.data.dataset.ArrayDataset(ndX, ndY)
      test_data_loader = mx.gluon.data.DataLoader(dataset, batch_size=batch_size, shuffle = shuffle) #, num_workers=CPU_COUNT)

      

      return train_data_loader, test_data_loader, num_inputs, num_outputs, num_examples, batch_size

    def test(self, dbg = None):


      
      npTrain = numpy.load(self.ntmTrainFpath)
      l = len(npTrain)
      ndX = nd.empty((l,150))
      ndY = nd.empty((l,20))
      for i, row in enumerate(npTrain):
        npDescriptionVector = row[0]
        ndX[i] = nd.array(npDescriptionVector)
        npGeoVector = row[2]
        ndY[i] = nd.array(npGeoVector)

      dataset = mx.gluon.data.dataset.ArrayDataset(ndX, ndY)

      #http://mxnet.incubator.apache.org/versions/master/tutorials/gluon/datasets.html
      from multiprocessing import cpu_count
      CPU_COUNT = cpu_count()

      CPU_COUNT = 1

      data_loader = mx.gluon.data.DataLoader(dataset, batch_size=5) #, num_workers=CPU_COUNT)

      for X_batch, y_batch in data_loader:
        print("X_batch has shape {}, and y_batch has shape {}".format(X_batch.shape, y_batch.shape))




      ndTrain0 = nd.array(npTrain[:,0])


      mx.random.seed(42) # Fix the seed for reproducibility
      X = mx.random.uniform(shape=(10, 3))
      y = mx.random.uniform(shape=(10, 1))
      dataset = mx.gluon.data.dataset.ArrayDataset(X, y)

      x= 1

    def testRead(self, dbg = None):
      #npTrain = numpy.array([])

      try:
        dtype = 'O'
        #xxx = numpy.fromfile(self.ntmTrainFpath, dtype, -1, sep=",")
        xxx = numpy.load(self.ntmTrainFpath)
        print(xxx.dtype)
      except Exception as e:
        print(e)
        raise



    def vectors_from_url_and_paragrpahs(self, item_NtmItemReader):
      item = item_NtmItemReader
      ### npDomain = []
      #print(item.url)
      from urllib.parse import urlparse
      # from urlparse import urlparse  # Python 2
      parsed_uri = urlparse(item.url )
      d = parsed_uri.hostname #['{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)]
      npDomain = docs_vectorizer(numpy.array([d]), self.kvDomain, self.kvDomain.vector_size)

      ### npText = []
      allTextList = [] #numpy.array([])
      for p in item.yield_paragraphs(My_n_grams): # should COMBINED paragraph
        #vvv = docs_vectorizer(p, self.kvText, self.kvText.vector_size)
        allTextList.extend(p)#numpy.concatenate((allTextTokens,p))
        #print(p)

      npText = docs_vectorizer(allTextList, self.kvText, self.kvText.vector_size)

      npCat = numpy.zeros(len(self.catDict))
      for cat in item.catList:
              
        if cat.lower() == "mixed-use":
          #skip
          x = 1
        elif cat.lower() in self.catDict:
          npCat = npCat + self.catDict[cat.lower()]
        #else:
          #print("missing in catDict:" + cat)

      return npDomain, npText, npCat

    def vectors_from_gg(self, gg_NtmItemReaderGG):
      ### npGeo = []
      gg = gg_NtmItemReaderGG #NtmItemReaderGG(itemEl)
      #print(gg.ggZips)

      npGeo = docs_vectorizer(gg.ggZips, self.kvGeo, self.kvGeo.vector_size)
            
      return npGeo

    @staticmethod
    def GetCategories1(categoriser, npText):
      ddd = mx.nd.array(npText.astype(numpy.float32)) #nd.Array(npText).astype(numpy.float32))
      #label = mx.nd.array(npCat.astype(numpy.float32)) #nd.Array(npCat).astype(numpy.float32)
      label_hat = categoriser.do1(ddd)
      #print(label_hat)
      #print(label)

      softResult = softmax1(label_hat)
      return softResult.asnumpy()

    def getCategories(self, categoriser, fry):
      try:
        rootOut = ET.Element("root")
        rootOut.set('categoriser', categoriser.describe())

        doc = ET.SubElement(rootOut, "doc")

        ET.SubElement(doc, "field1", name="blah").text = "some value1"
        f2 = ET.SubElement(doc, "field2", name="asdfasd")
        f2.text = "some vlaue2"
        f2.set('status', 'completed')


        fry.getItems()
        i = 0
        for itemEl in fry.itemEls: # here i do it 1 by 1 for each item .. i could pack the in a matrix .. do net
              

          item = NtmItemReader(itemEl)


          npDomain, npText, npCat = self.vectors_from_url_and_paragrpahs(item)

          ddd = mx.nd.array(npText.astype(numpy.float32)) #nd.Array(npText).astype(numpy.float32))
          label = mx.nd.array(npCat.astype(numpy.float32)) #nd.Array(npCat).astype(numpy.float32)
          label_hat = categoriser.do1(ddd)
          print(label_hat)
          print(label)

          softResult = softmax1(label_hat)
          print(softResult)
          print(softmax1(label))

          el = ET.SubElement(doc, "cats", nm="asdfasd", xx ="xxx")
          el.text = ','.join(str(p) for p in softResult.asnumpy().tolist())
          el.set('status', 'completed')

      except Exception as e:
        print(e)
        raise
      return rootOut
############################

    def testWrite(self,  dbg = None):
      
      dbgLocal = self.dbg
      if dbg is not None: dbgLocal =dbg


      ### get 
      #dbgSet = set(['dclust', 'dprint', 'fhead']) #'set(['dprint', 'fprint', 'fhead'])
      #dry = NtmDirectoryReaderY(ntmData.ntmDclustDpath, dbgSet)
      ##documents = list(dry.yield_paragraphs()) 
      #for ccc in dry.yield_paragraphs():
      #  print(ccc)
      #self.kvGeo = KeyedVectors.load(self.vectorsDpath + "\\" + self.geoVectors, mmap="r")

      self.loadVectors()

      try:

        #operating with lists is much faster than with ndarray's
        trainList = [] # numpy.array([])
        testList = [] #numpy.array([])

        # i have to read directory hierarchy
        for n, fpath in enumerate(glob.glob(os.path.join(self.ntmFilesDpath, "*.xml"))):
          
          if 'fhead' in dbgLocal: 
                if n > 100: break # prints only 3 first items
            
          
          if 'dprint' in dbgLocal: print(":: fpath ::" + fpath)


          fry = NtmReaderXml.fromFile(fpath,self.dbg) #NtmFileReaderY(fpath,dbgLocal)
          fry.getItems()
          i = 0
          for itemEl in fry.itemEls:
              
            if 'fhead' in dbgLocal: 
                if i > 100: break # prints only 3 first items
            i = i + 1

            
            if True:
              item = NtmItemReader(itemEl)
              npDomain, npText, npCat = self.vectors_from_url_and_paragrpahs(item)

              gg = NtmItemReaderGG(itemEl)
              npGeo = self.vectors_from_gg(gg)
              
            else:
              item = NtmItemReader(itemEl)
     
              ### npDomain = []
              from urllib.parse import urlparse
              # from urlparse import urlparse  # Python 2
              parsed_uri = urlparse(item.url )
              d = parsed_uri.hostname #['{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)]
              npDomain = docs_vectorizer(numpy.array([d]), self.kvDomain, self.kvDomain.vector_size)

              ### npText = []
              allTextList = [] #numpy.array([])
              for p in item.yield_paragraphs(My_n_grams): # should COMBINED paragraph
                #vvv = docs_vectorizer(p, self.kvText, self.kvText.vector_size)
                allTextList.extend(p)#numpy.concatenate((allTextTokens,p))
                #print(p)
              npText = docs_vectorizer(allTextList, self.kvText, self.kvText.vector_size)


            ### npGeo = []
              gg = NtmItemReaderGG(itemEl)


              #print(gg.ggZips)

              npGeo = docs_vectorizer(gg.ggZips, self.kvGeo, self.kvGeo.vector_size)
            
              npCat = numpy.zeros(len(self.catDict))
              for cat in item.catList:
              
                if cat.lower() == "mixed-use":
                  #skip
                  x = 1
                elif cat.lower() in self.catDict:
                  npCat = npCat + self.catDict[cat.lower()]
                else:
                  print("missing in catDict:" + cat)

            trainList.append([npText, npDomain, npGeo, npCat, gg.st2_index])


        npTrain = numpy.array(trainList)


        if True:
          #npTrain.tofile(self.ntmTrainFpath, sep=",")
          numpy.save(self.ntmTrainFpath, npTrain)
        else:
          #npTest.fileto(self.ntmTestFpath, sep="")
          numpy.save(self.ntmTestFpath, npTest)

      except Exception as e:
        print(e)
        raise

    def testWrite_Nocls(self,  dbg = None):
      
      dbgLocal = self.dbg
      if dbg is not None: dbgLocal =dbg

      self.loadVectors()

      try:

        #operating with lists is much faster than with ndarray's
        trainList = [] # numpy.array([])
        testList = [] #numpy.array([])


        for n, fpath in enumerate(glob.glob(os.path.join(self.ntmNoclsDpath + "/*/*", "*.xml"))):
          
          if 'fhead' in dbgLocal: 
                if n > 100: break # prints only 3 first items
            
          
          if 'dprint' in dbgLocal: print(":: fpath ::" + fpath)


          fry = NtmReaderXml.fromFile(fpath,self.dbg) #NtmFileReaderY(fpath,dbgLocal)
          fry.getItems()
          i = 0
          for itemEl in fry.itemEls:
              
            if 'fhead' in dbgLocal: 
                if i > 100: break # prints only 3 first items
            i = i + 1

            
            if True:
              item = NtmItemReader(itemEl, use_snap=True)
              npDomain, npText, npCat = self.vectors_from_url_and_paragrpahs(item)

              gg = NtmItemReaderGG(itemEl)
              npGeo = self.vectors_from_gg(gg)


            trainList.append([npText, npDomain, npGeo, npCat])


        npTrain = numpy.array(trainList)


        if True:
          #npTrain.tofile(self.ntmTrainFpath, sep=",")
          numpy.save(self.ntmTrainFpath_Nocls, npTrain)


      except Exception as e:
        print(e)
        raisebbb