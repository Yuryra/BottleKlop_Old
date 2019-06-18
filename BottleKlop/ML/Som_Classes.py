import numpy as np
import h5py

import somoclu

#to deal with categorisation L 
from NtmData2 import *
#import NtmData2
import mxnet as mx
from mxnet import nd

def Lmap_collect(som, labels, surf_state = None, bmus = None, data = None): #labels, of course, correspon to .. bmus
  #for topographical error see https://github.com/peterwittek/somoclu/issues/87

    #som = self
    if surf_state is  None: surf_state = som_get_surface_state(som, data) #som.get_surface_state()
    if bmus is  None: bmus = som.get_bmus(surf_state)
    
    n_columns = som.codebook.shape[1] 
    n_rows = som.codebook.shape[0] 

    #labels.shape[0]
    n_labels = int(np.max(labels)) + 1
    lmap = np.zeros(( n_labels, n_columns, n_rows  ), dtype = int)
    for i, bmu in enumerate(bmus): #looping thru all data vectors
      label = int(labels[i])
      
      row = bmu[1]
      column = bmu[0]

      lmap[label, column, row] += 1

    return lmap #n_labels, n_columns, n_rows, lmap

def BmusPlus_Collect(surf_state, bmus):
  bmusPlus = np.zeros( (bmus.shape[0], 3)) #som._n_columns, som._n_rows))

  #0 is column's index
  n_columns = np.max(bmus[:,0]) + 1 #som._n_columns

  for i, bmu in enumerate(bmus): #looping thru all bmus vectors
    #label = int(labels[i])
      
    row = bmu[1]
    column = bmu[0]
    idx = column + n_columns * row # * r #column + n_columns * row
    dist = surf_state[i, idx]

    bmusPlus[i] = [column, row, dist]

  return bmusPlus


      #self.dc = ..
class Som_Data: #
  def __init__(self, fname = None):
    if fname is None:
      self.fname = "no_name"
    else:
      self.fname = fname

    self.som = None
    self.labels = None
    self.bmusPlus = None
    
    
  @classmethod 
  def data_from(class_object, som, data, labels, fname=None, categoriser = None):
      


    self = class_object(fname = fname)
    try:
          
      #compute perceptron's classifications:
      if not categoriser is None:

        #import NtmData2
        self.cats = np.zeros((len(data),8))
        for i, d in enumerate(data): #looping thru all bmus vectors
          x = NtmData.GetCategories1(categoriser, d)
          y = np.around(x,2)
          self.cats[i] = x
          pass


      # measure data's dist to all cells
      surf_state = som.get_surface_state(data) #som.get_surface_state()
      # find best cells for the data
      bmus = som.get_bmus(surf_state)

      
      rmin = surf_state.min(axis=1) 
      quantization_error = np.mean(rmin) # / surf_state.shape[0]



      self.labels = labels


      self.bmusPlus = np.zeros( (bmus.shape[0], 3)) #som._n_columns, som._n_rows))

      n_columns = som._n_columns



      for i, bmu in enumerate(bmus): #looping thru all bmus vectors
        #label = int(labels[i])
      
        row = bmu[1]
        column = bmu[0]
        idx = column + n_columns * row # * r #column + n_columns * row
        dist = surf_state[i, idx]

        self.bmusPlus[i] = [column, row, dist]
      #self.dc = ..

      qe = np.mean(self.bmusPlus[:,2])
      yyy = 1
    except Exception as e:
        raise
    
    return self

  @classmethod 
  def dpath_from(class_object, dpath, fname = None):
    self = class_object(fname = fname)
    # hell with self.som = None
    hf = h5py.File(dpath + "\\" + self.fname + ".h5", 'r')
    self.name = np.array(hf.get('name_dataset'))
    self.labels = np.array(hf.get('labels_dataset'))
    self.cats = np.array(hf.get('cats_dataset'))
    self.bmusPlus = np.array(hf.get('bmusPlus_dataset'))
    hf.close()

    return self

  def dpath_to(self, dpath):

    try:
      hf = h5py.File(dpath + "\\" + self.fname + ".h5", 'w')
      hf.create_dataset('name_dataset', data=self.fname)
      hf.create_dataset('labels_dataset', data=self.labels)
      hf.create_dataset('cats_dataset', data=self.cats)
      hf.create_dataset('bmusPlus_dataset', data=self.bmusPlus)
      hf.close()
    except Exception as e:
      raise

  @classmethod 
  def restricted_from(class_object, him, max_samples = None):
    self = class_object(fname = "restricted_" + him.fname)
    try:
      self.labels = him.labels[:max_samples]
      self.bmusPlus = him.bmusPlus[:max_samples]
    except Exception as e:
      raise
    return self

  @staticmethod
  def softmax2(vector):
    power = 10 ** (vector-np.max(vector))
    return power / np.nansum(power)

  @staticmethod
  def hardmax(vector):
    power = - (vector-np.max(vector))
    return power / np.nansum(power)

  def cmp(self, lmap):#, max_samples = None):
    #lmap = labels_map_collect(som, labels, surf_state, bmus, data)

    #if lmap is None :
    #  lmap = labels_map_collect(som, labels, surf_state, bmus, data)
      


    n_labels = lmap.shape[0]
    label_max = np.zeros(n_labels)
    for l in range(n_labels):
      label_max[l] = np.sum(lmap[l])

    distSum = 0
    lcount = np.zeros(n_labels)
    profile = np.zeros(n_labels)
    for i, bmu in enumerate(self.bmusPlus):

      #if max_samples is None:
      #  pass
      #elif i > max_samples:
      #  break
        
      column = int(self.bmusPlus[i, 0])
      row = int(self.bmusPlus[i, 1])
      dist = self.bmusPlus[i, 2]
      distSum +=dist

      label = int(self.labels[i])

      #if label != 0:
      #  continue
      

      for l in range(n_labels):
        lmap_weight = lmap[l, column, row] / label_max[l]
        x = (1.4 - dist) * lmap_weight
        profile[l] += x
        lcount[l] += lmap_weight

      ppp = mx.nd.array(profile)
      som_cats = softmax1(ppp)
      som_cats = som_cats.asnumpy()

      som_cats = self.hardmax(profile)

      perceptron_cats = self.cats[i]

      som_cats = np.around(som_cats,3)
      perceptron_cats = np.around(perceptron_cats,3)

      pass

    distSum /= len(self.bmusPlus)
    return profile
#---------------------------------


class Som_Lmaped():
  def __init__(self, fname = None):
    if fname is None:
      self.fname = "no_name"
    else:
      self.fname = fname

    self.som = None
    self.bmusPlus = None
    self.lmap = None


  #https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
  @classmethod 
  def data_from(class_object, som, data, labels, fname = None):
    self = class_object(fname = fname)
    try:
      self.som = som
      
      if data is None:
        data = self.som._data

      # measure data's dist to all cells
      
      surf_state = som.get_surface_state(data) #som.get_surface_state()
      # find best cells for the data
      bmus = som.get_bmus(surf_state)


      self.lmap = Lmap_collect(self.som, labels, surf_state, bmus, data)

      self.bmusPlus = BmusPlus_Collect(surf_state, bmus)

    except Exception as e:
        raise

    return self

  @classmethod #https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
  def dpath_from(class_object, dpath, fname = None):
      #'''Call as
      #   d = Som_Lmaped.dpath_from('....')
      #'''

    self = class_object(fname = fname)

    hf = h5py.File(dpath + "\\" + self.fname + ".h5", 'r')

    self.lmap = np.array(hf.get('lmap_dataset'))
    n_labels = self.lmap.shape[0]
    n_columns = self.lmap.shape[1]
    n_rows = self.lmap.shape[2]
    
    self.surf_state = None # too big too save on disk?
    self.bmusPlus = np.array(hf.get('bmusPlus_dataset'))
    
    self.som = somoclu.Somoclu(n_columns, n_rows, data=None, kerneltype=0, verbose=2, initialization = 'pca')
    
    self.som.bmus = self.bmusPlus[:,0:1]
    self.som.umatrix = np.array(hf.get('som_umatrix_dataset'))
    self.som.codebook = np.array(hf.get('som_codebook_dataset'))

    hf.close()

    
    return self

  def dpath_to(self, dpath):

    try:
      #np.savetxt(dpath + "\\" + "umatrix.txt", self.som.umatrix)
      #np.savetxt(dpath + "\\" + "codebook.txt", self.som.codebook.reshape((self.som._n_rows * self.som._n_columns, self.som.n_dim)) )
      hf = h5py.File(dpath + "\\" + self.fname + ".h5", 'w')
      hf.create_dataset('lmap_dataset', data=self.lmap)
      hf.create_dataset('bmusPlus_dataset', data=self.bmusPlus)
      hf.create_dataset('som_umatrix_dataset', data=self.som.umatrix)
      hf.create_dataset('som_codebook_dataset', data=self.som.codebook)
      hf.close()
    except Exception as e:
      raise

  #def train(self): #more
  #  pass