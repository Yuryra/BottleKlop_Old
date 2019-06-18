from minisom import MiniSom  
from minisom import * # fast_norm(x)
import numpy as np
import time
import gc # #https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python


class MyMiniSom():
  def __init__(self, map_dim, data, labels, sigma= 1.0, random_seed=1):
    try:

      #if data == None: return

      self.map_dim_x = map_dim
      self.map_dim_y = map_dim

      self.data = data
      self.vectorSize = data.shape[0]

      self.labels = labels

      self.nTotalIterations = 0

      self.som = MiniSom(self.map_dim_x, self.map_dim_y, vectorSize, sigma, random_seed)
      #som.pca_weights_init(self.data)
      #som.random_weights_init()  - it does random anyway

      self.prev_weights = self.som.get_weights()

      xxx = fast_norm(self.data[0])
      xxx = 1

      

    except Exception as e:
      raise

  def fPathOf(self):
    return "c:/ml_space/minisom"+"_m" \
      + str(self.map_dim_x) + "x" + str(self.map_dim_y) \
      + "_t" + str(len(self.data)) + ".pkl"

  def pickleFrom(self):
    with open(self.fPathOf(), "rb") as input_file:
      return pickle.load(input_file)

  def pickleTo(self):
    with open(self.fPathOf(), "wb") as output_file:
      return pickle.dump(self, output_file)

  def eventFunctionPrint(self, i):
    print("iteration:", i)

    if self.log_limit == None: 
      self.log_limit = i
    else:
      self.log_limit = self.log_limit * 1.666
      if  i < self.log_limit: return True
    

    self.pickleTo()

    self.plot1(i)
    return True

  def plot1(self, nIteration): 
    nX = self.map_dim_x
    nY = self.map_dim_y

    plt.figure(figsize=(nX - 1, nY - 1))
  # Plotting the response for each pattern in the iris dataset
    plt.pcolor(self.som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
    plt.colorbar()
    #plt.show()

    
    # use different colors and markers for each label
    markers = ['o', 's', 'D', '*','h','+', '1', '2']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i, xx in enumerate(self.data):
        w = self.som.winner(xx)  # getting the winner
        # palce a marker on the winning position for the sample xx
        label = int(self.labels[i])
        plt.plot(w[0]+.5, w[1]+.5
                 , markers[label], markerfacecolor='None',
                 markeredgecolor=colors[label], markersize=12, markeredgewidth=2)
    plt.axis([0, nX - 1, 0, nY - 1])
    #plt.savefig('resulting_images/som_iris.png')

    ttt = time.time()
    deltaSecs = ttt - self.prev_time
    self.prev_time = ttt

    plt.title('n: ' + str(nIteration) + ", dt: " + str(deltaSecs), fontsize=16)
    plt.show() #block=False)
    success = True
    return success

  def train_batch(self, nIterations = None, eventFunction = None):
    self.prev_time = time.time()
    self.log_limit = None
    eventFunction = self.eventFunctionPrint

    if nIterations == None : nIterations = 100 * len(self.data)
    start_time = time.time()
    self.som.train_batch(self.data, nIterations, verbose=False, eventFunction = eventFunction)

    self.nTotalIterations += nIterations

    norm_sum = 0
    for i in range(0, self.map_dim_x - 1):
      for j in range(0, self.map_dim_y - 1):
        vdelta = np.subtract(self.prev_weights[i,j],self.som.get_weights()[i,j])
        norm_sum += fast_norm(vdelta)
    norm_sum = norm_sum / (self.map_dim_x * self.map_dim_y)

    return norm_sum, time.time() - start_time

  #def iterate_batch(self, nTimes):
  #  for i in range(nTimes):



#####################################################################
######################################################################



#from somoclu import *
import numpy as np
import matplotlib.pyplot as plt

import pickle

#import NtmData2


import globals # from this project
globals.Initialize()

#######################################################3

#xxx = MyMiniSom(32, None, None)

from NtmData2 import *
ntmData = globals.global_Ntd2 #NtmData(set(['dprint']) )
nTests = 10000
W, Y = ntmData.loadTextAndCls(num_examples = nTests)
#W = np.array(txt)
#Y = np.array(cls)
#Y = np.argmax(Y, axis=1)

labels = np.zeros(len(Y))
for i, label in enumerate(Y):
  labels[i] = np.argmax(label)

del Y
#del txt
#del cls

gc.collect()

vectorSize = 150

msom = MyMiniSom(32, W, labels)

#feeble:
for iteration in range(100):
  norm, delta = msom.train_batch()
  print(norm)
  print(int(delta), " seconds")
msom.pickleTo()


#############################################################

