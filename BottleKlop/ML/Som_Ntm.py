

from minisom import MiniSom  
import bs4


import numpy as np

#https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python





#row0 = np.array([0., 0., 1., 0., 0., 0., 0., 0.], dtype=object)
#row1 = np.array([0., 0., 0., 1., 0., 0., 0., 0.], dtype=object)
#rows = np.array([row0,row1])
#xxxxxx = np.argmax(rows, axis=1)


import matplotlib.pyplot as plt

import pickle

#import NtmData2


import globals # from this project
globals.Initialize()

#######################################################3
import gc # to clear memory

from NtmData2 import *
ntmData = globals.global_Ntd2 #NtmData(set(['dprint']) )
nTests = 1000
txt, cls = ntmData.loadTextAndCls(num_examples = nTests)
W = np.array(txt)
Y = np.array(cls)
#Y = np.argmax(Y, axis=1)

del txt
del cls
gc.collect()

labels = np.zeros(len(Y), dtype=int)
for i, label in enumerate(Y):
  labels[i] = np.argmax(label)

del Y
gc.collect()

vectorSize = 150
#############################################################



# Running minisom and visualizing the result
# ----
# ***Warning***: This may take a while.

# In[16]:


from minisom import MiniSom
map_dim = 32 #16
nIterations = len(W)*50

fpath = "c:/ml_space/minisom"+"_m" + str(map_dim) + "_t" + str(len(W)) + "_i" + str(nIterations) + ".pkl" # + vectorsize etc
if False:
  som = MiniSom(map_dim, map_dim, vectorSize, sigma=1.0, random_seed=1)
  #som.random_weights_init(W) - it does it anyway
  som.train_batch(W, nIterations, verbose=False)


  #filehandler = open(fpath,"wb")
  with open(fpath, "wb") as output_file:
    pickle.dump(som, output_file)

#filehandler = open(fpath,"rb")
with open(fpath, "rb") as input_file:
  som = pickle.load(input_file)

#TRAIN MORE
som.train_batch(W, len(W)*10, verbose=False)

# 
import time
start_time = time.time()
act_res = som.activation_response(W)
winner_map = som.win_map(W)
labelmap  = som.labels_map(W,labels)
end_time = time.time() - start_time
print(int(end_time),"seconds taken to extract data from results.")


def plot1(som, map_dim, W, labels ):

  plt.figure(figsize=(map_dim - 1, map_dim - 1))
# Plotting the response for each pattern in the iris dataset
  plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
  plt.colorbar()
  plt.show()

  # use different colors and markers for each label
  markers = ['o', 's', 'D', '*','h','+', '1', '2']
  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
  for i, xx in enumerate(W):
      w = som.winner(xx)  # getting the winner
      # palce a marker on the winning position for the sample xx
      plt.plot(w[0]+.5, w[1]+.5
               , markers[labels[i]], markerfacecolor='None',
               markeredgecolor=colors[labels[i]], markersize=12, markeredgewidth=2)
  plt.axis([0, map_dim - 1, 0, map_dim - 1])
  #plt.savefig('resulting_images/som_iris.png')
  plt.show()

plot1(som, map_dim, W, labels)

#https://seaborn.pydata.org/generated/seaborn.heatmap.html
import seaborn as sns
sns.heatmap(act_res)
plt.show()
xxx = 1









###plt.plot()
plt.show()

x = 1
# In[ ]:



