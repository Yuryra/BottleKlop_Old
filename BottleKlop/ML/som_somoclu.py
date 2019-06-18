
#https://peterwittek.com/somoclu-in-python.html

#google for 'what is "dense CPU kernel"'
#for how to save somoclu maps see https://github.com/peterwittek/somoclu/issues/35

import numpy as np
import matplotlib.pyplot as plt
#import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
import somoclu
from minisom import MiniSom  
from minisom import * # fast_norm(x)


import seaborn as sns

import globals # from this project
from Som_Classes import * # my classes

from scipy.spatial.distance import cdist
# i also need import sklearn for initialization = 'pca'
#%matplotlib inline  


My_Label_Colors = np.array(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'])#['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])


#from minisom!! - https://leandroagudelo189.github.io/Self-organizing-maps/
#def quantization_error(self, data):
#    """
#        Returns the quantization error computed as the average distance between
#        each input sample and its best matching unit.
#    """
#    error = 0
#    for x in data:
#        error += fast_norm(x-self.weights[self.winner(x)])
#    return error/len(data)


#def somoclu_plot_umatrix(umatrix, figsize=None, colormap=cm.Spectral_r,
#                     colorbar=False, bestmatches=False, bestmatchcolors=None,
#                     labels=None, zoom=None, filename=None):
#  #self.umatrix = np.zeros(n_columns * n_rows, dtype=np.float32)

#  n_columns = umatrix.shape[0]
#  n_rows = umatrix.shape[1]
#  #return self._view_matrix(umatrix, figsize, colormap, colorbar,
#  #                               bestmatches, bestmatchcolors, labels, zoom,
#  #                               filename)
#  if zoom is None:
#      zoom = ((0, self._n_rows), (0, self._n_columns))
#  if figsize is None:
#      figsize = (8, 8 / float(zoom[1][1] / zoom[0][1]))
#  fig = plt.figure(figsize=figsize)
#  if self._grid_type == "hexagonal":
#      offsets = _hexplot(matrix[zoom[0][0]:zoom[0][1],
#                                zoom[1][0]:zoom[1][1]], fig, colormap)
#      filtered_bmus = self._filter_array(self.bmus, zoom)
#      filtered_bmus[:, 0] = filtered_bmus[:, 0] - zoom[1][0]
#      filtered_bmus[:, 1] = filtered_bmus[:, 1] - zoom[0][0]
#      bmu_coords = np.zeros(filtered_bmus.shape)
#      for i, (row, col) in enumerate(filtered_bmus):
#          bmu_coords[i] = offsets[col * zoom[1][1] + row]
#  else:
#      plt.imshow(matrix[zoom[0][0]:zoom[0][1], zoom[1][0]:zoom[1][1]],
#                  aspect='auto', interpolation='bicubic')
#      plt.set_cmap(colormap)
#      bmu_coords = self._filter_array(self.bmus, zoom)
#      bmu_coords[:, 0] = bmu_coords[:, 0] - zoom[1][0]
#      bmu_coords[:, 1] = bmu_coords[:, 1] - zoom[0][0]
#  if colorbar:
#      cmap = cm.ScalarMappable(cmap=colormap)
#      cmap.set_array(matrix)
#      plt.colorbar(cmap, orientation='horizontal', shrink=0.5)

#  if bestmatches:
#      if bestmatchcolors is None:
#          if self.clusters is None:
#              colors = "white"
#          else:
#              colors = []
#              for bm in self.bmus:
#                  colors.append(self.clusters[bm[1], bm[0]])
#              colors = self._filter_array(colors, zoom)
#      else:
#          colors = self._filter_array(bestmatchcolors, zoom)
#      plt.scatter(bmu_coords[:, 0], bmu_coords[:, 1], c=colors)

#  if labels is not None:
#      for label, col, row in zip(self._filter_array(labels, zoom),
#                                  bmu_coords[:, 0], bmu_coords[:, 1]):
#          if label is not None:
#              plt.annotate(label, xy=(col, row), xytext=(10, -5),
#                            textcoords='offset points', ha='left',
#                            va='bottom',
#                            bbox=dict(boxstyle='round,pad=0.3',
#                                      fc='white', alpha=0.8))
#  plt.axis('off')
#  if filename is not None:
#      plt.savefig(filename)
#  else:
#      plt.show()
#  return plt

###############################################
def plt_test():
  # Fixing random state for reproducibility
  np.random.seed(19680801)
  N = 50
  x = np.random.rand(N) * 32
  y = np.random.rand(N) * 32
  colors = np.zeros(N)
  area = np.zeros(N)
  for i in range(colors.shape[0]):
    colors[i] = random.randint(0,8) 
    area[i] = (11 * random.uniform(0,1))**2  # 0 to 15 point radii

  plt.scatter(x, y, s=area, c=colors, alpha=0.5)
  plt.show()

def sig0to1(x) : #, x_min, x_max, x_scale):
  x_scale = 10
  x = x * x_scale
  x_min = -x_scale
  x = x - 4
  e = np.exp(x)
  return e / (e + 1)
  # 1.0 / (1 + 1.0 / np.exp(x * 10 - 1))

def lin0to1(x, y_min = 0, y_max = 1, x_min = 0, x_max = 1):
  if x < x_min: return y_min
  if x > x_max: return x_max
  y = y_max - (y_max - y_min) * (x_max - x) / (x_max - x_min)

  return 1.0 / (1 + 1.0 / np.exp(x * 10 - 5))

  return y

def labels_map_scatter(lmap, umatrix_row_by_col, title
                       , annotate_list = None):
  # Fixing random state for reproducibility

  n_labels = lmap.shape[0]
  n_columns = lmap.shape[1]
  n_rows = lmap.shape[2]

  label_color = range(n_labels)
  label_max = np.zeros(n_labels)
  label_sum = np.zeros(n_labels)
  for l in range(n_labels):
    label_max[l] = np.max(lmap[l])
    label_sum[l] = np.sum(lmap[l])
  label_max_cr = np.zeros((n_columns, n_rows),np.int)
  for c in range(n_columns):
      for r in range(n_rows):
        label_max_cr[c,r] = np.max(lmap[:, c, r])


  #plt.close() will close the figure window entirely, where plt.clf() will just clear the figure - you can still paint another plot onto it.

  #self.umatrix.shape = (self._n_rows, self._n_columns) !!!!!!!!
  #x = np.empty(0)
  #y = np.empty(0)
  #color = np.empty(0)
  #area = np.empty(0)
  x = []
  y = []
  color = []
  
  area = []
  a_scale = 25000
  for l in range(n_labels):
    for c in range(n_columns):
      for r in range(n_rows):
        v = lmap[l,c,r]
        if v > 0:
          x.append(c)
          y.append(r)
          color.append(l)
          a = a_scale * (v/label_sum[l])**2 #lin0to1((v/label_sum[l])**2,y_min = .00001) # even very small should be noticable
          area.append(a)
  #https://social.msdn.microsoft.com/Forums/en-US/fb4e86d4-8153-4862-8091-c8960f45daa5/visual-studio-2017-hover-tips?forum=visualstudiogeneral
  #https://kite.com/python/examples/4997/matplotlib-place-a-legend-outside-of-plot-axes
  #plt.legend(["blue", "green"], loc='center left', bbox_to_anchor=(legend_x, legend_y))
  
  if False:
    plt.scatter(x, y, s=area, c=color, alpha=0.6)
  else:
    fig, ax = plt.subplots()
    #cc = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
    #        'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
    #        'hot', 'afmhot', 'gist_heat', 'copper']
    #cmap=tuple(cc)
    #https://matplotlib.org/users/colormaps.html


    scatter = ax.scatter(x, y, s=area, c= color, alpha=0.6)#, cmap = cmap) #, cmap="Tab10") # - rather dark cmap
    #c= My_Label_Colors[color]

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes", bbox_to_anchor=(1,0))
    ax.add_artist(legend1)

    #Annotate with short string
    if annotate_list is not None:
      if annotate_list == "numbers":
        annotate_list = [str(i) for i in range(n_labels)] #https://djangostars.com/blog/list-comprehensions-and-generator-expressions/

      for l in range(n_labels):
        for c in range(n_columns):
          for r in range(n_rows):
            v = lmap[l,c,r]
            if v > 0:
              text = ax.annotate(annotate_list[l], (c,r), fontsize= 5)
              #opacity = lin0to1(v/label_max[l])

              if False:
                x = v/label_max[l] #from 0 to 1
              else:
                x = v/label_max_cr[c,r]
              #opacity = opacity ** 2 #sig0to1(opacity)

              opacity = 1.0 / (1 + 1.0 / np.exp(x * 10 - 5))
              text.set_alpha(opacity)
              # also trye https://kite.com/python/docs/matplotlib.text.Text.set_color




    #for i, txt in enumerate(n):
    #  ax.annotate("AZ", (x[i], y[i]))

  plt.imshow(umatrix_row_by_col, alpha=0.2, aspect='auto')
  plt.title(title)
  plt.show()

  xxx = 1

  
###############################################

#plt_test()
#exit()

def som_get_surface_state(self, data = None): #som_get_surface_state(self, data=None):
    """Return the Euclidean distance between codebook and data.

    :param data: Optional parameter to specify data, otherwise the
                  data used previously to train the SOM is used.
    :type data: 2D numpy.array of float32.

    :returns: The the dot product of the codebook and the data.
    :rtype: 2D numpy.array
    """

    if data is None:
        d = self._data
    else:
        d = data

    #see design note [surf_state flat formula] .. codebook goes row first, so reshape - packs row after row
    codebookReshaped = self.codebook.reshape(self.codebook.shape[0] * self.codebook.shape[1], self.codebook.shape[2])
    parts = np.array_split(d, 200, axis=0)
    am = np.empty((0, (self._n_columns * self._n_rows)), dtype="float64")
    for i, part in enumerate(parts):
        am = np.concatenate((am, (cdist((part), codebookReshaped, 'euclidean'))), axis=0)

    if data is None:
        self.activation_map = am
    return am



def quantization_error_from_minisom(win_weights, data):

    if win_weights.shape[0] != data.shape[0]:
      xxx = 1

    error = 0
    for i, x in enumerate(data):
        error += fast_norm(x-win_weights[i])
    return error/len(data)

def fast_norm_drom_minisom(x):
    """Returns norm-2 of a 1-D numpy array.

    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return sqrt(dot(x, x.T))




def labels_map_paint(lmap): # does NOT do plt.show()
  for l in range(lmap.shape[0]):
    plt.title('label:' + str(l))
    sns.heatmap(lmap[l])
    plt.show()
  # for heatmap annotation see https://stackoverflow.com/questions/33158075/custom-annotation-seaborn-heatmap


def labels_map_collect(self, labels, surf_state = None, bmus = None, data = None): #labels, of course, correspon to .. bmus
  #for topographical error see https://github.com/peterwittek/somoclu/issues/87

  som = self
  if surf_state is  None: surf_state = som_get_surface_state(som, data) #som.get_surface_state()
  if bmus is  None: bmus = som.get_bmus(surf_state)
    
  n_columns = self.codebook.shape[1] 
  n_rows = self.codebook.shape[0] 

  #labels.shape[0]
  n_labels = int(np.max(labels)) + 1
  lmap = np.zeros(( n_labels, n_columns, n_rows  ), dtype = int)
  for i, bmu in enumerate(bmus): #looping thru all data vectors
    label = int(labels[i])
      
    row = bmu[1]
    column = bmu[0]

    lmap[label, column, row] += 1

  return lmap #n_labels, n_columns, n_rows, lmap



def som_analyse(self, surf_state = None, bmus = None, data = None):
  #for topographical error see https://github.com/peterwittek/somoclu/issues/87

    som = self
    if surf_state is  None: surf_state = som_get_surface_state(som, data) #som.get_surface_state()
    if bmus is  None: bmus = som.get_bmus(surf_state)
    if data is None: data = som._data

    n_columns = self.codebook.shape[1] 
    n_rows = self.codebook.shape[0] 
    win_weights = np.zeros(bmus.shape[0])
    sum = 0
    for i, bmu in enumerate(bmus):
      row = bmu[1]
      column = bmu[0]

      #design note [surf_state flat formula] flat[c + cols * r]
      idx = column + n_columns * row # * r #column + n_columns * row

      win_weights[i] = surf_state[i, idx]

      w = self.codebook[row, column]
      d = data[i]
      n = fast_norm_drom_minisom(w - d)
      sum = sum + n

    sum = sum / data.shape[0]
    #eee = quantization_error_from_minisom(win_weights, som._data)
    eee = win_weights.mean()


    amin = surf_state.argmin(axis=1)

    rmin = surf_state.min(axis=1) 
  
    #sanity check:
    #s2 = surf_state.reshape(surf_state[0], n_columns, n_rows)
    #d = surf_state[:, amin[0]]
    data0 = data[0]
    codebook0 = som.codebook[bmus[0][1], bmus[0][0], :] #rows, columns
  
    quantization_error = np.mean(rmin) # / surf_state.shape[0]
    print(quantization_error)
    return quantization_error

def draw_data3(data3):
  n3 = int(data3.shape[0] / 3)
  #c1 = np.random.rand(50, 3)/5
  #c2 = (0.6, 0.1, 0.05) + np.random.rand(50, 3)/5
  #c3 = (0.4, 0.1, 0.7) + np.random.rand(50, 3)/5
  #data = np.float32(np.concatenate((c1, c2, c3)))
  colors = ["red"] * n3
  colors.extend(["green"] * n3)
  colors.extend(["blue"] * n3)
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.scatter(data3[:, 0], data3[:, 1], data3[:, 2], c=colors)
  labels = range(n3)
  plt.show()



def hhh(ntmData, fpath = None, noClsLabel = None):
    
  npNocls = numpy.load(fpath)
  n = npNocls.shape[0]
  txt = np.vstack(npNocls[0:n,0]).astype(np.float32)
  
  #y_index = 3
  labels = np.ones(n, dtype = np.int) * noClsLabel
  return txt, labels

def append_neutral(data, labels, nRows = 3000, randomScale=0.2, dim1 = 3, dim_labels = False):

  n = nRows

  

  c4 = np.ones(dim1)*0.5 + np.random.rand(n, dim1) * randomScale

  if data is None:
    data = c4
    labels = (np.ones(n) * 4-1).astype(np.int)
  else:
    data = np.float32(np.concatenate((data, c4)))
    labels = np.concatenate((labels, (np.ones(n) * 4-1).astype(np.int)) )

  return data, labels


def get_random_data3(nRows = 3000, randomScale=0.2, dim1 = 3, dim_labels = False):

  n = nRows



  c1 = np.random.rand(n, dim1) * randomScale
  c2 = np.ones(dim1)*0.2 + np.random.rand(n, dim1) * randomScale
  c3 = np.ones(dim1)*0.8 + np.random.rand(n, dim1) * randomScale

  #c2 = (0.6, 0.1, 0.05) + np.random.rand(n, dim1) * randomScale
  #c3 = (0.4, 0.1, 0.7) + np.random.rand(n, dim1) * randomScale

  data = np.float32(np.concatenate((c1, c2, c3)))
  #colors = ["red"] * n
  #colors.extend(["green"] * n)
  #colors.extend(["blue"] * n)
  #fig = plt.figure()
  #ax = Axes3D(fig)
  #ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)

  if dim_labels:
    labels = np.empty(0, dtype="int")
    #for i, part in enumerate(range(dim1)):
    #  labels = np.concatenate((labels, (np.ones(n) * i).astype(np.int)) )
    labels = np.concatenate((labels, (np.ones(n) * 0).astype(np.int)) )
    labels = np.concatenate((labels, (np.ones(n) * 1).astype(np.int)) )
    labels = np.concatenate((labels, (np.ones(n) * 2).astype(np.int)) )
  else: #uniquw labels
    labels = range(3 * n)
  #instead : 
  


  draw_data3(data)

  return data, labels 

import h5py
def somoclu_serialize(som, dpath, load=False):
  #https://github.com/peterwittek/somoclu/issues/35
  #https://www.christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html
  #for now i will try to use stupid original
  if load:
    som.load_umatrix(dpath + "\\" + "umatrix.txt")
    som.load_codebook(dpath + "\\" + "codebook.txt")

    hf = h5py.File(dpath + "\\" + "bmus.h5", 'r')
    bmus_dataset = hf.get('bmus_dataset')
    som.bmus = np.array(bmus_dataset)
    hf.close()

    x = 1
    #som.load_bmus(dpath + "\\" + "bmus.txt") 
    # TODO - bmus somehow screwed up - although its a pity to ocmpute them again
    # notice that som._data is NOT loaded

  else:
    np.savetxt(dpath + "\\" + "umatrix.txt", som.umatrix)
    np.savetxt(dpath + "\\" + "codebook.txt", som.codebook.reshape((som._n_rows * som._n_columns, som.n_dim)) )




    hf = h5py.File(dpath + "\\" + "bmus.h5", 'w')
    hf.create_dataset('bmus_dataset', data=som.bmus)
    hf.close()

    #np.savetxt(dpath + "\\" + "bmus.txt", som.bmus)

def displayVsData(som, data, labels, prefix = None, annotate_list = None):
  
  # measure data's dist to all cells
  surf_state = som_get_surface_state(som, data) #som.get_surface_state()
  # find best cells for the data
  bmus = som.get_bmus(surf_state)

  
  quantization_error = som_analyse(som,surf_state, bmus, data)

  #compute stats of labels fro each cell
  lmap = labels_map_collect(som, labels, surf_state, bmus, data)
  #display with backround as som's umatrix

  displayVsData_Inner(lmap, som.umatrix, quantization_error, prefix, annotate_list = annotate_list)
  #title = prefix + " " "quantization_error:" + str(quantization_error)
  #labels_map_scatter(lmap, som.umatrix, title)  
  #plt.title(title)

def displayVsData_Inner(lmap, umatrix, quantization_error, prefix = None, annotate_list = None):
 
  title = prefix + " " + "quantization_error:" + str(quantization_error)
  labels_map_scatter(lmap, umatrix, title, annotate_list = annotate_list)  
  #plt.title(title)

def minisom_progress(iteration, data = None, labels = None, epoch = None, minisom = None):
  if epoch % 10 == 0:
    if labels is None: labels = MinisomLabels
    minisom_display(som, data = data, labels = labels, prefix = "e:" + str(epoch))

  qe = minisom.quantization_error(data)
  print('%s, quant:%s', (epoch, qe))
  return True


def minisom_collectLmap(som, labels, bmus):


  self = som
  n_columns = self._weights.shape[0] 
  n_rows = self._weights.shape[1] 

  n_labels = int(np.max(labels)) + 1
  lmap = np.zeros(( n_labels, n_columns, n_rows  ), dtype = int)
  for i, bmu in enumerate(bmus): #looping thru all data vectors
    label = int(labels[i])
      
    row = int(bmu[0]) #bmu[1]
    column = int(bmu[1]) #bmu[0]

    lmap[label, column, row] += 1

  return lmap #n_labels, n_columns, n_rows, lmap

def minisom_display(som, data = None, labels = None, prefix = "no prefix"):
  bmus = np.zeros((data.shape[0],2))
  for i, d in enumerate(data):
    bmus[i] = som.winner(d)
  qe = som.quantization_error(data)
  
  #compute stats of labels fro each cell
  lmap = minisom_collectLmap(som, labels, bmus)
  #display with backround as som's umatrix

  displayVsData_Inner(lmap, som.distance_map(), qe, prefix)
  #title = prefix + " " + "quantization_error:" + str(qe)
  #labels_map_scatter(lmap, som.distance_map(), title)  
  #plt.title(title)



#================================================= Start ================================================
#what = 'minisom3'
#what = 'somoclu3'
SC_Dpath=  "c:/temp/ml/SC"
SD_Dpath=  "c:/temp/ml/SD"


what='real'
#what='exp2'
#what='exp3'
#what = 'minisom'
what = 'st2'



if what != 'test':
  import globals # from this project
  globals.Initialize()
  from NtmData2 import *
  Serialization_Dpath = globals.Ml_Space_Root
  ntmData = globals.global_Ntd2 #NtmData(set(['dprint']) )

  n_columns = 32
  n_rows = 32
  input_len = 150

  starting_index = 0
  bnTests = 6000
  rescaling = None #"aaa"
  


if False:
  crap = 1

elif what == "st2":
  ntmData = globals.global_Ntd2 #NtmData(set(['dprint']) )


  #max_label=9
  #dataNocls, labelsNocls = hhh(ntmData, fpath=r"C:\Ml_space\traindata\train_nocls10k.npy", noClsLabel = max_label + 1)

  n_columns = 32
  n_rows = 32
  nTests = 60 * 1000
  starting_index = 0

  train_from = 'geo' #'txt'
  train_to = 'st2' #'cls'
  train_from, train_to = ['txt', 'cls']
  annotate_list = St2List
  annotate_list = None


  #fname_prefix = "Dnm_vs_St2_"
  #fname_prefix = "Geo_vs_St2_"
  #fname_prefix = "Txt_vs_St2_"
  fname_prefix = train_from + "_vs_" + train_to
  fname = fname_prefix + "_" + str(n_columns) + "x" + str(n_rows) + "_" + str(int(starting_index / 1000)) + "_" + str(int(nTests / 1000)) + "k"
  display_name = "file : " + fname
  if False:



    W, labels = ntmData.loadVersus(num_examples = nTests, starting_index = starting_index, train_from = train_from, train_to = train_to)
    
    max_label = np.max(labels)
    
    

    dim1 = W.shape[1] # 150

    som = somoclu.Somoclu(n_columns, n_rows, data=None, kerneltype=0, verbose=2, initialization = 'pca')
      #somoclu_serialize(som, "c:\\temp\ml", load=True)

    nEpochs = 50
    som.train(epochs = nEpochs, data=W, yury_init = False)

    sc = Som_Lmaped.data_from(som, W, labels, fname = fname)
    sc.dpath_to(SC_Dpath)


    #som_analyse(som)
    displayVsData(som, W, labels, display_name, annotate_list = annotate_list)

    

  else:
    annotate_list = "numbers" 
    sc = Som_Lmaped.dpath_from(SC_Dpath, fname = fname)
    displayVsData_Inner(sc.lmap, sc.som.umatrix, 111, display_name, annotate_list)
      
  exit()


elif what == 'minisom':

  if True: # real stuff
    MinisomData, MinisomLabels = ntmData.loadSomWrap(num_examples = 60000, starting_index = 0, rescaling = rescaling)
  elif False: #random 
    dim1 = 150
    MinisomData, MinisomLabels = get_random_data3(nRows=3000, randomScale=0.4, dim1 = dim1, dim_labels = True)
  elif True:
    noClsLabel = 33
    MinisomData, MinisomLabels = hhh(ntmData, fpath=r"C:\Ml_space\traindata\train_nocls10k.npy", noClsLabel = noClsLabel)

  if True:
    som = somoclu.Somoclu(n_columns, n_rows, data=None, kerneltype=0, verbose=2, initialization = 'pca')
      #somoclu_serialize(som, "c:\\temp\ml", load=True)

    nEpochs = 50
    som.train(epochs = nEpochs, data=MinisomData, yury_init = True)
    som_analyse(som)
    displayVsData(som, MinisomData, MinisomLabels, "somoclu cmp with 'smp' data")
    exit()
  else:
    from minisom import MiniSom
  
    #W, labels = ntmData.loadSomWrap(num_examples = nTests, starting_index = starting_index)

    som = MiniSom(n_columns, n_rows, input_len, sigma=1.0, random_seed=1)
    #som.random_weights_init(W)
    nEpochs = 100
    som.train_batch(MinisomData, len(MinisomData)*nEpochs, eventFunction = minisom_progress)

elif what == 'exp3':
  #get saved & trained som
  #sc = Som_Lmaped.dpath_from(SC_Dpath)

  #get the data it was trained with

  if True:   
    nTests = 60000
    starting_index = 0
    fname = "train_data_0-60k"
    display_name = "Test Data"

    W, labels = ntmData.loadSomWrap(num_examples = nTests, starting_index = starting_index)
  else:
    nTests = 10000
    starting_index = 0
    fname = "nocls_data_0-10k"
    display_name = "No Cls"
    noClsLabel = 33
    W, labels = hhh(ntmData, fpath=r"C:\Ml_space\traindata\train_nocls10k.npy", noClsLabel = noClsLabel) #max_label + 1)
    pass
    

  max_label = np.max(labels)

  if True:

    som = somoclu.Somoclu(n_columns, n_rows, data=None, kerneltype=0, verbose=2, initialization = 'pca')
    #somoclu_serialize(som, "c:\\temp\ml", load=True)

    nEpochs = 10
    som.train(epochs = nEpochs, data=W)

    #som_analyse(som)
    displayVsData(som, W, labels, display_name)

    sc = Som_Lmaped.data_from(som, W, labels, fname = fname)
    sc.dpath_to(SC_Dpath)

  sc = Som_Lmaped.dpath_from(SC_Dpath, fname = fname)
  displayVsData_Inner(sc.lmap, sc.som.umatrix, 111, display_name)
  som = sc.som
  #som_analyse(som)
  som.train(epochs = 5, data = W)
  som_analyse(som)
  exit()

elif what == 'exp2':

  n_rows, n_columns = 32, 32
  #initialize with no data
  som = somoclu.Somoclu(n_columns, n_rows, data=None, kerneltype=0, verbose=2, initialization = 'pca')

  #load som
  somoclu_serialize(som, "c:\\temp\ml", load=True)

  sc = Som_Lmaped.dpath_from(SC_Dpath)

  if False:
    nTests = 10000
    starting_index = 70000
    W, labels = ntmData.loadSomWrap(num_examples = nTests, starting_index = starting_index)
    max_label = np.max(labels)
    sd_test = Som_Data.data_from(som, W, labels, fname = "testData", categoriser = globals.global_Categoriser)
    sd_test.dpath_to(SD_Dpath)
  else:
    sd_test = Som_Data.dpath_from(SD_Dpath, "testData")
    ppp1 = sd_test.cmp(sc.lmap)

  max_label = 6
  dataNocls, labelsNocls = hhh(ntmData, fpath=r"C:\Ml_space\traindata\train_nocls10k.npy", noClsLabel = max_label + 1)
  #displayVsData(som, dataNocls, labelsNocls, "pure nocls")

  #sd_noCls = Som_Data.data_from(som, dataNocls, labelsNocls, fname = "noCls", categoriser = globals.global_Categoriser)
  sd_noCls = Som_Data.dpath_from(SD_Dpath, "noCls")
  sd_noCls.dpath_to(SD_Dpath)

  ppp1 = sd_noCls.cmp(sc.lmap)


elif what == 'exp':
  sc = Som_Lmaped.dpath_from(SC_Dpath)

  sd_test = Som_Data.dpath_from(SD_Dpath, "testData")
  sd_test_restricted = Som_Data.restricted_from(sd_test, 10000)
  ppp1 = sd_test_restricted.cmp(sc.lmap) #,max_samples = 10000)
  ppp1 = sd_test.cmp(sc.lmap)
  
  x = 1

  sd_train = Som_Data.dpath_from(SD_Dpath, "no_name")
  ppp = sd_train.cmp(sc.lmap)
  x = 1
  sd_noCls = Som_Data.dpath_from(SD_Dpath, "noCls")
  ppp1 = sd_noCls.cmp(sc.lmap)
  x = 1



elif what == 'minisom3':

  n_rows, n_columns = 32, 32 #100, 160
  vectorSize = 3


  data, labels = get_random_data3(3000, dim1 = vectorSize)

  sigma= 1.0
  random_seed=1
  mini_som = MiniSom(n_rows, n_columns, vectorSize, sigma, random_seed)

  

  import time
  nEpochs = 30 # nEpochs == 1 cuases nan's
  for e in [2,5,10, 20, 40, 80]:
    
    start_time = time.time()

    nIterations = e * data.shape[0]
    mini_som.train_batch(data, nIterations, verbose=False, eventFunction = None)

    print(int(time.time() - start_time),"seconds on epochs")
    start_time = time.time()
    quantization_error = mini_som.quantization_error(data)
    print(quantization_error)
    print(int(time.time() - start_time),"seconds on quantization")
#0.3711362077960073


    sns.heatmap(mini_som.distance_map())

    plt.show()
    xxx = 1


elif what == 'somoclu3':

  n_rows, n_columns = 32, 32 #100, 160


  data, labels = get_random_data3(3000, dim1 = 3)


  
  som = somoclu.Somoclu(n_columns, n_rows, data=data, kerneltype=0, verbose=2,  initialization = 'pca')
  #%time som.train()


  sns.set()
  nEpochs = 30 # nEpochs == 1 cuases nan's
  for e in [2,5,10, 20, 40, 80]:
    
    som.train(epochs = e)
    som_analyse(som)

    #som.view_activation_map()
    #som.view_umatrix()

    sns.heatmap(som.umatrix)
    plt.show()
    
    som.view_umatrix(bestmatches=True) #, bestmatchcolors=colors, labels=labels)

    xxx = 1



elif what == 'test':
  n = 10000

  randomScale = 0.2

  c1 = np.random.rand(n, 3) * randomScale
  c2 = (0.6, 0.1, 0.05) + np.random.rand(n, 3) * randomScale
  c3 = (0.4, 0.1, 0.7) + np.random.rand(n, 3) * randomScale

  data = np.float32(np.concatenate((c1, c2, c3)))
  colors = ["red"] * n
  colors.extend(["green"] * n)
  colors.extend(["blue"] * n)
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)
  labels = range(3 * n)

  n_rows, n_columns = 32, 32 #100, 160
  som = somoclu.Somoclu(n_columns, n_rows, data=data, kerneltype=0, verbose=2,  initialization = 'pca')
  #%time som.train()

  nEpochs = 30 # nEpochs == 1 cuases nan's
  for i in range(100):
    
    som.train(epochs = nEpochs)
    som_analyse(som)
  #som.view_component_planes()

  som.view_umatrix(bestmatches=True, bestmatchcolors=colors, labels=labels)

  #few more times?
  nEpochs = 10
  som.train(epochs = nEpochs)
  som.view_umatrix(bestmatches=True, bestmatchcolors=colors, labels=labels)

  som.cluster()
  som.view_umatrix(bestmatches=True)

  xxx = 1


#elif what=='222': #https://pypi.org/project/somoclu/1.4/
  #data = np.loadtxt('rgbs.txt')
  #print(data)
  #data = np.float32(data)
  #nSomX = 50
  #nSomY = 50
  #nVectors = data.shape[0]
  #nDimensions = data.shape[1]
  #data1D = np.ndarray.flatten(data)
  #nEpoch = 10
  #radius0 = 0
  #radiusN = 0
  #radiusCooling = "linear"
  #scale0 = 0
  #scaleN = 0.01
  #scaleCooling = "linear"
  #kernelType = 0
  #mapType = "planar"
  #snapshots = 0
  #initialCodebookFilename = ''
  #codebook_size = nSomY * nSomX * nDimensions
  #codebook = np.zeros(codebook_size, dtype=np.float32)
  #globalBmus_size = int(nVectors * int(np.ceil(nVectors/nVectors))*2)
  #globalBmus = np.zeros(globalBmus_size, dtype=np.intc)
  #uMatrix_size = nSomX * nSomY
  #uMatrix = np.zeros(uMatrix_size, dtype=np.float32)
  #somoclu.trainWrapper(data1D, nEpoch, nSomX, nSomY,
  #                     nDimensions, nVectors,
  #                     radius0, radiusN,
  #                     radiusCooling, scale0, scaleN,
  #                     scaleCooling, snapshots,
  #                     kernelType, mapType,
  #                     initialCodebookFilename,
  #                     codebook, globalBmus, uMatrix)
  #print codebook
  #print globalBmus
  #print uMatrix

elif what=='real':

  import numpy as np
  import matplotlib.pyplot as plt

  import pickle

  

  #import NtmData2


  Serialization_Dpath = None
  #what="testwrite_nocls" #"ntm"
  what="ntm"
  what = 'testwrite'

  if False:
    pass

  elif what == "testwrite_nocls" or what == 'testwrite':
    import globals # from this project
    globals.Initialize()
    from NtmData2 import *

    Serialization_Dpath = globals.Ml_Space_Root


    ntmData = globals.global_Ntd2 #NtmData(set(['dprint']) )

    dbgSet = set(['dprint','fheadXXX']) 
    #ntmData2.
    if what == 'testwrite':
      dbgSet = set([  'dprint','fheadXXX']) 
      ntmData.testWrite(dbgSet)
    else:
      ntmData.testWrite_Nocls(dbgSet)


  elif what == "ntm":
    
    import globals # from this project
    globals.Initialize()
    from NtmData2 import *

    Serialization_Dpath = globals.Ml_Space_Root


    ntmData = globals.global_Ntd2 #NtmData(set(['dprint']) )


    #max_label=9
    #dataNocls, labelsNocls = hhh(ntmData, fpath=r"C:\Ml_space\traindata\train_nocls10k.npy", noClsLabel = max_label + 1)

    nTests = 60 * 1000
    starting_index = 70000
  
    #W, Y = ntmData.loadTextAndCls(num_examples = nTests, starting_index = starting_index)
    ##W = np.array(txt)
    ##Y = np.array(cls)
    ##Y = np.argmax(Y, axis=1)

    #labels = np.zeros(len(Y), dtype=int)
    #for i, label in enumerate(Y):
    #  labels[i] = np.argmax(label)

    #max_label = np.max(labels)

    #del Y
    ##del txt
    ##del cls

    #gc.collect()

    


    #W = np.vstack(W).astype(np.float32)

    W, labels = ntmData.loadSomWrap(num_examples = nTests, starting_index = starting_index)
    
    max_label = np.max(labels)

    dim1 = W.shape[1] # 150
    
    

  else:
    dim1 = 150

    W, labels = get_random_data3(nRows=3000, randomScale=0.4, dim1 = dim1, dim_labels = True)
    #add n..
    #W, labels = append_neutral(W, labels, nRows = 3000, randomScale=0.2, dim1 = dim1, dim_labels = False)


  n_rows, n_columns = 32, 32

  #initialize with no data
  som = somoclu.Somoclu(n_columns, n_rows, data=None, kerneltype=0, verbose=2, initialization = 'pca')
  
  
  if what == "ntm":
    #load som
    somoclu_serialize(som, "c:\\temp\ml", load=True)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++
    

    dataNocls, labelsNocls = hhh(ntmData, fpath=r"C:\Ml_space\traindata\train_nocls10k.npy", noClsLabel = max_label + 1)
    #displayVsData(som, dataNocls, labelsNocls, "pure nocls")
    sd_noCls = Som_Data.data_from(som, dataNocls, labelsNocls, fname = "noCls", categoriser = globals.global_Categoriser)
    sd_noCls.dpath_to(SD_Dpath)
    
    sc = Som_Lmaped.data_from(som, W, labels)
    sc.dpath_to(SC_Dpath)

    sd = Som_Data.data_from(som, W, labels, fname = "testData")
    sd.dpath_to(SD_Dpath)
    
    
    sd.cmp(sc.lmap)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++

    for i in range(max_label):
      ddd, lll = ntmData.select_by_label(data = W, labels = labels, good_labels = range(i))
      if len(ddd) > 0 :
        displayVsData(som, ddd, lll, "label " + str(i) )
    

    dataNocls, labelsNocls = hhh(ntmData, fpath=r"C:\Ml_space\traindata\train_nocls10k.npy", noClsLabel = max_label + 1)
    displayVsData(som, dataNocls, labelsNocls, "pure nocls")

  else:
    somoclu_serialize(som, "c:\\temp\ml_ntm", load=True)
    neutral_d,neutral_l = append_neutral(None, None, nRows = 3000, randomScale=0.2, dim1 = dim1, dim_labels = False)
    displayVsData(som, neutral_d, neutral_l, "pure neutral")


  nEpochs = 3
  som = somoclu.Somoclu(n_columns, n_rows, data=W, kerneltype=0, verbose=2, initialization = 'pca')

  #som.train(epochs = nEpochs)
  #som_analyse(som)

  dataName = "npmTxt" + "_" + str(W.shape[0]) + "x" + str(W.shape[1])
  mapName = "map_" + "c" + str(n_columns) + "x" + "r" + str(n_rows)

  for e in [50]:#[2,5,10, 20, 40, 80]:

    som = somoclu.Somoclu(n_columns, n_rows, data=W, kerneltype=0, verbose=2, initialization = 'pca')
    
    som.train(epochs = e)

    if True or e > 5:
      displayVsData(som, som._data, labels, prefix = "epoch:" + str(e) + "\n" + dataName + " " + mapName)

    ## now, after fast multicore do stupid slow -
    #surf_state = som_get_surface_state(som, som._data) #som.get_surface_state()
    #bmus = som.get_bmus(surf_state)

    

    #if True or e > 5:

    #  lmap = labels_map_collect(som, labels, surf_state, bmus, som._data)
    #  labels_map_scatter(lmap, som.umatrix)
    #  #labels_map_paint(lmap)
    #  plt.show()

      


    #som_analyse(som,surf_state, bmus)

 

    if False:
      som.view_activation_map(data_index = 0)
      #som.view_umatrix()
      sns.heatmap(som.umatrix)
      plt.title("epoch:" + str(e) + "\n" + dataName + " " + mapName)
      plt.show()

  #som.view_component_planes()

x = 1

#see how neutral data quantized visavi this som
if what == "ntm":
  neutral_d,neutral_l = append_neutral(None, None, nRows = 3000, randomScale=0.2, dim1 = dim1, dim_labels = False)
else:
  neutral_d,neutral_l = append_neutral(None, None, nRows = 3000, randomScale=0.2, dim1 = dim1, dim_labels = False)
displayVsData(som, neutral_d, neutral_l, "pure neutral")

#add neutral data to the one that som was trained on: 
d = som._data
l = labels
d,l = append_neutral(d, l, nRows = 3000, randomScale=0.2, dim1 = dim1, dim_labels = False)


displayVsData(som, d, l, "neutral_added")

#Ml_Space_Root

somoclu_serialize(som, "c:\\temp\ml", load=False)

som = somoclu.Somoclu(n_columns, n_rows, data=None, kerneltype=0, verbose=2, initialization = 'pca')
somoclu_serialize(som, "c:\\temp\ml", load=True)

displayVsData(som, d, l, "after serialization")