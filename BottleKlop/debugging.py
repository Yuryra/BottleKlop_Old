
from __future__ import print_function
import sys

import mxnet as mx
from mxnet import nd

import xml.etree.ElementTree as ET

#import ml.mlp_scratch_class2 - failed

#from ml import mlp_scratch_class2 - failed




import globals
globals.Initialize()

from Ntm2 import *
from NtmData2 import *

#r"c:\temp\indoc_sample1.xml"
dbgSet = set()#['dclust', 'dprint'])
fpath = r"C:\Users\yury\GitClonesHere\NtmData\FromProxy\sample_files_out\2017-02-27--2017-03-06.xml"
fry = NtmReaderXml.fromFile(fpath, dbgSet)
fry.getItems()


treeOut = globals.global_Ntd2.getCategories(globals.global_Categoriser,fry)
exit()

sys.path.insert(0, './ml')
import mlp_scratch_class2 as MC

num_inputs = 150
num_outputs = 8
ms1 = MC.mlp_scratch(num_inputs, num_outputs) 
ms1.params_load()


rootOut = ET.Element("root")
doc = ET.SubElement(rootOut, "doc")

ET.SubElement(doc, "field1", name="blah").text = "some value1"
f2 = ET.SubElement(doc, "field2", name="asdfasd")
f2.text = "some vlaue2"
f2.set('status', 'completed')

#treeOut = ET.ElementTree(root)
#treeOut.write("filename.xml").treeOut.
x = ET.tostring(rootOut,encoding='utf8', method='xml').decode() #et, )

print(x)







ntd = NtmData()
ntd.loadVectors()

i = 0
for itemEl in fry.itemEls: # here i do it 1 by 1 for each item .. i could pack the in a matrix .. do net
              
  #if 'fhead' in dbgSet: 
  #    if i > 100: break # prints only 3 first items
  #i = i + 1

  item = NtmItemReader(itemEl)


  npDomain, npText, npCat = ntd.vectors_from_url_and_paragrpahs(item)

  ddd = mx.nd.array(npText.astype(numpy.float32)) #nd.Array(npText).astype(numpy.float32))
  label = mx.nd.array(npCat.astype(numpy.float32)) #nd.Array(npCat).astype(numpy.float32)
  label_hat = ms1.do1(ddd)
  print(label_hat)
  print(label)
  print(softmax1(label_hat))
  print(softmax1(label))

#def softmax(vector):
#    exp = nd.exp(vector-nd.max(vector))
#    return exp / nd.nansum(exp)

#for i, (data, label) in enumerate(test_data):
#  ddd = data[0].astype(numpy.float32)
#  label = label[0]
#  label_hat = ms1.do1(ddd)
#  print(label_hat)
#  print(label)
#  print(softmax1(label_hat))
#  print(softmax1(label))
#exit()     