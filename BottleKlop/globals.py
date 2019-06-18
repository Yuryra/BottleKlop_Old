#import mlp_scratch_class2 as MC
import sys
sys.path.insert(0, './ml')
from NtmData2 import *
#from configs import *  # just my file ...
    
def DirPrefix_findUp(start): #change on VS side /// 555 checking syncing with github
  i = 0
  while True:
    i += 1
    if i > 10 :
      print('did not find up: ' + start)
      ##raise
      return None
    dpath = os.path.realpath(start)
    if os.path.exists(dpath): break
    start = "..\\" + start
  return dpath + "\\"

def Initialize(): #https://instructobit.com/tutorial/108/How-to-share-global-variables-between-files-in-Python
  global Test_Counter 
  Test_Counter = 0

  #global ConfigVal 
  #ConfigVal = ConfigVal1
  #print(ConfigVal.paths.path1)

  global Ml_Space_Root
  Ml_Space_Root = DirPrefix_findUp("Ml_Space")

  #while True:
  #  dpath = os.path.realpath(Ml_Space_Root)
  #  if os.path.exists(dpath): break
  #  Ml_Space_Root = "..\\" + Ml_Space_Root
  #Ml_Space_Root =  dpath + "\\"

    

  #import  NtmData2 # ?????????????????????????????????????????????
  global global_Ntd2
  global_Ntd2 = NtmData(set())
  global_Ntd2.loadVectors()



  import mlp_scratch_class2 as MC

  global global_Categoriser
  num_inputs = 150
  num_outputs = 8
  try:
    global_Categoriser = MC.mlp_scratch(global_Ntd2.kvText_size \
           , len(global_Ntd2.catList) \
           , global_Ntd2.modelParamsSaved \
           ) #'num_inputs, num_outputs) 
    global_Categoriser.params_load()
  except Exception as e:
    print(e)
    raise

