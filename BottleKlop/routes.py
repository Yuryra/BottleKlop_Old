"""
Routes and views for the bottle application.
"""

from bottle import route, view
from datetime import datetime

import gensim

from bottle import (post, request, response, route, run, )

import time #for fancy timing :
#https://stackoverflow.com/questions/3620943/measuring-elapsed-time-with-the-time-module/46544199

''' for xml : '''
import xml.etree.ElementTree as ET  #https://docs.python.org/3/library/xml.etree.elementtree.html#xml.etree.ElementTree.parse

import globals
globals.Initialize()

from Ntm2 import *
from NtmData2 import *


##Shared_Counter = 0

def handle_response():

  # notice port 1111 - it is the port of this project
  # e @ in front of the file - it makes curl read the file
  #C:\Users\yury>c:\curl\bin\curl.exe -X POST -d @c:\temp\indoc_sample1.xml http://localhost:1111/xml?name=mimi
  #c:\curl\bin\curl.exe -X POST -d @c:\temp\indoc_sample1.xml "http://localhost:1111/xml?op=cat&name=mimi"
  #c:\curl\bin\curl.exe -X POST -d @c:\temp\indoc_sample2.xml "http://localhost:1111/xml?op=cat&name=mimi"
  
  tsStart = time.perf_counter() #time()


  operation = request.query.op
  try:

    #global Shared_Counter
    #Shared_Counter += 1
    globals.Test_Counter +=1

    indocXml = request.body.read()
    ns = {'seaglex': "http://www.seaglex.com/ns",
                'geo': "http://www.w3.org/2003/01/geo/wgs84_pos#"}

    if operation == "cat":
      #r"c:\temp\indoc_sample1.xml"
      dbgSet = set()#['dclust', 'dprint'])
      fpath = r"C:\Users\yury\GitClonesHere\NtmData\FromProxy\sample_files_out\2017-02-27--2017-03-06.xml"

      fry = NtmReaderXml.fromXml(indocXml, dbgSet)
      rootOut = globals.global_Ntd2.getCategories(globals.global_Categoriser,fry)

      outXml = ET.tostring(rootOut,encoding='utf8', method='xml').decode() #et, )
      return outXml



    for i in range(1,20000):



      indocRoot = ET.fromstring(indocXml) #parses into element .. https://stackoverflow.com/questions/647071/python-xml-elementtree-from-a-string-source
      #indocRoot = indocTree.getroot()
      ''' modify ..'''



    
      snapList = list()
      for g in indocRoot.findall("channel/item/seaglex:snap", ns):
        snapList.append(g.text)

    tsEnd = time.perf_counter() #time()
    if len(snapList) == 0:
      outXml = "<root> "+ "no channel/item/seaglex:snap??" + "</root>"
    else:
      
      outXml = "<root> " \
        + "<op>" + str(operation) + "</op>" \
        + "<cnt>" + str(globals.Test_Counter) + "</cnt>" \
        + "<tsDelta>" + str(tsEnd - tsStart) + "</tsDelta>" \
        + "<snap>" + snapList[0] + "</snap>" \
        + "</root>"
    return outXml

  except Exception as e:
            #exc_type, exc_obj, exc_tb = sys.exc_info()
    print(e)
    raise

  return None



##log('yury: ROUTING')

@route('/xml', method="GET")
@route('/xml', method="POST")
def xml():

  '''
  Yes you can specify parameters via querystring to a resource you are posting to. Also you should use '@' before a filename if the data is being read from a file. So using your example I believe this should look like:
  curl -d @myfile.xml "http://www.example.dom/post/file/here?foo=bar&this=that"
  '''

  response.headers['Content-Type'] = 'text/xml'
  xml_content_whatever = handle_response() #"<root> saddasdada dsa asda das das </root>"
  return xml_content_whatever

#@post('/')
#def index():
#    postdata = request.body.read()
#    print(postdata) #this goes to log file only, not to client
#    name = request.forms.get("name")
#    surname = request.forms.get("surname")
#    return "Hi {name} {surname}".format(name=name, surname=surname)
###############################################
@route('/')
@route('/home')
@view('index')
def home():
    """Renders the home page."""
    return dict(
        year=datetime.now().year
    )

@route('/contact')
@view('contact')
def contact():
    """Renders the contact page."""
    return dict(
        title='Contact',
        message='Your contact page.',
        year=datetime.now().year
    )

@route('/about')
@view('about')
def about():
    """Renders the about page."""
    return dict(
        title='About',
        message='Your application description page.',
        year=datetime.now().year
    )
