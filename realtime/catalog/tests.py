from django.test import TestCase

# Create your tests here.
from django.shortcuts import render

# Create your views here.

#from .models import cpuutilInfo
from getdata import *
from util import *
import numpy as np

predict_data = np.array(18, np.int32)


now_data = np.zeros(18)
jsdata = get_data(url='http://api.cngrid.org/v2/show/cngrid/realtimeInfo')
#print("getjsdata: ",jsdata)
dict, status = from_json_to_dict(jsdata)

if status != 3:
    now_data, predict_data, status = getstatus(dict, predict_data)
    print("now_data",now_data)
    print("predict_data",predict_data)