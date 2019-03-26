from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt

# from .models import cpuutilInfo 先不用数据库
from .getdata import get_data
from .util import *
import numpy as np
import json
import warnings
# Create your views here.

warnings.filterwarnings("ignore")

predict_data = np.array(18, np.int32)
def bubble(request):
	return render(request,'realtime/bubble.html')

def get_rt_data(request):
	#print("get_data_status")
	now_data = np.zeros(18)
	jsdata = get_data(url = 'http://api.cngrid.org/v2/show/cngrid/realtimeInfo')
	##print(jsdata)
	dict,status = from_json_to_dict(jsdata)

	r_pre = ''
	r_now = ''
	if status != 3:
		#print("status ok")
		global predict_data
		now_data, predict_data,status = getstatus(dict,predict_data)
		#print("now_data:",now_data)
		#print("predict_data:",predict_data)
		#print("status:",status)
		r_pre = str(predict_data[0])  #网站中查看的只是casnw节点的实时数据
		r_now = str(now_data[0])
		'''
		先不需要数据库
		a_record = cpuutilInfo(casnw = dict['casnw'],dicp = dict['dicp'],era = dict['era'],erai = dict['erai'] , gspcc=dict['gspcc'], hku = dict['hku'],
								hust = dict['hust'], iapcm = dict['iapcm'], nscccs = dict['nscccs'], nsccgz = dict['nsccgz'],
								nsccjn = dict['nsccjn'], nscctj = dict['nscctj'], nsccwx = dict['nsccwx'], siat = dict['siat'],
								sjtu = dict['sjtu'], ssc = dict['ssc'], ustc = dict['ustc'], xjtu = dict['xjtu'], anomaly = status)


		#a_record.save()
	else:
		print("no data")
		a_record = cpuutilInfo(casnw= 0, dicp = 0, era= 0, erai=0, gspcc=0, hku=0,
								hust=0, iapcm=0, nscccs=0, nsccgz=0,
								nsccjn=0, nscctj=0, nsccwx=0, siat=0,
								sjtu=0, ssc=0, ustc=0, xjtu = 0, anomaly=status)

		#a_record.save()
		'''
	return JsonResponse({
						'status':status,
						'curr_usage' : r_now,
						'pred_usage' : r_pre
		})