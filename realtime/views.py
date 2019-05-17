from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt

# from .models import cpuutilInfo 先不用数据库
from .getdata import get_data
from .util import *
import numpy as np
import json
# Create your views here.


def realTime(request):
	return render(request,'realtime/realTime.html')

def allCenter(request):
	return render(request,'realtime/allcenter.html')



@csrf_exempt
def get_rt_data(request):
	name = request.POST.get("name",None)

	jsdata = get_data(url = 'http://api.cngrid.org/v2/show/cngrid/realtimeInfo')
	dict,status = from_json_to_dict(jsdata)

	if status != 3:
		for profile in dict:
			if profile['nodeName'] == name:
				return JsonResponse({
						'current':profile,
						'status' :status
					})

	return JsonResponse({
		'status':status
		})

@csrf_exempt
def get_rt_all_data(request):
	selector = request.POST.get('selector',None)
	if selector == None:
		print("selector error")
		return JsonResponse({
		'status':3
		})

	jsdata = get_data(url = 'http://api.cngrid.org/v2/show/cngrid/realtimeInfo')
	dict,status = from_json_to_dict(jsdata)

	#这里需要对数据做一个处理，需要的所有节点的数据
	all_data = {}
	if status != 3:
		for profile in dict:
			all_data[profile['nodeName']] = profile[selector]

		return JsonResponse({
				'all_data':all_data,
				'status' :status
			})

	return JsonResponse({
		'status':status
		})