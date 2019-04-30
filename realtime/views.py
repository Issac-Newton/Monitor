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