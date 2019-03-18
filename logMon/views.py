import os
import json
from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
# Create your views here.

def index(request):
	return render(request,'logMonitor/logMon.html')


#获取相应时间的json数据，应该返回一个时间，这里先忽略不计
@csrf_exempt
def get_log_data(request):
	day = request.POST.get("day",None)
	time = request.POST.get("time",None)   #这个就是前端点击之后返回的时间
	hour = time[0:2]
	minute = time[3:4]

	if os.path.exists('static/log/'+ day +'/'+ hour +'-'+ minute + '.json'):
	#open函数打开文件的路径其实是以项目所在目录为根目录
		with open('static/log/'+ day + "/" + hour + "-" + minute + ".json") as log_f:
			log_root = json.load(log_f)
			user_info_root = log_root['USER_INFO']
			user = user_info_root['user']
			user_count = 0
			for i in user:
				user_count += i['count']

			user_op = user_info_root['op']
			user_op_count = 0
			for i in user_op:
				user_op_count += i['value']

			cluster_info_root = log_root['CLUSTER_INFO']
			cluster_count = 0
			for i in cluster_info_root:
				cluster_count += i['count']

			job_info_root = log_root['JOB_INFO']
			jobs = job_info_root['jobs']
			job_count = 0
			for i in jobs:
				job_count += i['count']

			return JsonResponse({
					'user_count':user_count,
					'user_op_count':user_op_count,
					'cluster_count':cluster_count,
					'job_count':job_count
				})
	else:
		print("file doesn't exists")
		return JsonResponse({
					'user_count':0,
					'user_op_count':0,
					'cluster_count':0,
					'job_count':0
				})

def mosaic_chart(request):
	return render(request,'charts/logMon/mosaic.html')

def cluster_info(request):
	return render(request,'charts/logMon/grid_op.html')

def user_info(request):
	return render(request,'charts/logMon/user_op.html')