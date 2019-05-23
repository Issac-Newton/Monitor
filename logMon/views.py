import os
import json
from lof import LOF
from lof import outliers
import json,calendar
import warnings
from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
# Create your views here.

warnings.filterwarnings("ignore")

def index(request):
	return render(request,'logMonitor/logMon.html')


#获取相应时间的json数据，应该返回一个时间，这里先忽略不计
def i_to_s(n):
	if n < 10:
		return '0' + str(n)
	else:
		return str(n)

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

@csrf_exempt
def get_dm_data(request):
	day = request.POST.get("day",None)
	month = day[0:7]
	monthRange = calendar.monthrange(int(day[0:4]),int(day[5:7]))

	d_user = []
	d_cluster = []
	d_op = []
	d_job = []
	m_user = []
	m_op = []
	m_cluster = []
	m_job = []

	day_file_list = os.listdir("static/day/" + day + "/")
	for i in range(0,24,1):
		name = str(i) + ".json"
		if name in day_file_list:
			with open("static/day/" + day + "/" + name) as f:
				data = json.load(f)
				d_user.append(data['user'])
				d_op.append(data['op'])
				d_cluster.append(data['cluster'])
				d_job.append(data['job'])
		else:
			d_user.append(0)
			d_op.append(0)
			d_cluster.append(0)
			d_job.append(0)

	month_file_list = os.listdir("static/month/")
	days = monthRange[1]
	for i in range(1, days+1, 1):
		name = day[0:8] + i_to_s(i) + ".json"
		if name in month_file_list:
			with open("static/month/" + name) as mf:
				data = json.load(mf)
				m_user.append(data['user'])
				m_op.append(data['op'])
				m_cluster.append(data['cluster'])
				m_job.append(data['job'])
		else:
			m_user.append(0)
			m_op.append(0)
			m_cluster.append(0)
			m_job.append(0)

	return JsonResponse({
			'duser':d_user,
			'dop':d_op,
			'dcluster':d_cluster,
			'djob':d_job,
			'mdays':days,
			'muser':m_user,
			'mop':m_op,
			'mcluster':m_cluster,
			'mjob':m_job
		})

def mosaic_chart(request):
	return render(request,'charts/logMon/mosaic.html')

def cluster_info(request):
	return render(request,'charts/logMon/grid_op.html')

def user_info(request):
	return render(request,'charts/logMon/user_op.html')


def log3D(request):
	return render(request,'logMonitor/log3D.html')

def json_to_list(lines):
	instances = []
	for line in lines:
		line.replace('\n','')
		line.replace('[','')
		line.replace(']','')
		parts = line.split(',')


@csrf_exempt
def Anomaly_Detect(request):
	time = request.POST.get('time',None)
	rf = open('static/day/' + time + '/sum.json','r')
	lines = json.load(rf)

	all_nums = []
	for line in lines:
		all_nums.append(line[0])

	instances = []
	for i in range(len(all_nums)):
		if i!=0 and i%4 == 0:
			instances.append((all_nums[i-4],all_nums[i-3],all_nums[i-2],all_nums[i-1]))

	M = -1
	instances.append((all_nums[-4],all_nums[-3],all_nums[-2],all_nums[-1]))
	for tu in instances:
		M = max(M,max(tu))

	exceptions = outliers(5,instances)  

	error_index = ''  #有异常的点的下标集合，用空格分隔
	for outlier in exceptions:    #每个outlier对象有一个value值可以限制一个点是不是离群值
		if outlier['lof'] > 1.23:
			error_index = error_index + ' ' + str(outlier['index'])

	return JsonResponse({
		'count_max':M,
		'index':error_index
		})