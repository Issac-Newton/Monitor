file_name = 'sum.json'
kind = ['user','user_op','cluster','job']
with open('moment.txt','r') as f:
	count = 0
	last_day = ''
	lines = f.readlines()
	all_len = len(lines)
	for line in lines:
		line = line.replace('\n','')
		nums = line.split(' ')
		time = nums.pop(-1)  
		day = time[0:10]
		if day != last_day:
			last_day = day
			count = 0

		h_m = time[11:]
		w_f = open(day + '/' + file_name,'a')
		for i,n in enumerate(nums):
			if count ==  0 and i == 0:
				print('[',file=w_f)
			print('[%s,\"%s\",\"%s\"],' %(n,kind[i],h_m),file=w_f)
		count = count + 1
