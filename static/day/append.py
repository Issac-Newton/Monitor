import os

out = True
dirs = os.listdir('.')
for d in dirs:
	if os.path.isdir(d):
		rf = open(d + '/sum.json','r')
		lines = rf.readlines()
		last = lines[-1]
		last_s = list(last)
		last_s[-2] = ']'
		last = ''.join(last_s)
		lines[-1] = last

		wf = open(d + '/sum.json','w')
		for line in lines:
			wf.write(line)
