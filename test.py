#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import pandas_lib
import json
import scipy.stats as st
import matplotlib.mlab as mlab
from multiprocessing import Pool
import high_quality_kol
from config.config import *

high_quality_file_path = data_path + "high_test.txt"
high_quality_kol.get_full_high_data(high_quality_file_path)




##########################get video base info##############################
'''
def multi_get_data(num,r,srcfile,destfile):
	createVar = locals()
	p = Pool(num)
	for i in range(num):
		createVar['result'+str(i)] = p.apply_async(pandas_lib.get_all_video_data, \
						args=(srcfile+str(num*r+i)+'.txt',))
	p.close()
	p.join()

	for i in range(num):
		video_data_tmp = createVar.get('result'+str(i)).get()
		video_data_tmp.to_csv(destfile+str(num*r+i),\
									index=False,sep=',')
		#video_data = video_data.append(video_data_tmp)
	#return video_data



if __name__ == '__main__':
	mod = 37
	srcfile = 'C:\Users\lijiang\KOL\py\data\\ddd\\tmp1\\rate_full_'+str(mod)
	destfile = 'C:\Users\lijiang\KOL\py\data\\ddd\\video_tmp1\\video_tmp_'+str(mod)
	#for i in range(0,8):
	multi_get_data(4,0,srcfile,destfile)
'''
'''
createVar = locals()
if __name__ == '__main__':
	p = Pool(2)
	#for i in range(9):
	#    createVar['result'+str(i)] = p.apply_async(pandas_lib.get_tmp_video_data, args=('C:\Users\lijiang\KOL\py\data\\rate'+str(i)+'.txt',))
	result0 = p.apply_async(pandas_lib.get_all_video_data, args=('C:\Users\lijiang\KOL\py\data\\rate_full_0.txt',))
	result1 = p.apply_async(pandas_lib.get_all_video_data, args=('C:\Users\lijiang\KOL\py\data\\rate_full_1.txt',))
	print('Waiting for all subprocesses done...')
	p.close()
	p.join()
	print('All subprocesses done.')
	v = result0.get().append(result1.get())
	print(v)
	exit
'''
'''
for tnum in range(14,38):
	num = 0
	mod = 2500
	createVar = locals()
	for i in range(4):
		createVar['f'+str(i)] = open('C:\Users\lijiang\KOL\py\data\\ddd\\tmp1\\rate_full_'+str(tnum)+str(i)+'.txt', 'a')

	num = 0
	with open('C:\Users\lijiang\KOL\py\data\ddd\\tmp1\\rate_full_'+str(tnum)+'.txt', 'r') as f:
		for line in f:
			fnum = num/mod
			createVar.get('f'+str(fnum)).write(line)
			num = num + 1
'''
'''
def func(x, a, b, c):
#    return a * np.exp(-b * x) + c
    return a * pow(x,b) + c
 
xdata = np.linspace(0, 150, 50)
y = func(xdata, 1.5, 0.2, 0.5)
ydata = y + 0.2 * np.random.normal(size=len(xdata))
plt.plot(xdata,ydata,'b-')
popt, pcov = curve_fit(func, xdata, ydata)
#popt数组中，三个值分别是待求参数a,b,c
y2 = [func(i, popt[0],popt[1],popt[2]) for i in xdata]
plt.plot(xdata,y2,'r--')
print popt
plt.show()


#print(pandas_lib.time_trans("2018-7-20  0:16:14",2,'HongKong','United States'))


with open('C:\Users\lijiang\KOL\py\data\sub_num_region.txt', 'r') as f:
	rate = []
	for line in f:
		line = line.strip()
		video_data = json.loads(line)
		k = video_data['region']
		v = map(float,video_data['rate'].split(','))

		#for k,v in video_data.items():
		tmp=np.array(v)
		#print(tmp)
		tmp = tmp[tmp<100]
		mu = tmp.mean()
		sigma = tmp.std()
		num = tmp.size
		if k == "1000000+":
			min_sub = 1000000
			max_sub = None
		else:
			min_sub = int(k.split('-')[0])
			max_sub = int(k.split('-')[1])

		start_per = round(float(tmp[(tmp>0) & (tmp<1)].size)/float(num)*100,4)
		end_per = round(float(tmp[(tmp>0) & (tmp<50)].size)/float(num)*100,4)
		per_array = [0]
		for i in range(0,101):
			per_array.append(round(float(tmp[(tmp>0) & (tmp<(i+1))].size)/float(num)*100,4))
		sub_region = {
				'min':min_sub,
				'max':max_sub,
				'start':start_per,
				'end':end_per,
				'avg':mu,
				'std':sigma
		}
		print(min_sub,max_sub,tmp.shape[0])
		#print(sub_region)


		fw = open('C:\Users\lijiang\KOL\py\data\\percent\\'+str(min_sub)+'-'+str(max_sub)+'.txt', 'a')
		for i in range(0,101):
			wl = str(i)+'%, '+str(per_array[i])+'%'+'\n'
			fw.write(wl)
			#print(min_sub,max_sub,i,per_array[i])
		fw.close()
		rate.append(sub_region)

		#count, bins, _ = plt.hist(tmp, 0, normed=True)
		#plt.plot(bins,1./np.sqrt(2*np.pi)*sigma)*np.exp(-(bins-mu)**2/(2*sigma**2),lw=2,c='r')
		s_fit = np.linspace(tmp.min(), tmp.max())
		#plt.plot(s_fit, st.norm(mu, sigma).pdf(s_fit), lw=2, c='r')
		#plt.show()

		num_bins = 50
		n, bins, patches = plt.hist(tmp, num_bins, normed=1, facecolor='blue', alpha=0.5)  
		# add a 'best fit' line  
		y = mlab.normpdf(bins, mu, sigma)  
		#plt.plot(bins, y, 'r--')  
		plt.xlabel('Video measure index[views/fans]')  
		plt.ylabel('Index percent')
		  
		# Tweak spacing to prevent clipping of ylabel  
		plt.subplots_adjust(left=0.15)  
		plt.savefig('C:\Users\lijiang\KOL\py\data\pic\\'+str(min_sub)+'-'+str(max_sub)+ '.png')
		plt.cla()
		#plt.show()
		'''