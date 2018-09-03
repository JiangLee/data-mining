#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 

import numpy as np
import pandas as pd
import datetime
import pytz
import optimize
import json
import matplotlib.pyplot as plt
from multiprocessing import Pool

def get_big_data(file,index='',chunkSize=100000):
	f = open(file)
	reader = pd.read_csv(f, sep=',', iterator=True)
	loop = True
	chunks = []
	while loop:
	    try:
	        chunk = reader.get_chunk(chunkSize)
	        chunks.append(chunk)
	    except StopIteration:
	        loop = False
	df = pd.concat(chunks, ignore_index=True)
	if index != '':
		df.set_index(index,inplace=True)
	df = df.dropna()
	return df

def time_trans(strtime,mod,orign_area,to_area):
	time_dict = {
		'China':'Asia/Shanghai',
		'HongKong':'Asia/Shanghai',
		'Thailand':'Asia/Bangkok',
		'Indonesia':'Asia/Bangkok',
		'Vietnam':'Asia/Bangkok',
		'India':'Asia/Calcutta',
		'United States':'America/Chicago',
		'Russia':'Europe/Moscow'
	}

	#将当前时间加上中国时区 东八区
	ntz = pytz.timezone(time_dict[orign_area])
	datetime.datetime.now(ntz)
	dt = datetime.datetime.strptime(strtime,'%Y-%m-%d %H:%M:%S')
	origin_time = ntz.localize(dt)

	#转为时间为指定国家的时区
	tz = pytz.timezone(time_dict[to_area])
	dt = origin_time.astimezone(tz)
	week = dt.weekday()
	hour = dt.strftime("%H")
	m = int(hour)/mod
	return dt.strftime("%Y%m%d")+str(m).zfill(2),week,m

def get_week_day(strtime):
	dt = datetime.datetime.strptime(strtime,'%Y-%m-%d')
	return dt.weekday()

def choose_avail_data(file):
	##y=a*x**b+c-->b
	coef = {}
	#y=a*x**b+c-->a
	intercept = {}
	#measure the effect
	square = {}
	#7days view count
	seven_view = {}
	with open(file, 'r') as f:
		for line in f:
			line = line.strip()
			video_data = json.loads(line)
			video_id = video_data['video_id']
			view_dict = video_data['axis_value']
			hour = []
			views = []
			for i in view_dict:
				hour.append(i['hour'])
				views.append(i['views'])
			X = np.array(hour)
			Y = np.array(views)
			try:
				M=np.log10(X)
				N=np.log10(Y)
				s=optimize.linear_fit(M,N)
				#s = [1,2,3]
				#if s[0] > 0 and s[0] < 1.0 and Y[-1:][0]>2000:
				coef[video_id] = s[0]
				intercept[video_id] = s[1]
				square[video_id] = s[2]
				seven_view[video_id] = Y[-1:][0]
				#print("Available:",video_id,s)
				#else:
					#print("Invalid:",video_id,s)
			except:
				pass
				#print("Except:",video_id,s)
	view_fit = {
		'coef':coef,
		'intercept':intercept,
		'square':square,
		'seven_view':seven_view
	}
	return view_fit

def get_sub_rate(file):
	with open(file, 'r') as f:
		rate = []
		for line in f:
			line = line.strip()
			video_data = json.loads(line)
			k = video_data['region']
			v = map(float,video_data['rate'].split(','))
			tmp=np.array(v)
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
			per_array = [0]
			for i in range(0,101):
				per_array.append(round(float(tmp[(tmp>0) & (tmp<(i+1))].size)/float(num)*100,4))
			start_per = round(float(tmp[(tmp>0) & (tmp<1)].size)/float(num)*100,4)
			end_per = round(float(tmp[(tmp>0) & (tmp<=100)].size)/float(num)*100,4)
			sub_region = {
					'min':min_sub,
					'max':max_sub,
					'pre':per_array,
					'start':start_per,
					'end':end_per,
					'avg':mu,
					'std':sigma
			}
			rate.append(sub_region)
		return rate

def get_standard_rate(rate_array,rate,sub_num):
	rate_dict = {}
	for sub_rate in rate_array:
		if sub_rate['max'] != None:
			if sub_rate['min'] <= sub_num and sub_rate['max'] > sub_num:
				rate_dict = sub_rate
				break
		else:
			if sub_num >= sub_rate['min']:
				rate_dict = sub_rate
				break
	#percent = rate_dict['pre'][-1]
	percent = 0.0
	for i in range(100,0,-1):
		if rate*100 >= i:
			percent = rate_dict['pre'][i]
			break
	#percent = round((rate*100-1)/(rate_dict['end']-rate_dict['start'])*100,2)
	return percent

def get_all_video_data(file):
	df = pd.DataFrame(columns=('channel', 'area', 'subscribe', 'category','publish_date','rate','video_category'))
	columns = ['channel', 'area', 'subscribe', 'category','publish_date','rate','video_category']
	num = 0
	channel_num = 0
	with open(file, 'r') as f:
		for line in f:
			line = line.strip()
			arr = json.loads(line)
			channel = arr['i']
			area = arr['a']
			subscribe = arr['s']
			category = arr['c']
			video = arr['f']
			#if subscribe<10000 or area == 'UNKNOWN':
			if subscribe<1000:
				continue
			for i in video:
				publish_date = i['p']
				rate = i['r']
				video_category = i['v']
				#if publish_date < '2018-06-01' or publish_date > '2018-08-01' or rate > 100 or rate < 0:
				#	continue
				df.loc[num] = [channel, area, subscribe, category,publish_date,rate,video_category]
				num = num + 1
			if channel_num%100 == 0:
				print(channel_num)
			channel_num = channel_num + 1
		print(channel_num)
	return df

def get_tmp_video_data(file):
	df = pd.DataFrame(columns=('channel', 'area', 'subscribe', 'category','publish_date','rate'))
	channel_num = 0
	num = 0
	with open(file, 'r') as f:
		for line in f:
			line = line.strip()
			arr = json.loads(line)
			channel = arr['channel_id']
			area = 'Thailand'
			subscribe = arr['sub_num']
			category = 1
			video = arr['rate_list']
			if subscribe<10000:
				continue
			for i in video:
				publish_date = i['pub_date']
				rate = i['rate']
				if publish_date < '2018-07-01' or publish_date > '2018-08-01' or rate > 100 or rate < 0:
					continue
				df.loc[num] = [channel, area, subscribe, category,publish_date,rate]
				num = num + 1
			print(channel_num,num)
			channel_num = channel_num + 1
	return df

##########################get video base info##############################
def multi_get_data(num,srcfile,destfile):
	createVar = locals()
	p = Pool(num)
	for i in range(num):
		createVar['result'+str(i)] = p.apply_async(get_all_video_data, \
						args=(srcfile+str(num+i)+'.txt',))
	p.close()
	p.join()

	for i in range(num):
		video_data_tmp = createVar.get('result'+str(i)).get()
		video_data_tmp.to_csv(destfile+str(num+i),\
									index=False,sep=',')
		video_data = video_data.append(video_data_tmp)
	return video_data