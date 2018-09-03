#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 

import pandas_lib
from config.config import *
import logging
import logging.handlers
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.handlers.TimedRotatingFileHandler(logFilePath, 'd', interval = 1, backupCount=7)
formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_data():
	if os.path.exists(full_file):
		fr = pandas_lib.get_big_data(full_file,'')
	else:
		#first call to generate file video_data
		if os.path.exists(video_file):
			video_data = pandas_lib.get_big_data(video_file,'')
		else:
			sys.exit()
		#	video_data = multi_get_data(24)
		#	video_data.to_csv(video_file,index=False,sep=',')
		fr = video_data.dropna()

		############################get sub rate#####################################
		sub_rate = pandas_lib.get_sub_rate(sub_file)

		#########################count week and rate position########################
		fr['week'] = fr.apply(lambda x: pandas_lib.get_week_day(x.publish_date), axis = 1)
		fr['position'] = fr.apply(lambda x: pandas_lib.get_standard_rate(sub_rate,round(x.rate/100,5),x.subscribe), axis = 1)
		fr = fr.dropna()
		
		fr.to_csv(full_file,index=False,sep=',')
	return fr

def count_and_draw(fr):
	#all country list
	area_arr=fr.drop_duplicates(['area'])['area'].values
	#all category list
	category_arr=fr.drop_duplicates(['video_category'])['video_category'].values

	fr = fr.loc[(fr['rate']>=1) & (fr['rate']<=80)]

	##############count last picture with country and category####################
	#asia_country = ['Indonesia','Thailand','Vietnam','Malaysia']
	#fr = fr.loc[fr["area"].isin(asia_country)]
	area_arr = ['United States','India','Russia','Japan','South Korea']
	category_arr = ['Gaming','Science & Technology','Sports','Howto & Style','Shows',\
					'Music','Entertainment','Pets & Animals','Education',\
					'Film & Animation','News & Politics','Movies','Comedy'\
					'People & Blogs','Film & Animation','Autos & Vehicles']
	for area in area_arr:
		#for ca in category_arr:
			#fr_tmp = fr.loc[(fr["area"] == area) & (fr["video_category"] == ca)]
		fr_tmp = fr.loc[(fr["area"] == area)]
		#fr_tmp = fr.loc[(fr["video_category"] == ca)]
		num = 0
		px = []
		py = []
		for week in range(0,7):
			tmp = fr_tmp.loc[(fr_tmp['week'] == week)]
			value = round(tmp['position'].mean(),4)
			logger.info('area:' + area + ', week:' + str(week) + ', number:' + str(tmp.shape[0]) + ', percent:' + str(value))
			px.append('Week:' + str(week+1))
			py.append(value)
			num = num + tmp.shape[0]

		# plot raw data
		ppx=np.array(px)
		ppy=np.array(py)
		#plt.title("Week Time Count: " + area + "/" + ca + "(" + str(num) + ")")
		plt.title("Week Time Count: " + area + "/all(" + str(num) + ")")
		#plt.title("Week Time Count: Southeast Asia/" + ca + "(" + str(num) + ")")
		plt.xlabel('Week day')  
		plt.ylabel('Video measure index[views/fans] percent')
		plt.scatter(ppx, ppy,  color='black')
		plt.plot(ppx, ppy,  color='black')
		#plt.savefig(pic_path + area + "--"+ ca + ".png")
		plt.savefig(pic_path + area + "--all" + ".png")
		#plt.savefig(pic_path + "Southeast Asia--" + ca + ".png")
		plt.cla()
		#plt.show()
		logger.info('total video number:' + str(num))

def filter_data():
	start_date = '2018-05-01'
	end_date = '2018-08-16'
	sub_limit = 5000
	rate_percent = 78
	total_video_limit = 14
	last_month_limit = 4
	full_chanel = pd.DataFrame()
	sub_rate = pandas_lib.get_sub_rate(sub_file)
	list = os.listdir(video_src_path)
	for i in range(0,len(list)):
		path = os.path.join(video_src_path,list[i])
		if os.path.isfile(path):
			pdt = pandas_lib.get_big_data(path)
			pdt = pdt.loc[(pdt['publish_date']>=start_date) & (pdt['publish_date']<=end_date) \
						   & (pdt['subscribe']>=sub_limit)]
			channel_tmp = pdt['channel'].unique()
			pdt['percent'] = pdt.apply(lambda x: pandas_lib.get_standard_rate(sub_rate,round(x.rate/100,5),x.subscribe), axis = 1)
			pdt = pdt.dropna()
			pdt_tar = pdt.loc[pdt['percent']>=rate_percent]
			chanel_base = pd.DataFrame(columns=('channel', 'total_video_count', 'high_quality_count'))
			for ch in channel_tmp:
				pdt_ch = pdt.loc[pdt['channel'] == ch]
				row = pd.DataFrame([ch, pdt_ch.shape[0], pdt_tar.shape[0]], ['channel', 'total_video_count', 'high_quality_count'])
				chanel_base.append(row,ignore_index=True)
			pd_rm_same = pdt.drop_duplicates(subset=['channel'],keep='first')
			full_chanel = full_chanel.append(pd.merge(pd_rm_same,chanel_base,on='channel'))
			print(full_chanel)
			sys.exit(0)


if __name__ == '__main__':
	#fr = get_data()
	#count_and_draw(fr)
	filter_data()