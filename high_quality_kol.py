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

def get_full_high_data(high_quality_file_path):
	start_date = '2018-05-01'
	last_month_date = '2018-07-16'
	end_date = '2018-08-16'
	sub_limit = 5000
	rate_percent = 78
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
			chanel_base = pd.DataFrame()
			for ch in channel_tmp:
				total_video_count = pdt.loc[pdt['channel'] == ch].shape[0]
				high_quality_count = pdt_tar.loc[pdt_tar['channel'] == ch].shape[0]
				last_month_count = pdt.loc[(pdt['channel'] == ch) & (pdt['publish_date']>=last_month_date) & \
										(pdt['publish_date']<=end_date)].shape[0]
				row = pd.DataFrame([[ch, total_video_count, high_quality_count, last_month_count]], \
										columns = ['channel', 'total_video_count', 'high_quality_count', 'last_month_count'])
				chanel_base = chanel_base.append(row,ignore_index=True)
			pd_rm_same = pdt.drop_duplicates(subset=['channel'],keep='first')
			full_chanel = full_chanel.append(pd.merge(pd_rm_same,chanel_base,on='channel'),ignore_index=True)
			logger.info('completed file:' + list[i] + ", count:" + str(full_chanel.shape[0]))
		else:
			logger.error('file not exist:' + list[i])
	full_chanel.to_csv(high_quality_file_path,index=False,sep=',')
	return full_chanel

def filter_data():
	total_video_limit = 14
	last_month_limit = 4
	high_rate = 80
	sub_limit = 50000
	high_quality_file_path = data_path + "high_quality_video.txt"
	high_quality_channel = data_path + "high_quality_channel_portuguese.csv"
	if os.path.exists(high_quality_file_path):
		full_chanel = pandas_lib.get_big_data(high_quality_file_path,'')
	else:
		full_chanel = get_full_high_data(high_quality_file_path)
	full_chanel['high_rate'] = full_chanel.apply(lambda x: round(float(x.high_quality_count)/float(x.total_video_count),3)*100, axis = 1)
	final_chanel = full_chanel.loc[(full_chanel['high_rate']>=high_rate) & (full_chanel['total_video_count']>=total_video_limit) \
									& (full_chanel['last_month_count']>=last_month_limit) & (full_chanel['subscribe']>=sub_limit)]
	final_chanel['channel'] = final_chanel['channel'].map(lambda x: "https://www.youtube.com/channel/" + x)

	asia_country = ['Indonesia','Thailand','Vietnam']
	spanish_contry = ['Mexico','Argentina','Colombia','Spain','Chile','Peru','El Salvador',\
						'Ecuador','Costa Rica','Bolivia','Guatemala','Dominican Republic','Paraguay','Honduras']
	portuguese_country = ['Brazil','Portugal']
	final_chanel = final_chanel.loc[final_chanel["area"].isin(portuguese_country)]
	final_chanel.to_csv(high_quality_channel,index=False,sep=',')

if __name__ == '__main__':
	#high_quality_file_path = data_path + "high_quality_video.txt"
	#get_full_high_data(high_quality_file_path)
	filter_data()