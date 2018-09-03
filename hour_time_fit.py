#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 

import numpy as np
import pandas as pd
import pandas_lib
import matplotlib.pyplot as plt

##########################get video base info##############################
file = 'C:\Users\lijiang\KOL\py\data\DataSet.csv'
base_data = pandas_lib.get_big_data(file,'video_id')

########################count and choose avaliable data####################
same_file = 'C:\Users\lijiang\KOL\py\data\same_value.txt'
view_fit = pandas_lib.choose_avail_data(same_file)
fit=pd.DataFrame(view_fit)

#################join base data info and available data info################
fr = base_data.join(fit)
fr = fr.dropna()

############################get sub rate#####################################
sub_file = 'C:\Users\lijiang\KOL\py\data\sub_num_region.txt'
sub_rate = pandas_lib.get_sub_rate(sub_file)

#######count views/followers and merge time and transfer timezone###########
fr.eval("""
	percent0=seven_view/follower0
	percent1=seven_view/follower1
	avg_percent=seven_view*2/(follower0+follower1)""",inplace=True)

#hour cut
mod=6
fr['normal_time'] = fr.apply(lambda x: pandas_lib.time_trans(x.publish_date,mod,'HongKong',x.area)[0], axis = 1)
fr['week'] = fr.apply(lambda x: pandas_lib.time_trans(x.publish_date,mod,'HongKong',x.area)[1], axis = 1)
fr['hour'] = fr.apply(lambda x: pandas_lib.time_trans(x.publish_date,mod,'HongKong',x.area)[2], axis = 1)
fr['position'] = fr.apply(lambda x: pandas_lib.get_standard_rate(sub_rate,x.avg_percent,(x.follower0+x.follower1)/2), axis = 1)
fr = fr.dropna()
fr = fr.loc[(fr['avg_percent']>=0.01) & (fr['avg_percent']<=0.5)]


##############count last picture with country and category####################
country = ['Russia','Indonesia','United States','Thailand','India','Vietnam']
asia_country = ['Indonesia','Thailand','Vietnam']
#for area in country:
	#for ca in range(1,5):
		#tmp = fr.loc[(fr["area"] == area) & (fr["category"] == ca)]
fr = fr.loc[fr["area"].isin(asia_country)]
#fr = fr.loc[fr["area"]=='India']

num = 0
px = []
py = []
for week in range(0,7):
	for hour in range(0,int(24/mod)):
		tmp = fr.loc[(fr["week"] == week) & (fr["hour"] == hour)]
		#print(tmp)
		value = round(tmp['position'].mean(),4)
		#value = round(tmp['avg_percent'].mean(),4)
		print(week,hour,tmp.shape[0],value)
		#px.append("W"+str(week)+"["+str((hour+1)*mod)+"]")
		px.append(str(week+1)+"/"+str((hour+1)*mod))
		py.append(value)
		num = num + tmp.shape[0]

# plot raw data
ppx=np.array(px)
ppy=np.array(py)
plt.title("Raw data")
plt.scatter(ppx, ppy,  color='black')
plt.plot(ppx, ppy,  color='black')
plt.show()

print(num)
