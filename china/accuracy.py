# -*- coding: utf-8 -*-


import pandas as pd
import multiprocessing
import numpy as np
from numpy import *

import csv
import time
import os
import math
from scipy.optimize import minimize
from scipy.optimize import Bounds

import logging
from datetime import datetime
import pickle


def simulation(network,initial,infect):
	out=[append(infect[0],initial)]
	for i in range(infect.shape[0]):
		ratio=dot(network,out[-1])/float(len(out[-1]))
		tmp = np.asarray(1+out[-1][infect.shape[1]:], dtype=float)
		tmp2 =  np.asarray( log(tmp) + ratio, dtype=float )
		ratio2 = exp(tmp2)-1
		ratio2[ratio2<0] = 0
		ratio2[isnan(ratio2)] = 0
		ratio2[isinf(ratio2)] = 0
		out.append(append(infect[i],ratio2))
	return array(out)

def accuracy(infer,true):
	y=mean(absolute(infer-true)/absolute(true+1))
	z=mean(absolute(infer-true)/absolute(true+1),axis=0)
	print y,z
	return y,z

	
	
	
if __name__=='__main__':
	incub=14
	incub1=7
	os_T=2
	##read and process infect/hiddent infect data
	infect_prob=[]
	recovery=[]
	infect_matrix=[]
	boom_prob=[]
	hidden_infect=[]
	raw_data=[]
	infect_prob=[]
	recovery=[]
	infect_matrix=[]
	boom_prob=[]
	hidden_infect=[]
	raw_data=[]
	forecast_data=[]
	data=pd.ExcelFile('result8.xlsx')
	for i in range(27):
		item ='ad_matrix_'+str(2*i)
		infect_matrix.append(array(data.parse(item))[1:,2:])
		item='prob_'+str(2*i)
		infect_prob.append(array(data.parse(item))[:,2])
		boom_prob.append(array(data.parse(item))[:,1])
		item='recovery_'+str(2*i)
		recovery.append(array(data.parse(item))[0,1])
		item='hidden_data_'+str(2*i)
		hd=data.parse(item)
		hidden_infect.append(array(hd)[1:,1:])
		item='forecast_'+str(2*i)
		forecast_data.append(data.parse(item))
		item='raw_data_'+str(2*i)
		raw_data.append(array(data.parse(item))[1:,1:])

	raw_data=raw_data[::-1]
	cities=array(data.parse('recovery_0'))[1:,0:1]

	infect_matrix=infect_matrix[::-1]
	infect_prob=infect_prob[::-1]
	boom_prob=boom_prob[::-1]
	recovery=recovery[::-1]
	hidden_infect=hidden_infect[::-1]

	popu=pd.ExcelFile('moving_average_population_flow.xlsx')
	# popu.index=popu['origin_fips/destination_fips']
	popu_data={it.replace('inter-flow_',''):array(popu.parse(it))[:,1:] for it in popu.sheet_names}
	date_index=['2020-01-0'+str(i) for i in range(1,10)]+['2020-01-'+str(i) for i in range(10,32)]
	date_index=date_index+['2020-02-0'+str(i) for i in range(1,10)]+['2020-02-'+str(i) for i in range(10,30)]
	date_index=date_index+['2020-03-0'+str(i) for i in range(1,8)]#+['2020-03-'+str(i) for i in range(10,14)]

	for i in range(26):
		if i==0:
			hidden_data=hidden_infect[i+1]
			report_data=raw_data[i+1]
		else:
			report_data=append(report_data,raw_data[i+1][:,-os_T:],axis=1)
			hidden_data=append(hidden_data,hidden_infect[i+1][:,-os_T:],axis=1)
	hidden=[]
	# hidden_data=hidden_data[1:] # 暂时注释掉，要不然后面会报错
	hidden=hidden_data[:,incub+1:]-hidden_data[:,incub:-1]
	forecast_hidden=[]
	h=hidden.shape[1]
	f=h

	infect_data=report_data[:,-f-1:].T

	media_list=[u'城市-日期-论坛.xlsx',u'城市-日期-网页.xlsx',u'城市-日期-微博.xlsx',u'城市-日期-微信.xlsx',u'城市-日期-客户端.xlsx',u'城市-日期-报刊.xlsx']
	for i in range(len(media_list)):
		print("i", i)
		df = pd.read_excel(media_list[i])
		df.rename(columns={df.columns[1]: "city"}, inplace=True)
		df.dropna(axis=0, subset=['city'], inplace=True)

		if i == 0:
			media_data = array(df)[:, 11:11 + f + 1]
			cities1 = array(df)[:, 1:2]
		else:
			media_data = media_data + array(df)[:, 11:11 + f + 1]

	media_data=media_data.T
	deleted = where(dot(ones(media_data.shape[0]),media_data)==0)
	media_data=np.delete(media_data, deleted, axis=1)
	cities1=np.delete(cities1, deleted, axis=0)
	cities=np.append(cities,cities1,axis=0)
	initial=media_data[0]
	
	network=pd.read_csv('to_file_E_china_07062321.csv',header=None)
	network=network[network.shape[0]-cities1.shape[0]:]
	paths=simulation(network,initial,infect_data)

	oa,accu=accuracy(paths[1:,infect_data.shape[1]:],media_data)
	accu=pd.DataFrame(array([cities1.flatten(),accu]).T,columns=['cities','accuracy'])
	accu.to_csv('accuracy.csv',index=False,encoding='utf-8')
	print("done")
	