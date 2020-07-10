# -*- coding: utf-8 -*-

"""


第二种修改方式
"""


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
import cvxpy as cp
import traceback



def explicit_minimizer(args):
    # print("len(args):",len(args))
    try:
        if len(args) == 3:
            (D, y, pshape) = args

            # y = np.dot(x,x)
            x_val = None
            scale = 2
            while x_val is None and scale < 16:
                x = cp.Variable(D.shape[1])
                objective = cp.Minimize(cp.sum(cp.abs(x)) / (float(D.shape[1]) ** scale))

                constraints = [D * x - y.flatten() == 0]

                prob = cp.Problem(objective, constraints)
                result = prob.solve(verbose=False, solver='OSQP')
                x_val = x.value

                scale += 0.1

            if x_val is None:
                print("x.value is None. scale:" + str(scale))
            # print 'optimal val :', result
            # return np.append(np.zeros(pshape), x.value)  # .reshape(self.pshape+D.shape[1],1)
            return x.value
        else:
            (D, y, params, K, cshape, pshape) = args
            a = np.append(np.diag(np.ones(pshape)), np.zeros((pshape, D.shape[1]-pshape)), axis=1)
            for i in range(1, K):
                b = np.zeros((1, D.shape[1]))
                b[0, pshape + (i - 1) * cshape:pshape + (i) * cshape] = 1
                # print("a.shape, b.shape:", a.shape, b.shape)
                a = np.append(a, b, axis=0)

            scale = 2
            x_val = None
            while x_val is None and scale < 16:
                x = cp.Variable(D.shape[1])
                objective = cp.Minimize(cp.sum(cp.abs(x)) / (float(D.shape[1]) ** scale))
                # /float(D.shape[1])+cp.sum_square(a[:self.pshape]*x-params[:self.pshape])/float(self.pshape)+cp.sum_square(a[self.pshape:]*x-params[self.pshape:])/float(K-1))
                constraints = [D * x - y.flatten() == 0]

                constraints = constraints + [a * x - params.flatten() == 0]
                prob = cp.Problem(objective, constraints)

                result = prob.solve( solver='OSQP' )
                x_val = x.value

                scale += 0.1
            if x_val is None:
                print("x.value is None. scale:" + str(scale))
            # print 'optimal val:', result
            return x.value  # .reshape(D.shape[1],1)
    except Exception as e:
        print("???", str(e))
        print 'str(e):', str(e)
        print 'repr(e):\t', repr(e)
        print 'e.message:\t', e.message
        print 'traceback.print_exc():', traceback.print_exc()
        print 'traceback.format_exc():\n%s' % traceback.format_exc()
        print("D.shape, y.shape", D.shape, y.shape)
        exit()



class MiningHiddenLink:
    def __init__(self, save_path, method_inverse=True, fixed_input=None,non_nega_cons=[], all_non_nega_cons=False):
        self.save_path = save_path
        self.method_inverse = method_inverse
        self.non_nega_cons = non_nega_cons
        self.all_non_nega_cons = all_non_nega_cons
        self.fixed_input=fixed_input
        if fixed_input is not None:
            self.pshape=fixed_input.shape[0]

    def gaussiankernel(self, x, z, args, N):
        if N == 1:
            sigma = args
            y = (1. / math.sqrt(2. * math.pi) / sigma) * math.exp(-(x - z) ** 2 / (2. * sigma ** 2))
        else:
            sigma = args
            cov = []
            for j in range(N):
                cov += [1. / sigma[j, j] ** 2]
            N = float(N)

            y = 1. / (2. * math.pi) ** (N / 2.) * abs(np.linalg.det(sigma)) ** (-1.) * math.exp(
                (-1. / 2.) * np.dot((x - z) ** 2, np.array(cov)))
        return y


    def lasso(self, x, D, y):
        temp = (np.dot(D,x))/D.shape[1] - y
        eq = np.dot(temp,temp)
        return eq+np.dot(x,x)


    def lessObsUpConstrain(self, x, D, y):
        temp = (np.dot(D,x))/D.shape[1] - y
        eq = np.dot(temp,temp)
        return -eq+0.1


    def moreObsfunc(self, x, D, y):
        temp = y.reshape(len(y),1)-np.dot(D,x.reshape(len(x),1))
        temp = temp.reshape(1,len(temp))
        return np.asscalar(np.dot(temp,temp.T))


    def square_sum(self, x):
        # y = np.dot(x,x)
        y = np.sum( x**2 )
        return y

    # 1.4
    def minimizer_L1(self, x):
        # D: (M, N)
        D=x[1]
        y=x[0].T
        x0=np.ones(D.shape[1],)
        if(D.shape[0] < D.shape[1]):
            # less observations than nodes
            # Adjust the options' parameters to speed up when N >= 300
            # see https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
            options = {'maxiter': 10, 'ftol': 1e-01, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-02}
            upcons = {'type':'ineq','fun':self.lessObsUpConstrain,'args':(D,y)}
            cur_time = datetime.now()
            result = minimize(self.square_sum, x0, args=(), method='SLSQP', jac=None, bounds=Bounds(0,1),
                              constraints=[upcons], tol=None, callback=None, options=options)
            # logging.info("minimizer_L1 time:" + str( datetime.now() - cur_time ) + "," + str(options) + " result.fun:" + str(result.fun) + ", " + str(result.success) + ", " + str(result.message))
        else:
            logging.info("more observations than nodes")
            result = minimize(self.moreObsfunc, x0, args=(D,y), method='L-BFGS-B', jac=None, bounds=Bounds(0,1), tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': -1, 'maxls': 20})
        return result.x




    # 没有非负约束的目标函数
    def square_sum_Lagrange(self, x, lbd, D, y):
        # y = np.dot(x,x)
        z = 2 * x - np.dot(D.T, lbd)
        z1 = np.dot(D, x) - y
        z = sum(z ** 2) + sum(z1 ** 2)
        return z

    # 没有非负约束的梯度函数
    def square_sum_Lagrange_grad(self, x, lbd, D, y):
        # y = np.dot(x,x)
        # tt = np.array( [sum(2 * (np.dot(D, x) - y) * D[:, i]) for i in range(D.shape[1])]).reshape(D.shape[1],)
        # print(tt.shape)
        x_grad = 4 * (2 * x - np.dot(D.T, lbd)) + np.array(
            [sum(2 * (np.dot(D, x) - y) * D[:, i]) for i in range(D.shape[1])]).reshape(D.shape[1], )
        lbd_grad = -2 * np.dot(D, (2 * x - np.dot(D.T, lbd)))
        # print("111")
        # print(x_grad.shape)
        # print(lbd_grad.shape)
        # print(x_grad)
        return np.append(x_grad, lbd_grad, axis=0)



    # Initialize gradient adaptation.
    def grad_adapt(self, alpha, D, y, grad_fun):
        (m, n) = D.shape

        def theta_gen_const(theta):
            while True:
                theta = theta - alpha * grad_fun(theta[:n], theta[n:], D, y)
                # print(theta)
                yield theta

        return theta_gen_const

    def grad_adapt_ineq(self, alpha, D, y, grad_fun):
        (m, n) = D.shape

        def theta_gen_const(theta):
            while True:
                grad=grad_fun(theta[:n], theta[n:], D, y)
                non_nega_grad=grad[self.non_nega_cons]
                theta_non_nega=theta[self.non_nega_cons]
                indicator=np.where((non_nega_grad>=0)*(theta_non_nega<=0))[0]
                grad[np.array(self.non_nega_cons,dtype=int)[indicator]]=0
                theta = theta - alpha * grad
                nt=theta[self.non_nega_cons]
                theta[np.where(nt<0)[0]]=0
                # print(theta)
                yield theta

        return theta_gen_const



    def read_data_from_simulation(self, obs_filepath):


        spreading_sample = pd.read_csv(obs_filepath, encoding='utf-8')
        spreading_sample = np.array(spreading_sample)



        # T = list(set(index).difference(set(deleted)))
        return spreading_sample


    # 1.1 & 1.3
    def get_r_xit(self, x, i, t_l, features, spreading, K, bandwidth, dt, G):
        numerator = 0.0
        denominator = 0.0
        # print(features.shape)
        for j in range(features.shape[0]):
            # x_j = features.iloc[j]
            # g = self.gaussiankernel(x, x_j, bandwidth, features.shape[1])
            tmp = spreading[t_l+1][j*K+i] - spreading[t_l][j*K+i]
            numerator = numerator + (1.0 * G[j] * tmp)
            denominator = denominator + (G[j] * dt)
        return numerator/denominator


    def get_r_matrix(self, features, spreading):
        # r_ma = np.loadtxt(self.save_path + "r_matrix.csv", delimiter=',')
        # print(r_ma.shape)
        # return r_ma

        #bandwidth = np.diag(np.ones(features.shape[1]) * float(features.shape[0]) ** (-1. / float(features.shape[1] + 1)))

        current_time = datetime.now()
        r_matrix = []
        r_matrix=spreading[1:]-spreading[:-1]
        res = np.array(r_matrix)

        logging.info("r_matrix time: " + str(datetime.now() - current_time))
        np.savetxt(self.save_path + "r_matrix_2.csv", res, delimiter=',')
        return res


    def save_E(self, E, filepath):
        # print(E1)
        # print("E:", len(E), len(E[0]))
        with open(filepath, "w") as f:
            writer = csv.writer(f)
            writer.writerows(E)
        # print(filepath)
        return filepath

    def clear_zeros(self, mitrix):
        # delete t with all zeros
        all_zero_columns = np.where(~mitrix.any(axis=0))[0]
        res = np.delete(mitrix, all_zero_columns, axis=1)
        return res

    def get_E(self, spreading, K):
        r_matrix = self.get_r_matrix(None, spreading)
        # spreading = np.delete(spreading, -1, axis=0)

        cshape = int((spreading.shape[1] - self.pshape) / (K - 1))
        logging.info("spreading.shape:" + str(spreading.shape))

        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)

        m, n = spreading.shape
        iter = 0
        self.iter_max = 1000
        while iter < self.iter_max:
            args_all = []

            # r_matrix: 12,2265
            for ny in range(r_matrix.shape[1]):
                # dele=np.where(y<=0)[0]
                y = r_matrix[:, ny]
                if ny < self.pshape:
                    z = self.fixed_input[ny]
                    spd = spreading[:-1, self.pshape:]
                    spd = spd / float(spd.shape[1])
                    args = (spd, z, self.pshape)
                    args_all.append(args)
                else:
                    # y = y.astype(float)
                    tmp = np.asarray(1 + spreading, dtype=float)
                    z = np.log(tmp)[1:, ny] - np.log(tmp)[:-1, ny]

                    # spd=np.delete(spreading, dele, axis=0)
                    spd = spreading[:-1]
                    spd = spd / float(spd.shape[1])

                    if iter == 0:
                        # args = (x0, spd, z, 0.001, 10000, 10**-6)
                        args = (spd, z, self.pshape)
                    else:
                        ind = int((ny - self.pshape) / cshape)
                        args = (spd, z, uni_params[ind], K, cshape, self.pshape)
                    args_all.append(args)
            # for arg in args_all:
            #     print(arg[0].shape, arg[1].shape)

            # exit()
            edge_list = pool.map(explicit_minimizer, args_all)
            univ = []
            for i in range(1, K):
                if i >= 1:  # self.pshape:
                    av = np.mean(array(edge_list[self.pshape + (i - 1) * cshape:self.pshape + (i) * cshape]), axis=0)
                    avv = av[:self.pshape]
                    for j in range(K - 1):
                        avv = append(avv, np.ones(1) * sum(av[self.pshape + (j - 1) * cshape:self.pshape + (j) * cshape]))
                    univ.append(avv)
            if iter == 0:
                uni_params = array(univ)
            else:
                univ = array(univ)
                if sum((univ - uni_params) ** 2) / min([sum(univ ** 2), sum(uni_params ** 2)]) < 0.05:
                    uni_params = univ
                    break
                else:
                    uni_params = univ
            iter += 1

        # len(edge_list) 4934
        # edge_list[0].shape (4934,)
        # ('cshape', 2467)
        # ('self.pshape', 2467)
        # ('uni_params.shape:', (1, 2468))
        # 下面的代码，把edge_list从 (4934,4934) 扩展到 (7401, 7401)。其中7401=4934+2467,  4934=2467+2467。


        for i in range(self.pshape):
            edge_list[i] = np.append(zeros(cshape), edge_list[i], axis=0)
        # for i in range(self.pshape, len(edge_list)):
        #     edge_list[i] = np.append(uni_params[(i - self.pshape) / cshape, :self.pshape], edge_list[i][ self.pshape:], axis=0)
        #     for j in range(1, K):
        #         edge_list[i][ self.pshape + (j - 1) * cshape:self.pshape + (j) * cshape] = edge_list[i][ self.pshape + ( j - 1) * cshape:self.pshape + ( j) * cshape] *  uni_params[(  i - self.pshape) / cshape, self.pshape + j - 1] / sum( edge_list[i][ self.pshape + (j - 1) * cshape:self.pshape + (j) * cshape])
        # # edge_list = pool.map(self.sgd, args_all)
        return np.array(edge_list)


    def do(self, K, infect_data, media_data):
        spreading_sample=append(infect_data,media_data,axis=1)
        E = self.get_E( spreading_sample, K)
        E_filepath = self.save_E(E, self.save_path + "to_file_E_abs_spd_" + rundate + ".csv")
        logging.info(E_filepath)
        return E_filepath

def reg(infect_matrix,popu_data,incub1,incub,os_T,date_index,infect_prob):
    print(type(infect_prob), len(infect_prob))
    x,y=[],[]
    for i in range(len(infect_matrix)):
        if os_T*(i)+incub1+incub<=len(date_index):
            d_index=array(date_index)[len(date_index)-incub1-incub-os_T*i:len(date_index)-os_T*i]

        else:
            d_index=array(date_index)[:len(date_index)-os_T*i]
        x.append(moving_average(popu_data,infect_prob[-i-1],incub,d_index))
    x=x[::-1]

    im=[infect_matrix[i] for i in range(len(infect_matrix))]
    inm=array(im).flatten()
    xx=array(x,dtype=float).flatten()

    xx=array([ones_like(xx),xx],dtype=float).T
    print dot(xx.T,xx)
    print dot(xx.T,inm)

    coef=linalg.inv(dot(xx.T,xx)).dot(dot(xx.T,inm))

    #sumsquare=lambda coef:reduce(ssm,[sum((coef[0]+coef[1]*x[i]-infect_matrix[i][1:,1:])**2) for i in range(len(infect_matrix))[start:end]])
    #res=minimize(sumsquare,x0=zeros(2))
    #print res
    #xx=array(x[:7]).flatten()

    #print xx.shape,inm.shape
    #coef=sum((xx-mean(xx))*(inm-mean(inm)))/float(xx.shape[0])/std(xx)/std(inm)
    #coef1=res.x
    print 'R^2', coef#,sumsquare(coef)/float(len(inm))/std(inm)**2#coef1,res.fun/sum(inm**2),



    return coef
def moving_average(data,prob,incub,d_index,starting=None,theta=None,):
    matrix=zeros_like(data[d_index[0]])
    a=zeros(data[d_index[0]].shape[0])
    b=zeros(data[d_index[0]].shape[0])

    for s in range(len(d_index))[incub:]:

        dd=zeros_like(matrix)

        for i,k in enumerate(d_index[s-incub:s]):


            dd=dd+data[k]*prob[i]


        matrix=matrix+dd
    matrix=matrix/float(len(data)-incub)

    return matrix#dot(a,matrix)#matrix/float(len(data)-incub)

def counterfact_sim(coef,popu_data,infect_prob,incub1,incub,os_T,date_index):
    popu=popu_data.copy()
    x=[]
    for i in range(len(infect_matrix)):
        if os_T*(i+1)+incub<=len(date_index):
            d_index=array(date_index)[len(date_index)-incub-os_T*i:len(date_index)-os_T*i]
            d1_index=array(date_index)[len(date_index)-incub-os_T*i-1:len(date_index)-os_T*i-1]
        else:
            d_index=array(date_index)[:len(date_index)-os_T*i]
            d1_index=array(date_index)[:len(date_index)-os_T*i-1]
        a=moving_average(popu,infect_prob[-i-1],incub,d_index)
        y=a*coef[1]+coef[0]
        x.append(y)
        b=moving_average(popu,infect_prob[-i-1],incub,d1_index)
        y=b*coef[1]+coef[0]
        x.append(y)
    x=x[::-1]
    return x




if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = BASE_DIR + '/data/'
    rundate = time.strftime("%m%d%H%M", time.localtime())


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
    not_found = [38051, 38085, 2050, 38023, 19085, 80015, 2195, 38037, 46102, 90009, 90008, 15001, 15003, 90013, 99999, 90016, 15009, 90018, 90032, 90015, 90021, 90023, 80040, 90025, 2090, 78000, 90034, 90035, 88888, 15007, 90044, 2110, 90047, 2240, 90049, 66, 90053, 90017, 72, 2122, 53069, 78, 38029, 66000, 2130, 250, 2020, 46057, 26097, 2290, 27129, 2170, 38011, 90026, 90005]

    cities = []
    cities_2_str = []

    pickle_file = BASE_DIR + '/yuqing/data_2.pkl'
    if os.path.exists(pickle_file):
        with open(pickle_file,"rb") as f:
            infect_matrix = pickle.load(f)
            infect_prob = pickle.load(f)
            boom_prob = pickle.load(f)
            recovery = pickle.load(f)
            hidden_infect = pickle.load(f)
            raw_data = pickle.load(f)
            cities_2 = pickle.load(f)
            cities_2_str = pickle.load(f)
    else:
        for i in range(8):
            print("reading result_%d" % i)
            data=pd.ExcelFile(BASE_DIR + '/yuqing/result_'+str(i)+'.xlsx')

            item ="ad_matrix_%d" % (i*2)
            df = data.parse(item)[:-1]
            if cities==[] and cities_2_str==[]:
                cities = df.city
                cities = map(lambda x: int(x), cities)
                cities_2 = list( set(cities) - set(not_found) )
                cities_2_str = map(lambda x: str(x), cities_2)
            df.index = cities
            df = df.ix[cities_2][cities_2_str]
            infect_matrix.append(array(df)[:,:])

            item="prob_%d" % (i*2)
            infect_prob.append(array(data.parse(item))[:,2])
            boom_prob.append(array(data.parse(item))[:,1])

            item="recovery_%d" % (i*2)
            df = data.parse(item)[:-1]
            df.index = cities
            df = df.ix[cities_2]
            recovery.append(array(df)[0,1])

            item="hidden_data_%d" % (i*2)
            hd=data.parse(item)[:-1]
            hd.index = cities
            hd = hd.ix[cities_2]
            # hidden_infect.append(array(hd[[it for it in hd.columns if 'pre' in str(it)]]))
            hidden_infect.append(array(hd)[:,1:-1])

            item="raw_data_%d" % (i*2)
            df = data.parse(item)[:-1]
            df.index = cities
            df = df.ix[cities_2]
            raw_data.append(array(df)[:,1:])

        raw_data=raw_data[::-1]

        with open(pickle_file,"wb") as f:
            pickle.dump(infect_matrix, f)
            pickle.dump(infect_prob, f)
            pickle.dump(boom_prob, f)
            pickle.dump(recovery, f)
            pickle.dump(hidden_infect, f)
            pickle.dump(raw_data, f)
            pickle.dump(cities_2, f)
            pickle.dump(cities_2_str, f)

    # cities=array(data.parse('recovery_14'))[:,0:1]

    infect_matrix=infect_matrix[::-1]
    infect_prob=infect_prob[::-1]
    boom_prob=boom_prob[::-1]
    recovery=recovery[::-1]
    hidden_infect=hidden_infect[::-1]

    popu=pd.read_csv(BASE_DIR + '/yuqing/movement_matrix_county_03-08_to04-09_new.csv')
    popu.index=popu['origin_fips/destination_fips']
    popu.fillna(0,inplace=True)


    # popu=array(popu.ix[cities_2][cities_2_str])[:,1:].T
    popu=array(popu.ix[cities_2][cities_2_str]).T
    date_index=['2020-03-0'+str(i) for i in range(8,10)]+['2020-03-'+str(i) for i in range(10,32)]
    date_index=date_index+['2020-04-0'+str(i) for i in range(1,10)]
    popu_data={it:popu for it in date_index}


    coef=reg(infect_matrix,popu_data,incub1,incub,os_T,date_index,infect_prob)
    fitted_matrix=counterfact_sim(coef,popu_data,infect_prob,incub1,incub,os_T,date_index)
    for i in range(8):
        if i==0:
            hidden_data=hidden_infect[i]
            report_data=raw_data[i]
        else:
            report_data=append(report_data,raw_data[i][:,-os_T:],axis=1)
            hidden_data=append(hidden_data,hidden_infect[i][:,-os_T:],axis=1)
    hidden=[]
    # hidden_data=hidden_data[1:] # 暂时注释掉，要不然后面会报错
    hidden=hidden_data[:,incub+1:]-hidden_data[:,incub:-1]
    forecast_hidden=[]
    h=hidden.shape[1]
    f=h #len(fitted_matrix)

    print("hidden.shape:", hidden.shape)
    print("hidden_data.shape:", hidden_data.shape)

    for j in range(f):
        if j >= len(fitted_matrix):
            break
        t1 = dot(fitted_matrix[j],hidden_data[:,h-f+j:h-f+j+incub])
        print("t1.shape:", t1.shape)
        t2 = dot(t1, infect_prob[int(j/2)])
        print("t2.shape:", t2.shape)
        o1=hidden[:,h-f+j]- t2 + hidden_data[:,h-f+j+incub]*recovery[int(j/2)]
        forecast_hidden.append(o1)
    fixed_input=array(forecast_hidden).T
    infect_data=report_data[:,:f].T


    spreading = infect_data[1:] - infect_data[:-1]
    print("len(spreading)", len(spreading))
    print("len(cities_2)",len(cities_2))

    sum_col = np.sum(np.absolute(spreading), axis=0)
    deleted = []
    for i in range(len(sum_col)):
        if sum_col[i] == 0:
            deleted.append(i)

    infect_data = np.delete(infect_data, deleted, axis=1)
    fixed_input=np.delete(fixed_input, deleted, axis=0)
    cities_2 = array([cities_2[i] for i in range(len(cities_2)) if i not in deleted ] )
    cities_2_str = array([cities_2_str[i] for i in range(len(cities_2_str)) if i not in deleted ] )



    # to_file = save_path + "to_file_" + rundate + ".csv"
    today = time.strftime("%Y-%m-%d", time.localtime())
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=BASE_DIR + '/' + today + '.log')

    #time = 7.5
    #dt = 0.05

    K = 2
    obs_filepath = BASE_DIR + '/yuqing/obs_k1_with_header_19days_3.csv'
    media_data=pd.read_csv(obs_filepath)
    media_data=array(media_data[cities_2_str])[:f]


    print("media_data.shape",media_data.shape) # (13, 2545)

    (nodes_num, obs_num) = media_data.shape

    logging.info("start: " + str(nodes_num) + "x" + str(obs_num))

    save_path = BASE_DIR + '/data/'

    non_nega_cons = [i for i in range(nodes_num * K)]
    mhl = MiningHiddenLink(save_path, True,fixed_input, non_nega_cons, True)
    # ac = accuracy(save_path)

    current_time = datetime.now()
    e_filepath = mhl.do(K, infect_data, media_data)
    logging.info("done")
    print("done")
