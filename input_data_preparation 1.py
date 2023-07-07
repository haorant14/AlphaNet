#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from datetime import datetime
import idx_w_stock_basic_filter


# ## FILTER DATA

# In[2]:


def get_index(time,start_date,end_date):
    '''
        get the index of the row where the 'time' is after 'date'
    '''
    time = np.array(time)
    start_indices = np.where(time>=start_date) # return the tuple of index of the date wanted
    end_indices   = np.where(time<=end_date)
    return start_indices[0][0],end_indices[0][-1]+1

def get_ts_code_list():
    '''
        get the list of ts_code of 000905.SH stock where the list_date are not half a year ago and are not ST stock
    '''
    matrix_stock_basic = idx_w_stock_basic_filter.get_filtered_ts_code()
    ts_code_list = matrix_stock_basic[:,0]
    return ts_code_list


# ## READ INPUT DATA FROM MAT FILE

# In[3]:


def load_data(ts_code):
    mat_daily       = sio.loadmat('daily/' + ts_code + '.mat')
    mat_daily_basic = sio.loadmat('daily_basic/'+ ts_code + '.mat')
    daily           = mat_daily['daily']
    daily_basic     = mat_daily_basic['daily_basic']
    start_date      = 20160101
    end_date        = 20200801
    '''
    daily
    '''
    # read data starting from 2020/08/01
    time_daily   = daily[0,0]['time']
    start_index_daily,end_index_daily  = get_index(time_daily,start_date,end_date)

    open_        = daily[0,0]['open'][start_index_daily:end_index_daily]
    high         = daily[0,0]['high'][start_index_daily:end_index_daily]
    low          = daily[0,0]['low'][start_index_daily:end_index_daily]
    close        = daily[0,0]['close'][start_index_daily:end_index_daily]
    volume       = daily[0,0]['volume'][start_index_daily:end_index_daily] # 成交量
    turnover     = daily[0,0]['turnover'][start_index_daily:end_index_daily] # 成交额
    adj_factor   = daily[0,0]['adj_factor'][start_index_daily:end_index_daily] # 复权因子
    time_daily   = time_daily[start_index_daily:end_index_daily]
    matrix_daily = np.concatenate((time_daily,open_,high,low,close,volume,turnover,adj_factor),axis=1)
    matrix_daily = matrix_daily[np.lexsort((matrix_daily[:,-1],matrix_daily[:,0]))] # sort the matrix by time then by adj_factor
    # remove duplicate daily: duplicate rows of same time
    _,de_duplicate_index = np.unique(matrix_daily[:,0],return_index=True)
    matrix_daily = matrix_daily[de_duplicate_index]

    '''
    daily_basic
    '''
    # read data starting from year 2020/08/01
    time_daily_basic  = daily_basic[0,0]['time'] # obtain time in the format yyyymmdd as integer
    start_index_daily_basic,end_index_daily_basic = get_index(time_daily_basic,start_date,end_date)

    turnover_rate      = daily_basic[0,0]['turnover_rate'][start_index_daily_basic:end_index_daily_basic]
    turnover_rate_free = daily_basic[0,0]['turnover_rate_free'][start_index_daily_basic:end_index_daily_basic]
    float_share        = daily_basic[0,0]['float_share'][start_index_daily_basic:end_index_daily_basic]
    free_share         = daily_basic[0,0]['free_share'][start_index_daily_basic:end_index_daily_basic]
    time_daily_basic   = time_daily_basic[start_index_daily_basic:end_index_daily_basic]
    matrix_daily_basic = np.concatenate((time_daily_basic,turnover_rate,turnover_rate_free,float_share,free_share),axis=1)
    matrix_daily_basic = matrix_daily_basic[matrix_daily_basic[:,0].argsort()] # sort the matrix by time   
    # remove duplicate daily_basic: duplicate rows of same time
    _,de_duplicate_index = np.unique(matrix_daily_basic[:,0],return_index=True)
    matrix_daily_basic = matrix_daily_basic[de_duplicate_index]
    
    return matrix_daily, matrix_daily_basic


# ## OBTAIN VWAP, TURN, FREE_TURN

# In[4]:


def get_return(matrix_daily):
    '''
    通过收盘价计算日频收益率:
        (收盘价 - 昨日收盘价) / 昨日收盘价
    '''
    size  = len(matrix_daily)
    close = matrix_daily[:,4:5]
    close_yesterday = np.empty((1,1))
    close_yesterday[0,0] = np.nan
    close_yesterday = np.concatenate((close_yesterday,close),axis=0)[:-1]
    return_ = (close - close_yesterday) / close_yesterday
    return return_

def get_vwap(volume,turnover):
    vwap = turnover / volume
    return vwap

def get_turnover(volume,share):
    '''
    turnover rate = volume / float share
    free turnover rate = volume / free share
    '''
    turnover_rate = volume/share
    return turnover_rate

def get_vwap_turnover(matrix_daily,matrix_daily_basic):
    volume   = matrix_daily[:,5]
    turnover = matrix_daily[:,6]
    float_share = matrix_daily_basic[:,3]
    free_share  = matrix_daily_basic[:,4]
    
    vwap     = get_vwap(volume,turnover)
    turnover_rate      = get_turnover(volume,float_share)
    turnover_rate_free = get_turnover(volume,free_share)
    return vwap, turnover_rate, turnover_rate_free


# ## SPLIT ADJUST

# In[5]:


def split_adjust(matrix_daily):
    '''
    前复权
    '''
    size  = len(matrix_daily)
    open_ = matrix_daily[:,1:2]
    high  = matrix_daily[:,2:3]
    low   = matrix_daily[:,3:4]
    close = matrix_daily[:,4:5]
    adj_factor = matrix_daily[:,7:8]

    last_adj_factor     = adj_factor[-1]
    last_adj_factor_vec = np.empty(size)
    last_adj_factor_vec.fill(last_adj_factor[0])
    last_adj_factor_vec = np.reshape(last_adj_factor_vec,(size,1))
    
    matrix_tmp = np.concatenate((open_,high,low,close),axis=1)
    matrix_tmp = adj_factor / last_adj_factor_vec * matrix_tmp
    
    matrix_daily[:,1] = matrix_tmp[:,0]
    matrix_daily[:,2] = matrix_tmp[:,1]
    matrix_daily[:,3] = matrix_tmp[:,2]
    matrix_daily[:,4] = matrix_tmp[:,3]
    return matrix_daily


# ## GET ALL TRADE DATE

# In[6]:


def get_all_trade_date_and_raw_data(ts_code_list):
    '''
    得到2016/01-2020/07所有unique交易日
    得到所有股票原始daily和daily_basic的数据
    '''
    trade_date_list = np.array([]) # 所有交易日
    dict_matrix_daily       = {}
    dict_matrix_daily_basic = {}
    
    for ts_code in ts_code_list:
        matrix_daily, matrix_daily_basic = load_data(ts_code)
        dict_matrix_daily[ts_code]       = matrix_daily
        dict_matrix_daily_basic[ts_code] = matrix_daily_basic
        time_daily       = matrix_daily[:,0]
        time_daily_basic = matrix_daily_basic[:,0]
        trade_date_list  = np.concatenate((trade_date_list,time_daily),axis=0)
        trade_date_list  = np.concatenate((trade_date_list,time_daily_basic),axis=0)
        trade_date_list  = np.unique(np.sort(trade_date_list,axis=0)) #所有unique交易日
    trade_date_list   = np.reshape(trade_date_list,(len(trade_date_list),1))
    return trade_date_list,dict_matrix_daily,dict_matrix_daily_basic


# ## GET INPUT MATRIX

# In[7]:


def get_concat_matrix(matrix_daily,vwap,return_,turnover_rate,turnover_rate_free,trade_date_list):
    '''
    need open, high, low, close, vwap, volume, return, turn, free_turn
    '''
    size     = len(matrix_daily)
    volume   = matrix_daily[:,5:6]
    
    vwap     = np.reshape(vwap,(size,1))
    return_  = np.reshape(return_,(size,1))
    turnover_rate      = np.reshape(turnover_rate,(size,1))
    turnover_rate_free = np.reshape(turnover_rate_free,(size,1))

    x_matrix = matrix_daily[:,0:5] # time, open, high, low, close
    x_matrix = np.concatenate((x_matrix,vwap,volume,return_,turnover_rate,turnover_rate_free),axis=1)
    
    
    # fill rows of nan for the date where 个股 does not have any trade info
    nan_row  = np.full((1,9),np.nan)
    if len(matrix_daily) < len(trade_date_list):
        trade_date_missing = np.setdiff1d(trade_date_list,x_matrix[:,0]) # the missing trade date of a stock
        full_trade_date_matrix  = np.empty((len(trade_date_list),9)) # includes info such as open, close ... for all trade date
        
        iter_ = 0
        for trade_date in trade_date_list:
            if trade_date in trade_date_missing:
                full_trade_date_matrix[iter_,:] = nan_row
            else:
                index = np.where(x_matrix[:,0] == trade_date)
                full_trade_date_matrix[iter_,:] = x_matrix[index,1:]
            iter_ += 1
        return full_trade_date_matrix
    
    x_matrix = x_matrix[:,1:10]
    return x_matrix


# ## GET DATA IMAGE

# In[8]:


def get_single_data_matrix(data,backtrack_interval,img_dim):
    '''
    把单个数据，如open，close，的vector变为matrix
    speed up computation for constructing data image
    每隔两天采样一次
    '''
    data   = np.reshape(data,(1,len(data)))
    matrix = np.empty((img_dim,backtrack_interval))
    for i in range(img_dim):
        matrix[i,:] = data[:,i*2:backtrack_interval+i*2]
    matrix = matrix.T
    return matrix


# In[9]:


def get_data_image(x_matrix,backtrack_interval):
    '''
    param:
        x_matrix: 2016/01-2020/08个股数据
        backtrack_interval  (int): 回溯天数
    return: 
        n*1*9*30的个股数据图片
    '''
    # original x_matrix dimension
    orig_row_size = len(x_matrix)
    orig_col_size = len(x_matrix[0])
    # x_img dimension
    img_dim  = int((orig_row_size - backtrack_interval)/2) + 1
    
    # convert all data vectors to matrices
    x_img = np.empty((orig_col_size,1,backtrack_interval,img_dim))
    for i in range(orig_col_size):
        x_img[i,0,:,:] = get_single_data_matrix(x_matrix[:,i],backtrack_interval,img_dim)
    x_img = np.transpose(x_img,(3,1,0,2)) #number of sample, 1, number of data (open,close...), number of days
    return x_img


# In[10]:


def get_y_label(close,backtrack_interval,future_day):
    '''
    param:
        close: 收盘价
        backtrack_interval: 回溯天数
        future day: 计算未来多少天的收益
        
    return:
        y_label
    '''
    close_size  = len(close)
    close_shift = np.empty((1,close_size-backtrack_interval+1))
    close_shift[:,0:close_size-backtrack_interval-future_day+1] = close[backtrack_interval+future_day-1:]
    close_shift[:,close_size-backtrack_interval-future_day+1:]  = np.nan
    close   = close[backtrack_interval-1:].reshape((1,len(close_shift[0])))
    return_ = (close_shift-close) / close
    return_ = return_.ravel()[::2]
    return return_


# In[14]:


def construct_dataset():
    backtrack_interval = 30 # 回溯天数
    data_num           = 9 # open,high,low,...
    future_day_5       = 5
    future_day_10      =10
    X_train    = np.empty((1,1,data_num,backtrack_interval))
    Y_train_5  = np.empty((1,))
    Y_train_10 = np.empty((1,))
    X_test     = np.empty((1,1,data_num,backtrack_interval))
    Y_test_5   = np.empty((1,))
    Y_test_10  = np.empty((1,))
    
    ts_code_list    = get_ts_code_list() # 2016-2020 中证500股票代码
    trade_date_list,dict_matrix_daily,dict_matrix_daily_basic  = get_all_trade_date_and_raw_data(ts_code_list) # 所有交易日,所有股票数据
    train_index     = int(4 * ((len(trade_date_list))/5)) # index that divides the dataset into train and test sets

    for ts_code in ts_code_list:
        matrix_daily, matrix_daily_basic      = dict_matrix_daily[ts_code], dict_matrix_daily_basic[ts_code]
        matrix_daily                          = split_adjust(matrix_daily) # 前复权
        vwap,turnover_rate,turnover_rate_free = get_vwap_turnover(matrix_daily,matrix_daily_basic) # 计算vwap, turn, free_turn
        return_                               = get_return(matrix_daily) # 计算return1
        
        x_matrix = get_concat_matrix(matrix_daily,vwap,return_,turnover_rate,turnover_rate_free,trade_date_list)
        x_train_matrix = x_matrix[:train_index]
        x_test_matrix  = x_matrix[train_index:]
        x_train_img    = get_data_image(x_train_matrix,backtrack_interval)
        x_test_img     = get_data_image(x_test_matrix,backtrack_interval)
                
        y_train_5  = get_y_label(x_train_matrix[:,3],backtrack_interval,future_day_5)
        y_test_5   = get_y_label(x_test_matrix[:,3],backtrack_interval,future_day_5)
        y_train_10 = get_y_label(x_train_matrix[:,3],backtrack_interval,future_day_10)
        y_test_10  = get_y_label(x_test_matrix[:,3],backtrack_interval,future_day_10)
        
        X_train = np.concatenate((X_train,x_train_img),axis=0)
        X_test  = np.concatenate((X_test,x_test_img),axis=0)
        Y_train_5  = np.concatenate((Y_train_5,y_train_5),axis=0)
        Y_test_5   = np.concatenate((Y_test_5,y_test_5),axis=0)
        Y_train_10 = np.concatenate((Y_train_10,y_train_10),axis=0)
        Y_test_10  = np.concatenate((Y_test_10,y_test_10),axis=0)
        
    X_train = X_train[1:,:,:,:] # final train input
    X_test  = X_test[1:,:,:,:] # final test input
    Y_train_5  = Y_train_5[1:] # final train label future 5 days return
    Y_test_5   = Y_test_5[1:] # final test label future 5 days return
    Y_train_10 = Y_train_10[1:] # final train label future 10 days return
    Y_test_10  = Y_test_10[1:] # final test label future 10 days return
    return X_train, X_test, Y_train_5, Y_test_5, Y_train_10, Y_test_10


# In[15]:


import time
start_time = time.time()

X_train, X_test, Y_train_5, Y_test_5, Y_train_10, Y_test_10 = construct_dataset()
np.save('./X_train.npy',X_train)
np.save('./X_test.npy',X_test)
np.save('./Y_train_5.npy',Y_train_5)
np.save('./Y_test_5.npy',Y_test_5)
np.save('./Y_train_10.npy',Y_train_10)
np.save('./Y_test_10.npy',Y_test_10)

print("--- %s seconds ---" % (time.time() - start_time))