import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import datetime
from dateutil.relativedelta import relativedelta

def load_index_weight_stock_basic():
    '''
    读取indew_weight中证500数据并提取con_code
    读取stock_basic并提取ts_code与中证500的con_code对应的股票的ts_code,股票名字和上市日期
    '''
    mat_index_weight = sio.loadmat('index_weight.mat')
    mat_stock_basic  = sio.loadmat('stock_basic.mat')
    index_weight     = mat_index_weight['index_weight']
    stock_basic      = mat_stock_basic['stock_basic']
    con_code    = np.unique(index_weight[0,0]['con_code']).reshape(-1,1)
    ts_code     = stock_basic[0,0]['ts_code'].reshape(-1,1)
    name        = stock_basic[0,0]['name'].reshape(-1,1)
    list_status = stock_basic[0,0]['list_status'].reshape(-1,1)
    list_date   = stock_basic[0,0]['list_date'].reshape(-1,1)
    matrix_stock_basic = np.concatenate((ts_code,name,list_status,list_date),axis=1)
    # return con_code, ts_code, name, list_date, matrix_stock_basic
    return con_code, matrix_stock_basic

def get_000905_SH_stock(matrix_stock_basic,ts_code,con_code):
    '''
    筛选出中证500股票
    '''
    intersect = np.flatnonzero(np.in1d(ts_code,con_code).reshape(ts_code.shape).any(1)) # 得到中证500股票的index
    intersect = intersect.reshape(-1,1)
    matrix_stock_basic = matrix_stock_basic[intersect] # 只取中证500股票
    return np.squeeze(matrix_stock_basic)

def remove_st_and_new_stock(matrix_stock_basic,date=datetime.date(2021,1,1)):
    '''
    去除中证500股票中是ST的股票,去除上市日期在做实验期半年内的股票,去除退市股票
    '''
    target_date  = date - relativedelta(months=6) 
    target_date  = target_date.strftime("%Y%m%d")
    matrix_stock_basic = [stock_basic for stock_basic in matrix_stock_basic 
                          if ('ST' not in stock_basic[1] and 'L' == stock_basic[2] and target_date > stock_basic[3])]
    return np.squeeze(matrix_stock_basic)

def get_filtered_ts_code():
    '''
    获得筛选好的股票数据(ts_code,股票名称,是否上市,上市日期)
    '''
    con_code, matrix_stock_basic = load_index_weight_stock_basic()
    matrix_stock_basic = get_000905_SH_stock(matrix_stock_basic,matrix_stock_basic[:,0:1],con_code)
    matrix_stock_basic = remove_st_and_new_stock(matrix_stock_basic)
    return matrix_stock_basic

if __name__ == '__main__':
    ts_code_list = get_filtered_ts_code()
    print(ts_code_list)