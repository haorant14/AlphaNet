import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from datetime import datetime
import input_data_preparation
import matplotlib.pyplot as plt
from torchvision import datasets
from collections import OrderedDict
from sklearn.model_selection import KFold
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,Dataset
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class alphanet_dataset(Dataset):
    '''
    作为alphanet数据的dataset,可作为torch dataloader输入
    '''
    def __init__(self,data,label):
        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label)
    def __getitem__(self,index):
        return self.data[index],self.label[index]
    def __len__(self):
        return len(self.data)

def normalize_input(x):
    '''
    normalize每个数据图片
    '''
    mean = np.nanmean(x,axis=(3,4))[...,np.newaxis,np.newaxis]
    std  = np.nanstd(x,axis=(3,4))[...,np.newaxis,np.newaxis]
    norm = (x - mean) / (std + 1e-5)
    return norm

def normalize(y):
    '''
    normalize作为alphanet输入的标签
    '''
    y_norm = np.empty((y.shape))
    for i in range(len(y)):
        return_ = y[i]
        mean   = np.nanmean(return_)
        std    = np.nanstd(return_)
        y_norm[i] = (return_-mean) / std
    return y_norm

def load_dataset():
    '''
    载入或生成dataset
    '''
    if os.path.isfile('x5.npy')==False and os.path.isfile('x10.npy')==False \
        and os.path.isfile('y5.npy')==False and os.path.isfile('y10.npy')==False:
        input_data_preparation.construct_dataset() # 根据原始数据生成数据图片和对应的标签
    X_5  = np.load('x5.npy')
    X_10 = np.load('x10.npy')
    Y_5  = np.load('y5.npy')
    Y_10 = np.load('y10.npy')
    ts_code_list = np.load('ts_code_list.npy')
    sample_date = X_5[:,0,:,0,-1] # 截面期
    sample_date = [datetime.strptime(str(int(item)),"%Y%m%d") for item in sample_date]
    X_5  = X_5[:,:,:,1:,:] # 在数据图片中去除时间数据(时间数据for debugging preposes)
    X_10 = X_10[:,:,:,1:,:]
    return X_5, X_10, Y_5, Y_10, sample_date, ts_code_list

X_5, X_10, Y_5, Y_10, sample_date, ts_code_list = load_dataset()
print('Dataset is loaded')
# 标准化收益率
Y_norm5  = normalize(Y_5)
Y_norm10 = normalize(Y_10)
# 训练集和验证集按1:1划分
dev_idx   = int(len(sample_date)*0.5)
x_train5 = X_5[:dev_idx]
y_train5 = Y_norm5[:dev_idx]
x_train10 = X_10[:dev_idx]
y_train10 = Y_norm10[:dev_idx]
x_dev5 = X_5[dev_idx:]
y_dev5 = Y_norm5[dev_idx:]
x_dev10 = X_10[dev_idx:]
y_dev10 = Y_norm10[dev_idx:]
# batch size（所有输入都在一个batch）
input_size = X_5.shape[0]*X_5.shape[1]

def get_dataloader(x,y):
    x_,y_,nan_idx = preprocess_input(x,y) # 预处理数据(特征提取,池化,BN)为alphanet的输入和标签
    dataset = alphanet_dataset(np.squeeze(x_),y_)
    dataloader = DataLoader(dataset,batch_size=input_size,shuffle=False)
    return dataloader, nan_idx

# 特征提取
def generateC(n):
    '''
    生成组合数数对
    '''
    pair = []
    for i in range(n):
        for j in range(n):
            if all([i!=j,i<j]):
                pair.append([i,j])
    return pair

def ts_corr(X,pairs,d=10,stride=10):
    """
    过去d天X值构成的时序数列和Y值构成的时序数列的相关系数
    corr = cov(X,Y) / ( std(X) * std(Y) )
    """
    shape1,shape2,shape3,shape4,shape5 = X.shape
    all_corr = np.empty((shape1,shape2,shape3,len(pairs),3))
    X_ = np.reshape(X,(shape1,shape2,shape3,shape4,-1,stride))
    mean = np.mean(X_,axis=-1,keepdims=True)
    std  = np.std(X_,axis=-1,keepdims=True)
    x_center = X_ - mean
    for i, pair in enumerate(pairs):
        x_idx,y_idx = pair[0],pair[1]
        x = x_center[:,:,:,x_idx:x_idx+1,...]
        y = x_center[:,:,:,y_idx:y_idx+1,...]
        x_std = std[:,:,:,x_idx:x_idx+1,...]
        y_std = std[:,:,:,y_idx:y_idx+1,...]
        cov = np.sum(x*y,axis=-1,keepdims=True) / d
        corr = (cov / (x_std*y_std))[:,:,:,:,:,0]
        all_corr[:,:,:,i:i+1,:] = corr
    return all_corr

def ts_cov(X,pairs,d=10,stride=10):
    """
    过去d天X值构成的时序数列和Y值构成的时序数列的协方差
    Sum( (X-mean(X)) * (Y-mean(Y)) ) / # of time segment
    """
    shape1,shape2,shape3,shape4,shape5 = X.shape
    all_cov = np.empty((shape1,shape2,shape3,len(pairs),3))
    X_ = np.reshape(X,(shape1,shape2,shape3,shape4,-1,stride))
    mean = np.mean(X_,axis=-1,keepdims=True)
    x_center = X_ - mean
    for i, pair in enumerate(pairs):
        x_idx,y_idx = pair[0],pair[1]
        x = x_center[:,:,:,x_idx:x_idx+1,...]
        y = x_center[:,:,:,y_idx:y_idx+1,...]
        cov = np.sum(x*y,axis=-1) / d
        all_cov[:,:,:,i:i+1,:] = cov
    return all_cov

def ts_stddev(X,d=10,stride=10):
    """
    过去 d 天 X 值构成的时序数列的标准差
    """
    shape1,shape2,shape3,shape4,shape5 = X.shape
    X_ = np.reshape(X,(shape1,shape2,shape3,shape4,-1,stride))
    std  = np.std(X_,axis=-1)
    return std

def ts_zscore(X,d=10,stride=10):
    """
    过去 d 天 X 值构成的时序数列的平均值除以标准差。
    """
    shape1,shape2,shape3,shape4,shape5 = X.shape
    X_ = np.reshape(X,(shape1,shape2,shape3,shape4,-1,stride))
    mean = np.mean(X_,axis=-1)
    std  = np.std(X_,axis=-1)
    zscore = mean / (std+1e-5)
    return zscore

def ts_return(X,d=10,stride=10):
    """
    (X - delay(X, d))/delay(X, d)-1, delay(X, d)为 X 在 d 天前的取值。
    """
    shape1,shape2,shape3,shape4,shape5 = X.shape
    X_ = np.reshape(X,(shape1,shape2,shape3,shape4,-1,stride))
    x = X_[:,:,:,:,:,-1:]
    delay = X_[:,:,:,:,:,0:1]
    return_ = (x-delay) / (delay-1)
    return return_.reshape(shape1,shape2,shape3,shape4,-1)

def ts_decaylinear(X,d=10,stride=10):
    """
    过去 d 天 X 值构成的时序数列的加权平均值，权数为 d, d – 1, …, 1(权数之和应为 1，需进行归一化处理)，
    其中离现在越近的日子权数越大
    """
    shape1,shape2,shape3,shape4,shape5 = X.shape
    X_ = np.reshape(X,(shape1,shape2,shape3,shape4,-1,stride))
    weight = np.arange(1,d+1)
    weight = (weight / np.sum(weight)).reshape(1,-1)
    decaylinear = np.sum(X_*weight,axis=-1) / np.sum(weight)
    return decaylinear

def ts_mean(X,d=10,stride=10):
    """
    过去 d 天 X 值构成的时序数列之和
    """
    shape1,shape2,shape3,shape4,shape5 = X.shape
    X_ = np.reshape(X,(shape1,shape2,shape3,shape4,-1,stride))
    mean = np.mean(X_,axis=-1)
    return mean

def feature_extract(input_,pairs):
    '''
    特征提取层
    '''
    ts_corr10        = ts_corr(input_,pairs)
    ts_cov10         = ts_cov(input_,pairs)
    ts_stddev10      = ts_stddev(input_)
    ts_zscore10      = ts_zscore(input_)
    ts_return10      = ts_return(input_)
    ts_decaylinear10 = ts_decaylinear(input_)
    ts_mean10        = ts_mean(input_)
    feature_layer = np.concatenate((ts_corr10,ts_cov10,ts_stddev10,ts_zscore10,ts_return10,ts_decaylinear10,ts_mean10),axis=3)
    return feature_layer

def remove_nan(input_,label):
    '''
    在展平时间和股票数维度后的数据图片中选出不含nan的数据图片的index
    '''
    shape1,shape2,shape3,shape4,shape5 = input_.shape
    input_ = np.reshape(input_,(shape1*shape2,shape3,shape4,shape5))
    label  = np.reshape(label,(label.shape[0]*label.shape[1],1))
    nan_idx = np.unique((input_ != input_).nonzero()[0]) # 得含有nan的图片的index
    idx = [x for x in range(len(input_)) if x not in nan_idx]
    input_ = input_[idx]
    label  = label[idx]
    return input_, label

def get_nan_idx(input_):
    '''
    得到每个含有nan的图片数据所属的截面期和股票以及在输入中对应的index pair, (0,1)表示第一个截面期的第二支股票
    '''
    nan_idx = (input_ != input_).nonzero()
    date  = nan_idx[0]
    stock = nan_idx[1]
    date_stock_pair = [p for p in zip(date,stock) if None not in p]
    date_stock_pair = list(set(date_stock_pair))
    date_stock_pair = sorted(date_stock_pair,key=lambda tup:(tup[0],tup[1]))
    return date_stock_pair

def bn1(x):
    shape1,shape2,shape3,shape4,shape5 = x.shape
    mean = np.nanmean(x,axis=(0,1,2,4)).reshape(-1,1) 
    std  = np.nanstd(x,axis=(0,1,2,4)).reshape(-1,1)
    bn   = (x - mean) / (std + 1e-5)
    return bn

def pooling(x):
    '''
    池化层
    '''
    maxpool = np.max(x,axis=-1)[...,np.newaxis]
    minpool = np.min(x,axis=-1)[...,np.newaxis]
    avgpool = np.mean(x,axis=-1)[...,np.newaxis]
    pool    = np.concatenate((maxpool,minpool,avgpool),axis=3)
    return pool

def bn2(x):
    # 因子做normalize
    shape1,shape2,shape3,shape4,shape5 = x.shape
    # mean = np.nanmean(x,axis=(1,2,4)).reshape(shape1,1,1,shape4,1)
    # std  = np.nanmean(x,axis=(1,2,4)).reshape(shape1,1,1,shape4,1)
    mean = np.nanmean(x,axis=(0,1,2,4)).reshape(-1,1)
    std  = np.nanmean(x,axis=(0,1,2,4)).reshape(-1,1)
    bn   = (x - mean) / (std + 1e-5)
    return bn

def preprocess_input(x,y):
    '''
    将数据图片做特征提取和池化以及BN操作,作为alphanet输入
    '''
    shape1,shape2,shape3,shape4,shape5 = x.shape
    nan_pair = get_nan_idx(x)
    pairs = generateC(9)
    x_feature = feature_extract(x,pairs)
    x_bn1 = bn1(x_feature)
    x_pool= pooling(x_bn1)
    # flatten feature extract layer
    x_bn1 = x_bn1.reshape(shape1,shape2,shape3,-1,1)
    x_bn2 = bn2(x_pool)
    x_bn  = np.concatenate((x_bn1,x_bn2),axis=-2)
    # 展平截面和股票维度并去除含有nan的图片
    x,y = remove_nan(x_bn,y)
    return x,y,nan_pair

print('Constructing Dataloader')
# 训练集dataloader
tr_loader5,tr_nan5   = get_dataloader(x_train5,y_train5)
tr_loader10,tr_nan10 = get_dataloader(x_train10,y_train10)
# 验证集dataloader
dev_loader5,dev_nan5   = get_dataloader(x_dev5,y_dev5)
dev_loader10,dev_nan10 = get_dataloader(x_dev10,y_dev10)
# 所有数据dataloader
wholeloader5,nan5 = get_dataloader(X_5,Y_norm5)
wholeloader10,nan10 = get_dataloader(X_10,Y_norm10)
# 2020年开始数据集
wholeloader2020,nan2020 = get_dataloader(X_5[473:],Y_norm5[473:])
print('Dataloader constructed')

def reconstruct_alpha(y_pred,y_train,unique_nan_idx):
    '''
    还原alphanet预测的因子值的截面期和股票数维度
    '''
    y_construct = np.zeros(y_train.shape)
    for idx in unique_nan_idx:
        y_construct[idx] = np.nan
    iter_ = 0
    for i in range(y_construct.shape[0]):
        for j in range(y_construct.shape[1]):
            if (~np.isnan(y_construct[i,j,0])):
                y_construct[i,j,0] = y_pred[iter_,0]
                iter_ += 1
    return y_construct

class AlphaNet(nn.Module):
    def __init__(self):
        super(AlphaNet,self).__init__()
        self.fc      = nn.Linear(702,30)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.output  = nn.Linear(30,1)
        # 用truncated_normal初始化全连接层和输出层的权重值
        nn.init.trunc_normal_(self.fc.weight)
        nn.init.trunc_normal_(self.output.weight)
        nn.init.trunc_normal_(self.fc.bias)
        nn.init.trunc_normal_(self.output.bias)
    def forward(self,input_):
        output_       = self.fc(input_)
        output_       = self.dropout(output_)
        output_       = self.relu(output_)
        output_       = self.output(output_)
        return output_

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AlphaNet()
loss_fn = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(),lr=0.0001)

def train(trainloader, devloader, model, loss_fn, optimizer, path):
    min_valid_loss = np.inf
    early_stop_cnt = 0 # 判断是否early stop的数值
    save_model = False # 判断是否保存模型
    epochs = 1000
    train_loss_list = []
    eval_loss_list = []
    for epoch in range(epochs):
        print('\nEpoch {} / {}'.format(epoch+1,epochs))
        # 训练
        model.train()
        train_loss = 0.0
        for data,label in trainloader:
            pred = model(data.float())
            loss = loss_fn(pred,label.float())
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss_list.append(train_loss.item()/len(trainloader)) # 每个epoch中训练集的loss
        # 验证
        model.eval()
        eval_loss = 0.0
        for data,label in devloader:
            pred = model(data.float())
            loss = loss_fn(pred,label.float())
            eval_loss += loss
        eval_loss_list.append(eval_loss.item()/len(devloader)) # 每个epoch中验证集的loss
        print('train loss: ', train_loss.item()/len(trainloader),'| dev loss: ', eval_loss.item()/len(devloader))
        if eval_loss < min_valid_loss: #验证集loss改善则保存模型
            min_valid_loss = eval_loss
            early_stop_cnt = 0
            print('save model')
            torch.save(model.state_dict(), path)
            save_model = True
        # loss没有update, increment cnt
        early_stop_cnt += 1
        if early_stop_cnt >= 10 and save_model == True: #验证集loss在10次epoch没有改善则early stop
            print('early stop')
            break
        elif early_stop_cnt >= 10 and save_model == False:
            print('early stop and save model')
            torch.save(model.state_dict(),path)
            save_model = True
            break
    plt.figure(figsize=(20,12))
    plt.plot(train_loss_list,label='train loss')
    plt.plot(eval_loss_list,label='dev loss')
    plt.legend()

def layering(alpha,origin, sample_date):
    '''
    按因子值给每个截面的个股收益率排序且分层
    '''
    layer_size = 5 # 分5层
    layer=[[] for i in range(layer_size)]
    alpha = np.nan_to_num(alpha)
    for i in range(len(alpha)):
        alpha_daily = pd.DataFrame(alpha[i,...],columns=['return']) # 每天每股因子值
        alpha_daily = alpha_daily.sort_values(by='return',ascending=False,na_position='last') # 按照因子值排序
        rank_idx = list(alpha_daily.index)
        return_daily = pd.DataFrame(origin[i,...],columns=['return'])
        return_daily = return_daily.reindex(rank_idx)
        return_daily_cent = return_daily - return_daily.mean() # 减去单个截面大盘对收益率的影响
        split = np.array_split(return_daily_cent,layer_size) # 分层
        for j in range(layer_size):
            mean_ = split[j]['return'].mean()
            layer[j].append(mean_)
    for i in range(layer_size):
        layer[i] = np.nancumsum(layer[i]) # 每层累积收益
    plt.figure(figsize=(15,6))
    plt.plot(sample_date,layer[0],label = "top")
    plt.plot(sample_date,layer[4],label = "bot")
    plt.legend()
    return

def RankIC_mean_std(original,pred,sample_date):
    '''
    计算rankic均值和std
    '''
    rankic = []
    for i in range(len(original)): # 每个截面
        ori_cross = original[i] # 该截面所有股票真实收益率
        pred_cross = pred[i] # 该截面所有股票因子值
        rankic.append(pd.DataFrame([ori_cross.reshape(-1),pred_cross.reshape(-1)]).T.corr(method='spearman').iloc[0,1])
    rankic_mean = np.nanmean(rankic)
    rankic_std = np.nanstd(rankic)
    rankic_cumsum = np.nancumsum(rankic)
    plt.figure(figsize=(15,6))
    plt.plot(sample_date,rankic_cumsum,label='rankic')
    plt.legend()
    return rankic_mean,rankic_std,rankic_cumsum

def normalize_alpha(pred):
    mean = np.nanmean(pred,axis=(1,2)).reshape(-1,1,1)
    std  = np.nanstd(pred,axis=(1,2)).reshape(-1,1,1)
    norm = (pred - mean) / (std + 1e-5)
    return norm

def test(testloader,origin, nan_pair,sample_date,model, loss_fn):
    model.eval()
    test_loss_list = []
    alpha = []
    with torch.no_grad():
        test_loss = 0.0
        for data,label in testloader:
            pred = model(data.float())
            loss = loss_fn(pred,label.float())
            test_loss += loss
            alpha.append(pred.cpu().numpy())
        test_loss_list.append(test_loss.item()/len(testloader))
        print('test loss: ', test_loss.item()/len(testloader))
    arr = np.vstack(alpha)
    pred = reconstruct_alpha(arr,origin,nan_pair)
    pred = normalize_alpha(pred)
    rankic_mean,rankic_std,rankic_cumsum = RankIC_mean_std(origin,pred,sample_date)
    print('RankIC mean: {} | RankIC std {}'.format(rankic_mean,rankic_std))
    layering(pred,origin, sample_date)

if __name__ == '__main__':
    path5 = './model_param.pth' # 模型保存文件
    train(tr_loader5,dev_loader5,model,loss_fn,optimizer,path5)
    model.load_state_dict(torch.load(path5)) # 加载pre-trained模型
    test(wholeloader2020,Y_5[473:],nan2020,sample_date[473:],model,loss_fn)


    