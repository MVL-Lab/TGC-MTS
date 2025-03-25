import copy
from torch._dynamo import allow_in_graph
import torch
from pypots.imputation import SAITS
from pypots.imputation.csdi import CSDI
from pypots.imputation.brits import BRITS
from pypots.imputation.mrnn import MRNN
from pypots.imputation.gpvae import GPVAE
from pypots.imputation.transformer import Transformer
from pypots.imputation.timesnet import TimesNet
import pypots.imputation.locf as locf
import pypots.imputation.mean as pymean
import pypots.imputation.median as pymedian
import pickle
import numpy as np
from gcimpute.helper_evaluation import get_mre
from gcimpute.helper_mask import mask_MCAR
from tqdm import tqdm
import random
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
from pypots.utils.metrics import calc_mae,calc_rmse,calc_mre
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeSVD, BiScaler,MatrixFactorization,IterativeImputer
from missingpy import MissForest
import pypots.imputation.frets as FRET
from pypots.imputation import koopa
from pypots.imputation import crossformer
from pypots.imputation import SCINet
from pypots.imputation import film
from pypots.imputation import usgan
#高斯关联
from gcimpute.gaussian_copula import GaussianCopula
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(42)
T=31
#从三维张量转回字典
def toList(data,result,feat,user,flag):
    for i in range(len(list(data.keys()))):
        for j in range(len(feat)):
            for t in range(T):
                if data[user[i]][feat[j]][t] != data[user[i]][feat[j]][t]:
                    #print(flag)
                    data[user[i]][feat[j]][t] = result[t, i, j]
    return data
def toList2(data,result,feat,user,flag):
    for i in range(len(list(data.keys()))):
        for j in range(len(feat)):
            for t in range(T):
                if data[user[i]][feat[j]][t] != data[user[i]][feat[j]][t]:
                    #print(flag)
                    data[user[i]][feat[j]][t] = result[i, t, j]
    return data
#变成可以比较的形式
def dictotable(X_mask,feat,ids,data_imp):
    X_imp=np.zeros_like(X_mask)
    for i in range(len(X_mask)):
        t=X_mask[i][-1]
        id=ids[i]
        for j in range(len(feat)):
            X_imp[i,j]=data_imp[id][feat[j]][int(t)]
    return X_imp
def preData(data,feat):
    # 重构
    T = 31
    ids = []
    X=[]
    Xs = np.zeros((len(data),T,len(feat)))
    for (i,(id, patient)) in enumerate(data.items()):
        for t in range(T):
            row = [patient[f][t] for f in feat]
            row.append(t)
            X.append(row)
            ids.append(id)
            for f in range(len(feat)):
                Xs[i,t,f]=patient[feat[f]][t]
    X = np.array(X).astype("float32")
    Xs = Xs.astype("float32")
    rej=[]
    for i in range(T):
        for j in range(len(feat)):
            cross=Xs[:,i,j].reshape(-1)
            if sum(list(cross==0))==0:
                rej.append(j)
    print(X.shape)
    X = np.delete(X, rej, 1)
    print(X.shape)
    dropFeat=[]
    for i in rej:
        dropFeat.append(feat[i])
    for i in dropFeat:
        feat.remove(i)
    lastX=X.copy()
    X[X==0]=np.nan
    ids=np.array(ids)
    ids.reshape(-1,1)
    X_mask1=X[:,:-1]
    X_mask1 = StandardScaler().fit_transform(X_mask1)
    Xresult=np.zeros((int(X_mask1.shape[0]/T),T,X_mask1.shape[1]))
    for i in range(int(X_mask1.shape[0]/T)):
        mat=X_mask1[i*T:(i+1)*T,:]
        Xresult[i,:,:]=mat
    print(Xresult.shape)
    return Xresult

def erroPrint(X_imp,X,X_mask,kind):
    # Evaluation: compute the scaled-MAE (SMAE) for each data type (scaled by MAE of median imputation)
    smae = get_mre(X_imp, X,X_mask)
    print(kind+f': mae {smae:.3f}')
    return smae
#回复mask数据
def recovery(X_mask,ids,feat):
    data = {}
    T = 31
    nameList=list(set(ids))
    for name in nameList:
       data[name]={}
       for f in feat:
           data[name][f]=[-1 for i in range(T)]
    for id in tqdm(data.keys()):
        for i in range(len(ids)):
            if id==ids[i]:
                t=X_mask[i][-1]
                for j in range(len(feat)):
                    data[id][feat[j]][int(t)]=X_mask[i][j]
    pid=[]
    for id in data.keys():
        negativeOne=[]
        for f in feat:
            lists=data[id][f]
            for i in range(len(lists)):
                if lists[i]==-1:
                    negativeOne.append(i)
                    continue
        if len(negativeOne)>0:
            minLong=min(negativeOne)
        else:
            continue
        if minLong<10:
            pid.append(id)
            continue
        for f in range(len(feat)):
            data[id][feat[f]]=data[id][feat[f]][:minLong]
    for p in pid:
        data.pop(p)
    return data
#回复mask数据
def recovery2(X_mask,ids,feat):
    data = {}
    T = 31
    nameList=list(set(ids))
    for name in nameList:
       data[name]={}
       for f in feat:
           data[name][f]=[-1 for i in range(T)]
    for id in tqdm(data.keys()):
        for i in range(len(ids)):
            if id==ids[i]:
                t=X_mask[i][-1]
                for j in range(len(feat)):
                    data[id][feat[j]][int(t)]=X_mask[i][j]
    return data
#原数据获取
filename="./data/FiledData/mimic3hourly.pkl"
with open(filename, 'rb') as f:
    data = pickle.load(f)
feat=['glucose','lactate','potassium','chloride','hematocrit','hemoglobin','sodium','aniongap','bicarbonate','bun','creatinine',
         'wbc','pt','bilirubin','heart_rate','sysbp','diasbp','meanbp','respratory','temperature','spo2']
print(len(data[30146566]['outcome']))
X=preData(data,feat)
print(X.shape)
def train_and_test_model(model,train_dataset,val_dataset,test_dataset,indicating_mask,test_X_ori,model_name,result):
    if model_name in ["SCINet","Koopa"]:
        pret=train_dataset["X"]
        pree=test_dataset["X"]
        a=np.zeros((train_dataset["X"].shape[0],1,train_dataset["X"].shape[2]))
        train_dataset["X"]=np.concatenate([train_dataset["X"],a],axis=1)
        b = np.zeros((test_dataset["X"].shape[0], 1, test_dataset["X"].shape[2]))
        test_dataset["X"]=np.concatenate([test_dataset["X"],b],axis=1)
    model.fit(train_dataset)
    imputation =model.predict(test_dataset)
    if len(imputation['imputation'].shape)==4:
        imputation['imputation']=imputation['imputation'].reshape(imputation['imputation'].shape[0],imputation['imputation'].shape[2],imputation['imputation'].shape[3])
    if model_name in ["SCINet", "Koopa"]:
        imputation['imputation']=imputation['imputation'][:,:-1,:]
        train_dataset["X"]=pret
        test_dataset["X"]=pree
    mre = calc_mre(imputation['imputation'], np.nan_to_num(test_X_ori), indicating_mask)
    result[model_name]['mre'].append(mre)
    mae = calc_mae(imputation['imputation'], np.nan_to_num(test_X_ori), indicating_mask)
    result[model_name]['mae'].append(mae)
    rmse = calc_rmse(imputation['imputation'], np.nan_to_num(test_X_ori), indicating_mask)
    result[model_name]['rmse'].append(rmse)
    return result
def simple_impute(imputation,train_dataset,val_dataset,test_dataset,indicating_mask,test_X_ori,model_name,result):
    if len(imputation['imputation'].shape)==4:
        imputation['imputation']=imputation['imputation'].reshape(imputation['imputation'].shape[0],imputation['imputation'].shape[2],imputation['imputation'].shape[3])
    mre = calc_mre(np.nan_to_num(imputation['imputation']), np.nan_to_num(test_X_ori), indicating_mask)
    result[model_name]['mre'].append(mre)
    mae = calc_mae(np.nan_to_num(imputation['imputation']), np.nan_to_num(test_X_ori), indicating_mask)
    result[model_name]['mae'].append(mae)
    rmse = calc_rmse(np.nan_to_num(imputation['imputation']), np.nan_to_num(test_X_ori), indicating_mask)
    result[model_name]['rmse'].append(rmse)
    return result
def train_and_test(X,result,random_state=42,mask_rate=0.2,max_iter=5,save_dataset=False,if_GC_Impute=True,if_deep_learning=True,if_simple_impute=True,if_mat_method=True,if_ablation_experiment=False,if_new_model=True):
    print("="*10,"随机参数：",random_state,"遮掩率：",mask_rate,"最大迭代数：",max_iter,"="*10)
    set_seed(random_state)
    per = np.random.permutation(X.shape[0])
    #划分训练集，测试集
    train_X,val_X,test_X=X[per[:int(0.8*X.shape[0])],:,:],X[per[int(0.8*0.8*X.shape[0]):int(0.8*X.shape[0])],:,:],X[per[int(0.8*X.shape[0]):],:,:]
    val_X_ori = val_X  # keep X_ori for validation
    test_X_ori = test_X  # keep X_ori for validation

    #随机遮掩
    val_X = mcar(val_X, mask_rate)  # randomly hold out 10% observed values as ground truth
    test_X=mcar(test_X,mask_rate)
    indicating_mask = np.isnan(test_X) ^ np.isnan(test_X_ori)
    #数据包装
    train_dataset = {"X": train_X}
    val_dataset={"X":val_X,"X_ori":val_X_ori}
    test_dataset={"X":test_X}
    alldata={"dataset":test_dataset,"X_ori":test_X_ori}
    #保存
    if save_dataset:
        import pickle
        f_save = open('E:\\MMIC\\文献\\复现论文\\Gauss Copula Imputer\\复现代码\\gcimpute-master\\MIMIC0.2\\'+str(random_state)+'\\datamissing.pkl', 'wb')
        pickle.dump(alldata, f_save)
        f_save.close()
    if if_simple_impute:
        # simple impute
        t = train_X
        lt = t.shape[0]
        a = np.concatenate([t, test_X], axis=0)
        # locf
        model_name = "locf"
        imputation = {"imputation": locf.locf_numpy(a)[lt:, :, :]}
        simple_impute(imputation=imputation, train_dataset=None, val_dataset=val_dataset,
                      test_dataset=test_dataset,
                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        # locf
        model_name = "mean"
        model = pymean.Mean()
        imputation = {"imputation": model.predict({"X": a})["imputation"][lt:, :, :]}
        simple_impute(imputation=imputation, train_dataset=None, val_dataset=val_dataset,
                      test_dataset=test_dataset,
                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        # locf
        model_name = "median"
        model = pymedian.Median()
        imputation = {"imputation": model.predict({"X": a})["imputation"][lt:, :, :]}
        simple_impute(imputation=imputation, train_dataset=None, val_dataset=val_dataset,
                      test_dataset=test_dataset,
                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
    # Model training.
    if if_deep_learning:
        #saits
        model = SAITS(n_steps=X.shape[1], n_features=X.shape[2], n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, dropout=0.1, epochs=max_iter,d_ffn=128)
        model_name="SAITS"
        result=train_and_test_model(model=model,train_dataset=train_dataset,val_dataset=val_dataset,test_dataset=test_dataset,
                             indicating_mask=indicating_mask,test_X_ori=test_X_ori,model_name=model_name,result=result)
        #timenet
        model =TimesNet(n_steps=X.shape[1], n_features=X.shape[2], n_layers=2, d_model=16, dropout=0.1, epochs=max_iter,top_k=3,d_ffn=32,n_kernels=5)
        model_name = "TimesNet"
        result=train_and_test_model(model=model, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset,
                             indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        #BRITS
        model=BRITS(n_steps=X.shape[1], n_features=X.shape[2],  epochs=max_iter,rnn_hidden_size=256)
        model_name="BRITS"
        result=train_and_test_model(model=model, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset,
                             indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        #MRNN
        model = MRNN(n_steps=X.shape[1],n_features=X.shape[2],rnn_hidden_size=64,epochs=max_iter)
        model_name = "MRNN"
        result=train_and_test_model(model=model, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset,
                             indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        #GPVAE
        model = GPVAE(n_steps=X.shape[1],n_features=X.shape[2],latent_size=35,encoder_sizes=(128,128),decoder_sizes=(256,256),
                      window_size=24,sigma=1.005,length_scale=7.0,beta=0.2,epochs=max_iter)
        model_name = "GPVAE"
        result=train_and_test_model(model=model, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset,
                             indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
    if if_GC_Impute:
        #展开
        train_mat=[]
        test_mat=[]
        #合并训练和验证（高斯无需验证）
        for i in range(train_X.shape[1]):
            train_mat.append(train_X[:,i,:])
            test_mat.append(test_X[:,i,:])
        train_table=np.concatenate(train_mat,axis=1)
        test_table=np.concatenate(test_mat,axis=1)
        model = GaussianCopula(verbose=2,max_iter=max_iter)
        model.fit(X=train_table)
        GCX_imp=model.transform(test_table)
        GCimpute=np.zeros_like(test_X)
        for i in range(test_X.shape[1]):
            GCimpute[:,i,:]=GCX_imp[:,i*test_X.shape[2]:(i+1)*test_X.shape[2]]
        indicating_mask = np.isnan(test_X) ^ np.isnan(test_X_ori)
        mre = calc_mre(GCimpute, np.nan_to_num(test_X_ori), indicating_mask)
        result['GC']['mre'].append(mre)
        mae = calc_mae(GCimpute, np.nan_to_num(test_X_ori), indicating_mask)
        result['GC']['mae'].append(mae)
        rmse = calc_rmse(GCimpute, np.nan_to_num(test_X_ori), indicating_mask)
        result['GC']['rmse'].append(rmse)
    if if_mat_method:
        # simple impute
        # 展开
        train_mat = []
        test_mat = []
        # 合并训练和验证（高斯无需验证）
        # train_X_GC=np.concatenate([train_X,val_X_ori],axis=0)
        for i in range(train_X.shape[1]):
            train_mat.append(train_X[:, i, :])
            test_mat.append(test_X[:, i, :])
        train_table = np.concatenate(train_mat, axis=1)
        test_table = np.concatenate(test_mat, axis=1)
        t = np.concatenate([train_table, test_table], axis=0)
        lt = train_table.shape[0]
        #KNN
        model_name="KNN"
        imp = KNN(k=2, print_interval=3*7282).fit_transform(t)[lt:,:]
        impute = np.zeros_like(test_X)
        for i in range(test_X.shape[1]):
            impute[:,i,:]=imp[:,i*test_X.shape[2]:(i+1)*test_X.shape[2]]
        imputation = {"imputation": impute}
        simple_impute(imputation=imputation, train_dataset=None, val_dataset=val_dataset,
                      test_dataset=test_dataset,
                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        # SoftImpute
        model_name = "SoftImpute"
        imp = SoftImpute(max_iters=5).fit_transform(t)[lt:,:]
        impute = np.zeros_like(test_X)
        for i in range(test_X.shape[1]):
            impute[:, i, :] = imp[:, i * test_X.shape[2]:(i + 1) * test_X.shape[2]]
        imputation = {"imputation": impute}
        simple_impute(imputation=imputation, train_dataset=None, val_dataset=val_dataset,
                      test_dataset=test_dataset,
                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        #IterativeSVD
        model_name = "IterativeSVD"
        imp = IterativeSVD(max_iters=5).fit_transform(t)[lt:,:]
        impute = np.zeros_like(test_X)
        for i in range(test_X.shape[1]):
            impute[:, i, :] = imp[:, i * test_X.shape[2]:(i + 1) * test_X.shape[2]]
        imputation = {"imputation": impute}
        simple_impute(imputation=imputation, train_dataset=None, val_dataset=val_dataset,
                      test_dataset=test_dataset,
                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        #MatrixFactorization
        # model_name = "MatrixFactorization"
        # imp = MatrixFactorization(max_iters=5).fit_transform(t)[lt:,:]
        # impute = np.zeros_like(test_X)
        # for i in range(test_X.shape[1]):
        #     impute[:, i, :] = imp[:, i * test_X.shape[2]:(i + 1) * test_X.shape[2]]
        # imputation = {"imputation": impute}
        # simple_impute(imputation=imputation, train_dataset=None, val_dataset=val_dataset,
        #               test_dataset=test_dataset,
        #               indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
    if if_ablation_experiment:
        #TGC-GC
        # 展开
        train_mat = []
        test_mat = []
        # 合并训练和验证（高斯无需验证）
        for i in range(train_X.shape[1]):
            train_mat.append(train_X[:, i, :])
            test_mat.append(test_X[:, i, :])
        train_table = np.concatenate(train_mat, axis=1)
        test_table = np.concatenate(test_mat, axis=1)
        t=np.concatenate([train_table,test_table],axis=0)
        lt=train_table.shape[0]
        # IterativeSVD
        model_name = "IterativeSVD"
        imp = IterativeSVD(max_iters=5).fit_transform(t)[lt:,:]
        impute = np.zeros_like(test_X)
        for i in range(test_X.shape[1]):
            impute[:, i, :] = imp[:, i * test_X.shape[2]:(i + 1) * test_X.shape[2]]
        imputation = {"imputation": impute}
        simple_impute(imputation=imputation, train_dataset=None, val_dataset=val_dataset,
                      test_dataset=test_dataset,
                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
    if if_new_model:
        model = FRET.FreTS(n_steps=X.shape[1], n_features=X.shape[2], epochs=max_iter)
        model_name = "FreTS"
        result = train_and_test_model(model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                                      test_dataset=test_dataset,
                                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name,
                                      result=result)
        model = koopa.Koopa(n_steps=X.shape[1]+X.shape[1] % 2, n_features=X.shape[2], epochs=max_iter, n_seg_steps=X.shape[1] // 2,
                            n_blocks=3, d_dynamic=256, d_hidden=512, n_hidden_layers=3, )
        model_name = "Koopa"
        result = train_and_test_model(model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                                      test_dataset=test_dataset,
                                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name,
                                      result=result)
        model = crossformer.Crossformer(n_steps=X.shape[1], n_features=X.shape[2], epochs=max_iter, n_layers=3,
                                        d_model=256, n_heads=4, dropout=0.2, d_ffn=512, win_size=2, seg_len=6,
                                        factor=10)
        model_name = "Crossformer"
        result = train_and_test_model(model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                                      test_dataset=test_dataset,
                                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name,
                                      result=result)
        model = SCINet(n_steps=X.shape[1]+X.shape[1] % 2, n_features=X.shape[2], epochs=max_iter, n_stacks=1,n_levels=3,n_groups=1,
                       n_decoder_layers=1, d_hidden=1, kernel_size=5,)
        model_name = "SCINet"
        result = train_and_test_model(model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                                      test_dataset=test_dataset,
                                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name,
                                      result=result)
        model = film.FiLM(n_steps=X.shape[1], n_features=X.shape[2], epochs=max_iter, window_size=[256],
                          multiscale=[1, 2, 4], )
        model_name = "FiLM"
        result = train_and_test_model(model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                                      test_dataset=test_dataset,
                                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name,
                                      result=result)
        model = usgan.USGAN(n_steps=X.shape[1], n_features=X.shape[2], epochs=max_iter,rnn_hidden_size=64)
        model_name = "FiLM"
        result = train_and_test_model(model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                                      test_dataset=test_dataset,
                                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name,
                                      result=result)
    print("="*13,"运行报告","="*13)
    print(result)
    print("="*30)
#参数设置
random_states=[42,3366,28]#,3366,28]
mask_rates=[0.2,0.4,0.6,0.8]
max_iter=5

results={}
for mask_rate in mask_rates:
    result = {"FiLM": {'mae': [], 'mre': [], 'rmse': []},
              "SCINet": {'mae': [], 'mre': [], 'rmse': []},
              "Crossformer": {'mae': [], 'mre': [], 'rmse': []},
              "Koopa": {'mae': [], 'mre': [], 'rmse': []},
              "FreTS": {'mae': [], 'mre': [], 'rmse': []},
              'GC1': {'mae': [], 'mre': [], 'rmse': []},
              'MatrixFactorization': {'mae': [], 'mre': [], 'rmse': []},
              'IterativeSVD': {'mae': [], 'mre': [], 'rmse': []},
              'SoftImpute': {'mae': [], 'mre': [], 'rmse': []},
              'KNN': {'mae': [], 'mre': [], 'rmse': []},
              'locf': {'mae': [], 'mre': [], 'rmse': []},
              'median': {'mae': [], 'mre': [], 'rmse': []},
              'mean': {'mae': [], 'mre': [], 'rmse': []},
              'SAITS': {'mae': [], 'mre': [], 'rmse': []},
              'TimesNet': {'mae': [], 'mre': [], 'rmse': []},
              'CSDI': {'mae': [], 'mre': [], 'rmse': []},
              'BRITS': {'mae': [], 'mre': [], 'rmse': []},
              'MRNN': {'mae': [], 'mre': [], 'rmse': []},
              'GPVAE': {'mae': [], 'mre': [], 'rmse': []},
              'Transformer': {'mae': [], 'mre': [], 'rmse': []},
              'GC': {'mae': [], 'mre': [], 'rmse': []}}
    for random_state in random_states:
        train_and_test(X,result=result,random_state=random_state,mask_rate=mask_rate,max_iter=max_iter,if_deep_learning=False,
                       if_GC_Impute=False,if_simple_impute=False,if_mat_method=False,if_ablation_experiment=False,if_new_model=True)
    result1={}
    for key,loss in result.items():
        mae=np.array(loss['mae'])
        mre = np.array(loss['mre'])
        rmse = np.array(loss['rmse'])
        print(key)
        print('mae:',np.mean(mae),'(+-',np.std(mae),')')
        print('mre:', np.mean(mre), '(+-', np.std(mre), ')')
        print('rmse:', np.mean(rmse), '(+-', np.std(rmse), ')')
        result1[key]={'mean':[np.mean(mae),np.mean(mre),np.mean(rmse)],'std':[np.std(mae),np.std(mre),np.std(rmse)]}
    results[mask_rate]=result1
for key,item in results.items():
    print('loss rate:',key)
    tablemean=[]
    tablestd=[]
    name=[]
    for i,j in item.items():
        print(i)
        print(j['mean'])
        print(j['std'])
        name.append(i)
        tablemean.append(np.array(j['mean']).reshape(1,-1))
        tablestd.append(np.array(j['std']).reshape(1,-1))
    tablemeans=np.concatenate(tablemean,axis=0)
    tablestds=np.concatenate(tablestd,axis=0)
    np.savetxt("MIMIC\\mean"+str(key)+".csv",tablemeans, delimiter=",")
    np.savetxt("MIMIC\\std" + str(key) + ".csv", tablestds, delimiter=",")
    print(name)






