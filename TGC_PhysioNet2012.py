import numpy as np
import pypots.imputation
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
from pypots.data import load_specific_dataset
from pypots.imputation import SAITS
from pypots.imputation.csdi import CSDI
from pypots.imputation.brits import BRITS
from pypots.imputation.mrnn import MRNN
from pypots.imputation.gpvae import GPVAE
from pypots.imputation.transformer import Transformer
from pypots.imputation.timesnet import TimesNet
from pypots.utils.metrics import calc_mae,calc_rmse,calc_mre
import pypots.imputation.locf as locf
import pypots.imputation.mean as pymean
import pypots.imputation.median as pymedian
import random
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeSVD, BiScaler,MatrixFactorization
import pypots.imputation.frets as FRET
from pypots.imputation import koopa
from pypots.imputation import crossformer
from pypots.imputation import SCINet
from pypots.imputation import film
import pandas as pd
from gcimpute.gaussian_copula import GaussianCopula
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(42)
# Data preprocessing. Tedious, but PyPOTS can help.
data = load_specific_dataset('physionet_2012')
# X = data
# num_samples = len(X['RecordID'].unique())
# X = X.drop(['RecordID', 'Time'], axis = 1)
# X = StandardScaler().fit_transform(X.to_numpy())
# X = X.reshape(num_samples, 48, -1)
X = np.concatenate([data['train_X'],data['val_X'],data['test_X']],axis=0)
col=[]
X=np.delete(X,18,axis=2)
drop=[]
for i in range(X.shape[0]):
    if np.sum(np.isnan(X[i, :, :]))==X.shape[1]*X.shape[2]:
        drop.append(i)
if len(drop)!=0:
    X = np.delete(X, drop, axis=0)
print("drop list:",drop)
print(X.shape)
def tableCreate(X):
    empty = []
    col = []
    for j in range(X.shape[2]):
        for i in range(X.shape[1]):
            a = X[:, i, j]
            na = np.sum(~np.isnan(a))
            if na >= 1:
                col.append(j)
            uniques, counts = np.unique(a.reshape(-1), return_counts=True)
            if len(counts) <= 2:
                empty.append((i, j))
    for (i, j) in empty:
        flag = 0
        while flag <= 10:
            random_int = random.randint(0, X.shape[0] - 1)
            while X[random_int, i, j] == X[random_int, i, j]:
                random_int = random.randint(0, X.shape[0] - 1)
            seq = X[random_int, :, j].reshape(-1)
            uniques, counts = np.unique(seq, return_counts=True)
            if len(counts) == 1:
                continue
            else:
                flag += 1
                li = list(seq[~np.isnan(seq)])
                l = len(li)
                X[random_int, i, j] = list(seq[~np.isnan(seq)])[random.randint(0, l - 1)] + random.gauss(0, 0.001)
    X = X.astype('float32')
    mat = []
    for i in range(X.shape[1]):
        mat.append(X[:, i, :])
    table = np.concatenate(mat, axis=1)
    return table
def train_and_test_model(model,train_dataset,val_dataset,test_dataset,indicating_mask,test_X_ori,model_name,result):
    model.fit(train_dataset)
    imputation =model.predict(test_dataset)
    if len(imputation['imputation'].shape)==4:
        imputation['imputation']=imputation['imputation'].reshape(imputation['imputation'].shape[0],imputation['imputation'].shape[2],imputation['imputation'].shape[3])
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
def train_and_test(X,result,random_state=42,mask_rate=0.2,max_iter=5,save_dataset=False,if_GC_Impute=True,if_deep_learning=True,if_simple_impute=True,if_mat_method=True,if_ablation_experiment=True,if_new_model=True):
    print("="*10,"随机参数：",random_state,"遮掩率：",mask_rate,"最大迭代数：",max_iter,"="*10)
    set_seed(random_state)
    per = np.random.permutation(X.shape[0])
    #划分训练集，测试集
    train_X,val_X,test_X=X[per[:int(0.8*0.8*X.shape[0])],:,:],X[per[int(0.8*0.8*X.shape[0]):int(0.8*X.shape[0])],:,:],X[per[int(0.8*X.shape[0]):],:,:]
    val_X_ori = val_X  # keep X_ori for validation
    test_X_ori = test_X  # keep X_ori for validation

    #随机遮掩
    val_X = mcar(val_X, mask_rate)  # randomly hold out 10% observed values as ground truth
    test_X=mcar(test_X,mask_rate)
    indicating_mask = np.isnan(test_X) ^ np.isnan(test_X_ori)
    #数据包装
    train_X_Comb = np.concatenate([train_X, val_X_ori], axis=0)
    train_dataset_Comb = {"X": train_X_Comb}
    val_dataset={"X":val_X,"X_ori":val_X_ori}
    test_dataset={"X":test_X}
    alldata={"dataset":[train_dataset_Comb,val_dataset,test_dataset],"X_ori":[val_X_ori,test_X_ori]}
    #保存
    if save_dataset:
        import pickle
        f_save = open(r'.\data0.6missing.pkl', 'wb')
        pickle.dump(alldata, f_save)
        f_save.close()
    if if_simple_impute:
        # simple impute
        t=train_X_Comb
        lt=t.shape[0]
        a=np.concatenate([t,test_X],axis=0)
        # locf
        model_name = "locf"
        imputation = {"imputation": locf.locf_numpy(a)[lt:,:,:]}
        simple_impute(imputation=imputation, train_dataset=None, val_dataset=val_dataset,
                          test_dataset=test_dataset,
                          indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        # locf
        model_name = "mean"
        model = pymean.Mean()
        imputation = {"imputation":model.predict({"X":a})["imputation"][lt:,:,:]}
        simple_impute(imputation=imputation, train_dataset=None, val_dataset=val_dataset,
                          test_dataset=test_dataset,
                          indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        # locf
        model_name = "median"
        model = pymedian.Median()
        imputation = {"imputation":model.predict({"X":a})["imputation"][lt:,:,:]}
        simple_impute(imputation=imputation, train_dataset=None, val_dataset=val_dataset,
                          test_dataset=test_dataset,
                          indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
    # Model training.
    if if_deep_learning:
        # Model training.
        #saits
        model = SAITS(n_steps=X.shape[1], n_features=X.shape[2], n_layers=2, d_model=256, d_ffn=128, n_heads=4, d_k=64, d_v=64, dropout=0.1, epochs=max_iter)
        model_name="SAITS"
        result=train_and_test_model(model=model,train_dataset=train_dataset_Comb,val_dataset=val_dataset,test_dataset=test_dataset,
                             indicating_mask=indicating_mask,test_X_ori=test_X_ori,model_name=model_name,result=result)
        #timenet
        model =TimesNet(n_steps=X.shape[1], n_features=X.shape[2], n_layers=2, d_model=16, dropout=0.1, epochs=max_iter,top_k=3,d_ffn=32,n_kernels=5)
        model_name = "TimesNet"
        result=train_and_test_model(model=model, train_dataset=train_dataset_Comb, val_dataset=val_dataset, test_dataset=test_dataset,
                             indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        #BRITS
        model=BRITS(n_steps=X.shape[1], n_features=X.shape[2],  epochs=max_iter,rnn_hidden_size=256)
        model_name="BRITS"
        result=train_and_test_model(model=model, train_dataset=train_dataset_Comb, val_dataset=val_dataset, test_dataset=test_dataset,
                             indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        #MRNN
        model = MRNN(n_steps=X.shape[1],n_features=X.shape[2],rnn_hidden_size=64,epochs=max_iter)
        model_name = "MRNN"
        result=train_and_test_model(model=model, train_dataset=train_dataset_Comb, val_dataset=val_dataset, test_dataset=test_dataset,
                             indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        #GPVAE
        model = GPVAE(n_steps=X.shape[1],n_features=X.shape[2],latent_size=35,encoder_sizes=(128,128),decoder_sizes=(256,256),
                      window_size=24,sigma=1.005,length_scale=7.0,beta=0.2,epochs=max_iter)
        model_name = "GPVAE"
        result=train_and_test_model(model=model, train_dataset=train_dataset_Comb, val_dataset=val_dataset, test_dataset=test_dataset,
                             indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
    if if_GC_Impute:
        #展开
        test_mat=[]
        #合并训练和验证（高斯无需验证）
        train_X_GC=np.concatenate([train_X,val_X_ori],axis=0)
        train_table=tableCreate(train_X_GC)
        for i in range(train_X_GC.shape[1]):
            test_mat.append(test_X[:,i,:])
        test_table=np.concatenate(test_mat,axis=1)
        model = GaussianCopula(verbose=2,max_iter=max_iter)
        model.fit(X=train_table)
        #import pandas as pd
        #tableall = pd.DataFrame(model.get_params()['copula_corr'], columns=[str(i)+c for i in range(train_X_GC.shape[1]) for c in characters],index=[str(i)+c for i in range(train_X_GC.shape[1]) for c in characters])
        #tableall.to_csv(r'E:\MMIC\文献\复现论文\Gauss Copula Imputer\复现代码\gcimpute-master\corr\PhysioNet2012.csv', index=True,header=True)
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
        t = train_X_Comb
        lt = t.shape[0]
        a = np.concatenate([t, test_X], axis=0)
        test_mat = []
        for i in range(a.shape[1]):
            test_mat.append(a[:,i,:])
        test_table = np.concatenate(test_mat, axis=1)
        #KNN
        model_name="KNN"
        imp = KNN(k=2, print_interval=3*7282).fit_transform(test_table)[lt:,:]
        impute = np.zeros_like(test_X)
        for i in range(test_X.shape[1]):
            impute[:,i,:]=imp[:,i*test_X.shape[2]:(i+1)*test_X.shape[2]]
        imputation = {"imputation": impute}
        simple_impute(imputation=imputation, train_dataset=None, val_dataset=val_dataset,
                      test_dataset=test_dataset,
                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        # SoftImpute
        model_name = "SoftImpute"
        imp = SoftImpute(max_iters=5).fit_transform(test_table)[lt:,:]
        impute = np.zeros_like(test_X)
        for i in range(test_X.shape[1]):
            impute[:, i, :] = imp[:, i * test_X.shape[2]:(i + 1) * test_X.shape[2]]
        imputation = {"imputation": impute}
        simple_impute(imputation=imputation, train_dataset=None, val_dataset=val_dataset,
                      test_dataset=test_dataset,
                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        #IterativeSVD
        model_name = "IterativeSVD"
        imp = IterativeSVD(max_iters=5).fit_transform(test_table)[lt:,:]
        impute = np.zeros_like(test_X)
        for i in range(test_X.shape[1]):
            impute[:, i, :] = imp[:, i * test_X.shape[2]:(i + 1) * test_X.shape[2]]
        imputation = {"imputation": impute}
        simple_impute(imputation=imputation, train_dataset=None, val_dataset=val_dataset,
                      test_dataset=test_dataset,
                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
        # #MatrixFactorization
        # model_name = "MatrixFactorization"
        # imp = MatrixFactorization(max_iters=5).fit_transform(test_table)[lt:,:]
        # impute = np.zeros_like(test_X)
        # for i in range(test_X.shape[1]):
        #     impute[:, i, :] = imp[:, i * test_X.shape[2]:(i + 1) * test_X.shape[2]]
        # imputation = {"imputation": impute}
        # simple_impute(imputation=imputation, train_dataset=None, val_dataset=val_dataset,
        #               test_dataset=test_dataset,
        #               indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name, result=result)
    if if_ablation_experiment:
        # TGC-GC
        # 展开
        test_mat = []
        # 合并训练和验证（高斯无需验证）
        train_X_GC = np.concatenate([train_X, val_X_ori], axis=0)
        train_table = tableCreate(train_X_GC)
        for i in range(train_X_GC.shape[1]):
            test_mat.append(test_X[:, i, :])
        test_table = np.concatenate(test_mat, axis=1)
        t = np.concatenate([train_table, test_table], axis=0)
        lt = train_table.shape[0]
        # IterativeSVD
        model_name = "IterativeSVD"
        imp = IterativeSVD(max_iters=5).fit_transform(t)[lt:, :]
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
        result = train_and_test_model(model=model, train_dataset=train_dataset_Comb, val_dataset=val_dataset,
                                      test_dataset=test_dataset,
                                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name,
                                      result=result)
        model = koopa.Koopa(n_steps=X.shape[1], n_features=X.shape[2], epochs=max_iter,n_seg_steps=X.shape[1]//2,n_blocks=3,d_dynamic=256,d_hidden=512,n_hidden_layers=3,)
        model_name = "Koopa"
        result = train_and_test_model(model=model, train_dataset=train_dataset_Comb, val_dataset=val_dataset,
                                      test_dataset=test_dataset,
                                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name,
                                      result=result)
        model = crossformer.Crossformer(n_steps=X.shape[1], n_features=X.shape[2], epochs=max_iter,n_layers=3,d_model=256,n_heads=4,dropout=0.2,d_ffn=512,win_size=2,seg_len=6,factor=10)
        model_name = "Crossformer"
        result = train_and_test_model(model=model, train_dataset=train_dataset_Comb, val_dataset=val_dataset,
                                      test_dataset=test_dataset,
                                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name,
                                      result=result)
        model = SCINet(n_steps=X.shape[1], n_features=X.shape[2], epochs=max_iter,n_stacks=1,n_levels=3,n_groups=1,n_decoder_layers=1,d_hidden=1,kernel_size=5)
        model_name = "SCINet"
        result = train_and_test_model(model=model, train_dataset=train_dataset_Comb, val_dataset=val_dataset,
                                      test_dataset=test_dataset,
                                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name,
                                      result=result)
        model = film.FiLM(n_steps=X.shape[1], n_features=X.shape[2], epochs=max_iter,window_size=[256],multiscale=[1,2,4],)
        model_name = "FiLM"
        result = train_and_test_model(model=model, train_dataset=train_dataset_Comb, val_dataset=val_dataset,
                                      test_dataset=test_dataset,
                                      indicating_mask=indicating_mask, test_X_ori=test_X_ori, model_name=model_name,
                                      result=result)

    print("="*13,"运行报告","="*13)
    print(result)
    print("="*30)
#参数设置
random_states=[42,3366,28]
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
    for i,j in item.items():
        print(i)
        print(j['mean'])
        print(j['std'])
        tablemean.append(np.array(j['mean']).reshape(1,-1))
        tablestd.append(np.array(j['std']).reshape(1,-1))
    tablemeans=np.concatenate(tablemean,axis=0)
    tablestds=np.concatenate(tablestd,axis=0)
    np.savetxt("PhysioNet2012\\mean"+str(key)+".csv",tablemeans, delimiter=",")
    np.savetxt("PhysioNet2012\\std" + str(key) + ".csv", tablestds, delimiter=",")

