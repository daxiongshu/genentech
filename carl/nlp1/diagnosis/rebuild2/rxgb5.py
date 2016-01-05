from sklearn.linear_model import SGDClassifier
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
import numpy as np
import gc
from scipy import sparse
mem = Memory("./mycache")

@mem.cache
def get_data(name):
    data = load_svmlight_file(name)
    return data[0], data[1]

X, y = get_data('train.svm')
print X.shape, y.shape, np.mean(y)
X1,_=get_data('../../procedure/rebuild2/train.svm')

X=sparse.hstack((X,X1),format='csr')
del X1
gc.collect()

Xt, _ = get_data('test.svm')
X1,_=get_data('../../procedure/rebuild2/test.svm')
Xt=sparse.hstack((Xt,X1),format='csr')
del X1
gc.collect()

from sklearn.cross_validation import KFold
import gc
import pandas as pd
import numpy as np
from xgb_classifier import xgb_classifier
import pickle
from scipy import sparse
import sys
from sklearn.metrics import roc_auc_score, f1_score, log_loss, make_scorer,auc,roc_curve
def myauc(y,pred):
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    return auc(fpr, tpr)
train=pd.read_csv('../../../input/patients_train.csv',index_col='patient_id')
idx=np.array(train.index)
del train
if False:
    kf=KFold(len(y),n_folds=4)

    for train_index, test_index in kf:
        Xt=X[test_index]
        X=X[train_index]
        idx=idx[test_index]
        yt=y[test_index]
        y=y[train_index]
        break
test=pd.read_csv('../../../input/patients_test.csv',index_col='patient_id')
idx=np.array(test.index)
del test
gc.collect()
print X.shape,y.shape,Xt.shape

from xgb_classifier import xgb_classifier
eta=0.1
myname=sys.argv[0]
for seed in [0]:#[i*777 for i in range(1,10)]:
    for depth in [10]:
        for child in [2]:
            for col in [0.4]:
                for sub in [1]:
                    for num in [2000]:
                        clf=xgb_classifier(eta=eta,min_child_weight=child,depth=depth,num_round=num,col=col,subsample=sub,seed=seed)
                        ypred=clf.train_predict(X,y,Xt)        
                        s=pd.DataFrame({'patient_id':idx,'predict_screener':ypred})     
                        s.to_csv('rxgb5.csv',index=False)           
                        #s.to_csv('va_result/%s_eta_%f_depth_%d_child_%d_col_%f_sub_%f_num_%d_seed_%d_score_%f'% (myname,eta,depth,child,col,sub,num,seed,score),index=False)

