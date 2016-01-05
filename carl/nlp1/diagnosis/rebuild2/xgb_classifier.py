import xgboost as xgb
import numpy as np



class xgb_classifier:
    def __init__(self,eta,min_child_weight,depth,num_round,threads=8,exist_prediction=0,exist_num_round=20,col=1,subsample=1,seed=0):
        self.eta=eta
        self.seed=seed
        self.subsample=subsample
        self.col=col
        self.min_child_weight=min_child_weight
        self.depth=depth
        self.num_round=num_round
        self.exist_prediction=exist_prediction
        self.exist_num_round=exist_num_round  
        self.threads=threads
       
    def train_predict(self,X_train,y_train,X_test,y_test=[]):
        xgmat_train = xgb.DMatrix(X_train, label=y_train,missing=-999)
        test_size = X_test.shape[0]
        param = {}
        param['seed']=self.seed
        param['objective'] = 'binary:logistic'
        param['bst:eta'] = self.eta
        param['colsample_bytree']=self.col
        param['min_child_weight']=self.min_child_weight
        param['bst:max_depth'] = self.depth
        param['subsample']=self.subsample
        param['eval_metric'] = 'auc'
        param['silent'] = 1
        param['nthread'] = self.threads
        plst = list(param.items())

        #watchlist = [ (xgmat_train,'train') ]
        num_round = self.num_round
        if len(y_test):
            xgmat_test = xgb.DMatrix(X_test,missing=-999,label=y_test)
            watchlist = [ (xgmat_train,'train'),(xgmat_test,'test') ]
        else:
            xgmat_test = xgb.DMatrix(X_test,missing=-999)
            watchlist = [ (xgmat_train,'train') ]
    
        bst = xgb.train( plst, xgmat_train, num_round,  watchlist)
        #xgmat_test = xgb.DMatrix(X_test,missing=-999)
    
        if self.exist_prediction:
        # train xgb with existing predictions
        # see more at https://github.com/tqchen/xgboost/blob/master/demo/guide-python/boost_from_prediction.py
       
            tmp_train = bst.predict(xgmat_train, output_margin=True)
            tmp_test = bst.predict(xgmat_test, output_margin=True)
            xgmat_train.set_base_margin(tmp_train)
            xgmat_test.set_base_margin(tmp_test)
            bst = xgb.train(param, xgmat_train, self.exist_num_round, watchlist )

        ypred = bst.predict(xgmat_test)
        return ypred
        
    def train_predict_all_labels(self,X_train,y_train,X_test,predict_y14=True):
        xgb_predict=[]
        for i in range(33) :
            y = y_train[:, i]    
            predicted = None            
            if i==13:            
               if predict_y14:
                  predicted = np.ones(X_test.shape[0]) * 0 
               else:
                  continue
               print "%d is constant like: " % (i)
            else :
               print i, 'train xgb'    
               predicted =self.train_predict(X_train, y,X_test)
            xgb_predict.append(predicted)
        return np.column_stack(xgb_predict)

    def train_predict_label(self,X_train,y_train,X_test,label):
        xgb_predict=[]
        for i in range(label-1,label) :
            y = y_train[:, i]    
            predicted = None            
            if i==13:            
               if predict_y14:
                  predicted = np.ones(X_test.shape[0]) * 0 
               else:
                  continue
               print "%d is constant like: " % (i)
            else :
               print i, 'train xgb'    
               predicted =self.train_predict(X_train, y,X_test)
            xgb_predict.append(predicted)
        return np.column_stack(xgb_predict)


