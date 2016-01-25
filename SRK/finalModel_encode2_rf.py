import sys
import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn import ensemble
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import scipy as sp
import cPickle as pkl
from scipy.sparse import csr_matrix, hstack


np.random.seed(123456)
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Dropout
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import regularizers
from keras.layers.advanced_activations import PReLU, ELU



def runXGB(train_X, train_y):
        params = {}
        params["objective"] = "binary:logistic"
        params['eval_metric'] = 'auc'
        params["eta"] = 0.1
        params["min_child_weight"] = 10
        params["subsample"] = 0.8
        params["colsample_bytree"] = 0.8 
        params["silent"] = 1
        params["max_depth"] = 10 
        params["seed"] = 0
        num_rounds = 650

        plst = list(params.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)
	model = xgb.train(plst, xgtrain, num_rounds)

	return model

def predictXGB(test_X, model):
	xgtest = xgb.DMatrix(test_X)
        pred_test_y = model.predict(xgtest)
        return pred_test_y



def runRF(train_X, train_y):
        n_est_val=80
        depth_val=30
        split_val=10
        leaf_val=10
        feat_val= 0.05
        jobs_val=-1
        random_state_val=0
        clf = ensemble.RandomForestClassifier(
                n_estimators = n_est_val,
                max_depth = depth_val,
                min_samples_split = split_val,
                min_samples_leaf = leaf_val,
                max_features = feat_val,
                n_jobs = jobs_val,
                random_state = random_state_val)
        clf.fit(train_X, train_y)
	return clf

def predictRF(test_X, clf):
	pred_test_y = clf.predict_proba(test_X)[:,1]
	return pred_test_y


def runNN(train_X, train_y):
        sc = preprocessing.StandardScaler()
        train_X = sc.fit_transform(train_X)
        #test_X = sc.transform(test_X)

        model = Sequential()
        #model.add(Dropout(0.2))

        model.add(Dense(100, input_shape=(train_X.shape[1],), init='he_uniform', W_regularizer=regularizers.l1(0.001)))
        model.add(Activation('relu'))
        #model.add(ELU())
        model.add(Dropout(0.3))
        #model.add(BatchNormalization())

        model.add(Dense(100, init='he_uniform'))
        model.add(Activation('relu'))
        #model.add(ELU())
        model.add(Dropout(0.3))
        #model.add(BatchNormalization())

        model.add(Dense(100, init='he_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(50, init='he_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(50, init='he_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(1, init='he_uniform'))
        model.add(Activation('sigmoid'))

        #sgd_opt = SGD(lr=0.01)
        model.compile(loss='binary_crossentropy', optimizer='adagrad')

        for i in xrange(1):
                model.fit(train_X, train_y, batch_size=256, nb_epoch=44, validation_split=0.05, verbose=2, shuffle=True)

	return model, sc


def predictNN(test_X, model, sc):
		test_X = sc.transform(test_X)
                preds = model.predict(test_X, verbose=0)
        	return preds.ravel()

def getSelectIndices(dev_X, select_indices):
        dev_X = dev_X.tocsc()
        dev_X = dev_X[:,select_indices]
        return dev_X.todense()






def getTrainData():
	sp_array = pkl.load(open("./FinalPkl/train_mod_v2_sparse.pkl"))
	print sp_array.shape
	sp2_array = pkl.load(open("./FinalPkl/train_mod_v4_sparse.pkl"))
	print sp2_array.shape
	sp3_array = pkl.load(open("./FinalPkl/train_mod_v5_sparse.pkl"))
	print sp3_array.shape
	sp4_array = pkl.load(open("./Encode2/train_sparse.pkl"))
        print sp4_array.shape

	sp_array = hstack([sp_array,sp2_array,sp3_array,sp4_array])
	return sp_array.tocsr()

def getTestData():
        sp_array = pkl.load(open("./FinalPkl/test_mod_v2_sparse.pkl"))
        print sp_array.shape
        sp2_array = pkl.load(open("./FinalPkl/test_mod_v4_sparse.pkl"))
        print sp2_array.shape
        sp3_array = pkl.load(open("./FinalPkl/test_mod_v5_sparse.pkl"))
        print sp3_array.shape
	sp4_array = pkl.load(open("./Encode2/test_sparse.pkl"))
        print sp4_array.shape

        sp_array = hstack([sp_array,sp2_array,sp3_array,sp4_array])
        return sp_array.tocsr()

if __name__ == "__main__":
        print "Reading files.."
	train = getTrainData()
        print train.shape 

        print "Getting DV and ID.."
        train_y  = pd.read_csv("../Data/patients_train_mod.csv").is_screener.values

        print "Cross validating.."
	model = runRF(train, train_y)

	del train
	del train_y
	import gc
	gc.collect()

	print "Reading test files.."
	test = getTestData()
	print test.shape
	test_id = pd.read_csv("../Data/patients_test.csv").patient_id.values
	test_preds = predictRF(test, model)

	preds_df = pd.DataFrame({'patient_id':test_id, 'predict_screener':test_preds})
	preds_df.to_csv("sub_rf_withencode2_dep30_split10_leaf10_tree80_seed0_srk.csv", index=False)
	

