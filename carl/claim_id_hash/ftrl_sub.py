

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random
import pickle
import sys
import csv
csv.field_size_limit(sys.maxsize)
# TL; DR, the main training process starts on line: 250,
# you may want to start reading the code from there


##############################################################################
# parameters #################################################################
##############################################################################

# A, paths
train=['/home/carl/kaggle/cancer/input/trainx.csv']#,'../encode/encode4/dp_encode.csv']
test=['/home/carl/kaggle/cancer/input/testx.csv']
trainsvm=['../../boat/claim/full_line_claim_id_train_small.svm']

testsvm=['../../boat/claim/full_line_claim_id_test_small.svm']
submission = 'ftrl_sub.csv'  # path of to be outputted submission file

# B, model
alpha = .01  # learning rate
beta = 1   # smoothing parameter for adaptive learning rate
L1 = 0.5     # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 28             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions
length=30
# D, training/validation
epoch = 1       # learn training data for N passes
holdafter = 9   # data after date N (exclusive) are used as validation
holdout = None  # use every N training instance for holdout validation


##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [random() for k in range(D)]#[0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, min(i+length,L)):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(path,pathsvm, D,cc=None):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''
    if len(path)>0:
        readers={}
        for c,name in enumerate(path[1:]):
            readers[c]=DictReader(open(name))
    if len(pathsvm)>0:
        readersvm={}
        for c,name in enumerate(pathsvm):
            readersvm[c]=(open(name))

    readerx=DictReader(open(path[0]), delimiter=',')
    for t, row in enumerate(readerx):
        # process id
        #print row['patient_id']
        #assert(False)
        if cc is not None and t<289440:
            ID,x,y='','',''
            for c,name in enumerate(path[1:]):
                readers[c].next()
            for  c,name in enumerate(pathsvm):
                readersvm[c].readline()
        else:
            try:
                ID=row['patient_id']
                del row['patient_id']
            except:
                pass
        # process clicks
            y = 0.
            target='is_screener'#'IsClick' 
            if target in row:
                if row[target] == '1':
                   y = 1.
                del row[target]

        # extract date

        # turn hour really into hour, it was originally YYMMDDHH

        # build x
            x = []
            tmp=''
            tmp1=[]
            if len(path)>0:
                for c,reader in readers.items():
                    tmp1=tmp1+['%d_%s'%(c,q) for q in reader.next()['doc'].split()]
                for c,reader in readersvm.items():
                    tmp1=tmp1+['%d_%s'%(c,q) for q in  reader.readline().split()[1:]]
            for d,key in enumerate(tmp.split()+tmp1):
                index = abs(hash(key )) % D
                x.append(index)

        yield ID,  x, y


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
print 'start'
# start training
for e in xrange(epoch):
    loss = 0.
    count = 0
    for t,  x, y in data(train,trainsvm, D):  # data is a generator
        count+=1
        p = learner.predict(x)
        loss += logloss(p, y)
        learner.update(x, p, y)
        if count%10000==0:
            #print count,loss/count
            print('%s\tencountered: %d\tcurrent logloss: %f y: %f' % (
                datetime.now(), count, loss/(count),y))

count=0
loss=0
#import pickle
#pickle.dump(learner,open('ftrl3.p','w'))
print 'write result'
##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################
with open(submission, 'w') as outfile:
    outfile.write('patient_id,predict_screener\n')
    for  ID, x, y in data(test,testsvm, D):
        count+=1
        p = learner.predict(x)
        loss += logloss(p, y)

        outfile.write('%s,%s\n' % (ID, str(p)))
        if count%100000==0:
            #print count,loss/count
            print('%s\tencountered: %d\tcurrent logloss: %f y: %f' % (
                datetime.now(), count, loss/count,y))
#f=open('score','a')
#f.write('fea %s\n'%(sys.argv[1]))
#f.close()
