import pandas as pd
import numpy as np
from sklearn import metrics

def myauc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    return metrics.auc(fpr, tpr)
import sys
name=sys.argv[1]
s=pd.read_csv(name)
y=s['real']
yr=s['is_screener']
print 'auc %f'%(myauc(y,yr))
