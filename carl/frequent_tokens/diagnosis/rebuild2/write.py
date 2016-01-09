import csv
import gc
import sys
import pickle
csv.field_size_limit(sys.maxsize)
name='../../../master/diagnosis_p.csv'

X={}
for i in range(16):
    tmp=pickle.load(open('count_train/count%d.p'%i))
    for j in tmp:
        X[j]=tmp[j]
    print i

#X=pickle.load(open('count.p'))
print 'load done'
f=open('train.svm','w')
name='../../../master/patients_train.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    p=row['patient_id']
    tmp=[]
    if p in X:
        tmp=X[p]
    line=' '.join([i for d,i in enumerate(tmp) ])
    f.write('%f %s\n'%(float(row['is_screener']),line))
    if c%100000==0:
        print c
f.close()
"""
f=open('test.svm','w')
name='../../../master/patients_test.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    p=row['patient_id']
    tmp=[]
    if p in X:
        tmp=X[p]
    line=' '.join([i for d,i in enumerate(tmp) ])
    f.write('0 %s\n'%(line))
    if c%100000==0:
        print c
f.close()
"""
