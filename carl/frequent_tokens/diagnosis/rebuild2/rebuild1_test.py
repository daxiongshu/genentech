import csv
import gc
import sys
import pickle
csv.field_size_limit(sys.maxsize)
name='../diagnosis_p_test.csv'
num=int(sys.argv[1])
tokens=pickle.load(open('../ftoken2.p'))
X={}
print len(tokens)
for c,row in enumerate(csv.DictReader(open(name))):
    if c%16!=num%16:
        continue
    tmptoken={}
    for i in tokens:
        tmptoken[i]=0
    for i in row['doc'].split():
        if i in tmptoken:
            tmptoken[i]+=1
    X[row['patient_id']]=['%d:%d'%(t,tmptoken[i]) for t,i in enumerate(tokens) if tmptoken[i]>0]
    del tmptoken
    if c%100==0:
        print num,c#,X[row['patient_id'][-10:]
        gc.collect()

pickle.dump(X,open('count_test/count%d.p'%num,'w'))

