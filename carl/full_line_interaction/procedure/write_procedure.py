import csv
import pickle
patient=pickle.load(open('procedure.p'))
print 'load done'
f=open('procedure_encode.csv','w')
f.write('patient_id,doc\n')
name='../../input/patients_train.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    p=row['patient_id']
    tmp=[]
    if p in patient:
        tmp=[str(i) for i in patient[p]]
    line=' '.join(tmp)
    f.write('%s,%s\n'%(p,line))
    if c%100000==0:
        print c
f.close()


f=open('procedure_encode_test.csv','w')
f.write('patient_id,doc\n')
name='../../input/patients_test.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    p=row['patient_id']
    tmp=[]
    if p in patient:
        tmp=[str(i) for i in patient[p]]
    line=' '.join(tmp)
    f.write('%s,%s\n'%(p,line))
    if c%100000==0:
        print c
f.close()

