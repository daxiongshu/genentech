import csv
name='../../input/procedure_head.csv'
exclude={}
f=open('../../input/train_patients_to_exclude.csv')
for line in f:
    exclude[line[:-1]]=1
f.close()
f=open('../../input/test_patients_to_exclude.csv')
for line in f:
    exclude[line[:-1]]=1
f.close()
diag={}
pmap={}
D=2**30
fea='procedure_code,place_of_service,plan_type,units_administered,primary_physician_role'.split(',')
for c,row in enumerate(csv.DictReader(open(name))):
    if row['patient_id'] in exclude:
        continue
    tmp=' '.join([row[i] for i in fea])
    fname=row['patient_id']
    if fname not in diag:
        diag[fname]=[]
    diag[fname].append('%s_%d'%(row['claim_id'],abs(hash(tmp))%D))
    if c%10000==0:
        print c,tmp,abs(hash(tmp))%D
import pickle
#pickle.dump(diag,open('diag.p','w'))
print 'load done'
f=open('procedure_encode.csv','w')
f.write('patient_id,doc\n')
name='../../input/patients_train.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    p=row['patient_id']
    if p in exclude:
        continue
    tmp=[]
    if p in diag:
        tmp=[str(i) for i in diag[p]]
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
    if p in exclude:
        continue
    tmp=[]
    if p in diag:
        tmp=[str(i) for i in diag[p]]
    line=' '.join(tmp)
    f.write('%s,%s\n'%(p,line))
    if c%100000==0:
        print c
f.close()

