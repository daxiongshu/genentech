import csv
name='../../input/diagnosis_head.csv'
patient={}
for c,row in enumerate(csv.DictReader(open(name))):
    if row['patient_id']  not in patient:
        patient[row['patient_id']]=[]
    patient[row['patient_id']].append(' '.join(["%d_%s"%(d,row[i]) for d,i in enumerate(row) if i not in ['patient_id','claim_id']]))
    if c%100000==1:
        print c

f=open('diagnosis_p.csv','w')
f.write('patient_id,doc\n')
name='patients_train.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    p=row['patient_id']
    tmp=[]
    if p in patient:
        tmp=patient[p]
    line=' '.join(tmp)
    f.write('%s,%s\n'%(p,line))
    if c%100000==0:
        print c
f.close()


f=open('diagnosis_p_test.csv','w')
f.write('patient_id,doc\n')
name='patients_test.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    p=row['patient_id']
    tmp=[]
    if p in patient:
        tmp=patient[p]
    line=' '.join(tmp)
    f.write('%s,%s\n'%(p,line))
    if c%100000==0:
        print c
f.close()

