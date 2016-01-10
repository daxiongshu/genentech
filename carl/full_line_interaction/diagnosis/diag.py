import csv
name='../input/diagnosis_head.csv'

diag={}
D=2**30
fea='claim_type,diagnosis_code,primary_practitioner_id,primary_physician_role'.split(',')
for c,row in enumerate(csv.DictReader(open(name))):
    tmp=' '.join([row[i] for i in fea])
    if row['patient_id'] not in diag:
        diag[row['patient_id']]=[]
    diag[row['patient_id']].append(abs(hash(tmp))%D)
    if c%10000==0:
        print c,tmp,abs(hash(tmp))%D
import pickle
pickle.dump(diag,open('diag.p','w'))
