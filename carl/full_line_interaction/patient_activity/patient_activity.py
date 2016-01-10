import csv
name='../../input/patient_activity_head'

diag={}
D=2**15
fea='activity_type,activity_year'.split(',')
#,attending_practitioner_id,referring_practitioner_id,rendering_practitioner_id,ordering_practitioner_id,operating_practitioner_id'.split(',')
for c,row in enumerate(csv.DictReader(open(name))):
    tmp=' '.join([row[i] for i in fea])
    if row['patient_id'] not in diag:
        diag[row['patient_id']]=[]
    diag[row['patient_id']].append(abs(hash(tmp))%D)
    if c%10000==0:
        print c,tmp,abs(hash(tmp))%D
import pickle
pickle.dump(diag,open('patient_activity.p','w'))
