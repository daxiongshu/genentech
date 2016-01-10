import csv
name='../../input/procedure_head.csv'

diag={}
D=2**30
fea='procedure_code,place_of_service,plan_type,units_administered,primary_physician_role'.split(',')

#,attending_practitioner_id,referring_practitioner_id,rendering_practitioner_id,ordering_practitioner_id,operating_practitioner_id'.split(',')
for c,row in enumerate(csv.DictReader(open(name))):
    tmp=' '.join([row[i] for i in fea])
    if row['patient_id'] not in diag:
        diag[row['patient_id']]=[]
    diag[row['patient_id']].append(abs(hash(tmp))%D)
    if c%10000==0:
        print c,tmp,abs(hash(tmp))%D
import pickle
pickle.dump(diag,open('procedure.p','w'))
