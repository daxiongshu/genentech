import csv
name='../frequent_tokens/diagnosis/diagnosis_p.csv'
f1=open('diagnosis_2000_train.csv','w')
f2=open('diagnosis_4000_train.csv','w')
f1.write('patient_id,doc\n')
f2.write('patient_id,doc\n')

for c,row in enumerate(csv.DictReader(open(name))):
    line=row['doc'][:2000]
    f1.write('%s,%s\n'%(row['patient_id'],line)) 
    line=row['doc'][2000:4000] 
    f2.write('%s,%s\n'%(row['patient_id'],line))    
f1.close()
f2.close()      

name='../frequent_tokens/diagnosis/diagnosis_p_test.csv'
f1=open('diagnosis_2000_test.csv','w')
f2=open('diagnosis_4000_test.csv','w')
f1.write('patient_id,doc\n')
f2.write('patient_id,doc\n')

for c,row in enumerate(csv.DictReader(open(name))):
    line=row['doc'][:2000]
    f1.write('%s,%s\n'%(row['patient_id'],line))
    line=row['doc'][2000:4000]
    f2.write('%s,%s\n'%(row['patient_id'],line))
f1.close()
f2.close()
 
