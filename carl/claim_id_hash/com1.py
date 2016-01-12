import csv
import sys
import csv
csv.field_size_limit(sys.maxsize)
D=2**30
def get(dline,pline):
    dd={}
    dp={}
    for d in dline.split():
        xx=d.split('_')
        cid=xx[0]
        dd[cid]=xx[1]
    for d in pline.split():
        xx=d.split('_')
        cid=xx[0]
        dp[cid]=xx[1]
    tmp=[]
    for d in dd:
        x,y=dd[d],''
        if d in dp:
            y=dp[d]
        tmp.append(str(abs(hash(x+'_'+y))))
    return ' '.join(tmp)
diag=csv.DictReader(open('diagnosis_encode.csv'))

name='procedure_encode.csv'
fo=open('dp_encode.csv','w')
fo.write('patient_id,doc\n')
for c,row in enumerate(csv.DictReader(open(name))):
    pline=row['doc']
    dline=diag.next()['doc']
    nline=get(dline,pline)
    fo.write('%s,%s\n'%(row['patient_id'],nline))
    if c%10000==0:
        print c
fo.close()

diag=csv.DictReader(open('diagnosis_encode_test.csv'))
name='procedure_encode_test.csv'
fo=open('dp_encode_test.csv','w')
fo.write('patient_id,doc\n')
for c,row in enumerate(csv.DictReader(open(name))):
    pline=row['doc']
    dline=diag.next()['doc']
    nline=get(dline,pline)
    fo.write('%s,%s\n'%(row['patient_id'],nline))
    if c%10000==0:
        print c
fo.close()
