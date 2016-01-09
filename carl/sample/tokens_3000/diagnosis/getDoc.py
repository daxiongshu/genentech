import csv
import os
import re
import sys
num=int(sys.argv[1])
tokens={}
def pad(x):
    return ''.join(re.findall(r'[a-zA-Z0-9_]',x))
def pad2(x):
    a='_'.join(x.split())
    b=re.sub('-','_',a)
    b=re.sub(':','_',b)
    return b
for f in ['diagnosis_head.csv']:
    print f
    if True:
        for c,row in enumerate(csv.DictReader(open('../../../input/%s'%f))):
            if int(row['patient_id'])%16!=num%16:
                continue
            if row['patient_id'] not in tokens:
                tokens[row['patient_id']]=''
            if c%10000==0:
                print c
            if len(tokens[row['patient_id']])>3000:
                continue
            tokens[row['patient_id']]=tokens[row['patient_id']]+' '+' '.join([i+'_'+pad(row[i]) for i in row if i!='patient_id'])

 
print 'combine done',patient_files
import pickle
print len(tokens)
pickle.dump(tokens,open('tokens/token%d.p'%num,'w'))       
