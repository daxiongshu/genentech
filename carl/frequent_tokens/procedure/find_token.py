import csv
import gc
import sys
csv.field_size_limit(sys.maxsize)
name='../../master/write_procedure/procedure2_p.csv'

tokens={}
for c,row in enumerate(csv.DictReader(open(name))):
    tmptoken={}
    for i in row['doc'].split():
        if i not in tmptoken:
            tmptoken[i]=0
        tmptoken[i]+=1
    for i in tmptoken:
        if tmptoken[i]>10:
            if i not in tokens:
                tokens[i]=0
            tokens[i]+=tmptoken[i]
    if c%10000==0:
        print c,len(tokens)
        gc.collect()
import pickle
pickle.dump(tokens,open('tokens.p','w'))
