import pickle
tokens=pickle.load(open('tokens.p'))

count=0
frequent_token={}
for i in tokens:
    if tokens[i]>1000:
        count+=1
        frequent_token[i]=tokens[i]
print count,len(tokens),len(frequent_token)
pickle.dump(frequent_token,open('ftoken.p','w'))
