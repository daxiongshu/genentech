f=open('run.sh','w')
for i in range(16):
    if i%3 !=2:
        f.write('pypy getDoc.py %d &\n'%i)
    else:
        f.write('pypy getDoc.py %d \n'%i)
    #f.write('pypy writedoc.py %d\n'%i)
f.close()
