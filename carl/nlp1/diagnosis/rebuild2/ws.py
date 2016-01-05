f=open('run.sh','w')
for i in range(16):
    f.write('pypy rebuild1_test.py %d &\n'%i)
f.close()
