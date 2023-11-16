import os



train_s_scp = 'scp_se_wsj/tr_s.scp'
train_s1 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/min/tr/s1'
train_s2 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/min/tr/s2'

test_s_scp = 'scp_se_wsj/tt_s.scp'
test_s1 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/min/tr/s1'
test_s2 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/min/tr/s2'

cv_s_scp = 'scp_se_wsj/cv_s.scp'
cv_s1 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/min/tr/s1'
cv_s2 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/min/tr/s2'




tr_s = open(train_s_scp,'w')
for root, dirs, files in os.walk(train_s1):
    files.sort()
    for file in files:
        tr_s.write(file+"_src1"+" "+root+'/'+file)
        tr_s.write('\n')
for root, dirs, files in os.walk(train_s2):
    files.sort()
    for file in files:
        tr_s.write(file+"_src2"+" "+root+'/'+file)
        tr_s.write('\n')


tt_s = open(test_s_scp,'w')
for root, dirs, files in os.walk(test_s1):
    files.sort()
    for file in files:
        tt_s.write(file+"_src1"+" "+root+'/'+file)
        tt_s.write('\n')
for root, dirs, files in os.walk(test_s2):
    files.sort()
    for file in files:
        tt_s.write(file+"_src2"+" "+root+'/'+file)
        tt_s.write('\n')



cv_s = open(cv_s_scp,'w')
for root, dirs, files in os.walk(cv_s1):
    files.sort()
    for file in files:
        cv_s.write(file+"_src1"+" "+root+'/'+file)
        cv_s.write('\n')
for root, dirs, files in os.walk(cv_s2):
    files.sort()
    for file in files:
        cv_s.write(file+"_src2"+" "+root+'/'+file)
        cv_s.write('\n')

