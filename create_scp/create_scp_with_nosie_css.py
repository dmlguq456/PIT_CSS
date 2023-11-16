import os
import numpy as np
import soundfile as sp
import math


train_mix_scp = 'scp_css/tr_mix_css_v1.scp'
train_s1_scp = 'scp_css/tr_s1_css_v1.scp'
train_s2_scp = 'scp_css/tr_s2_css_v1.scp'
train_n_scp = 'scp_css/tr_n_css_v1.scp'
train_mix_n_scp = 'scp_css/tr_mix_n_css_v1.scp'


test_mix_scp = 'scp_css/tt_mix_css_v1.scp'
test_s1_scp = 'scp_css/tt_s1_css_v1.scp'
test_s2_scp = 'scp_css/tt_s2_css_v1.scp'
test_n_scp = 'scp_css/tt_n_css_v1.scp'
test_mix_n_scp = 'scp_css/tt_mix_n_css_v1.scp'


train_mix = '/home/nas/DB/wsj0-mix/2speakers/wav16k/css/tr/mix'
train_s1 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/css/tr/s1'
train_s2 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/css/tr/s2'
train_n = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/css/tr/n'
train_mix_n = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/css/tr/mix_n'

test_mix = '/home/nas/DB/wsj0-mix/2speakers/wav16k/css/tt/mix'
test_s1 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/css/tt/s1'
test_s2 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/css/tt/s2'
test_n = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/css/tt/n'
test_mix_n = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/css/tt/mix_n'

cv_mix_scp = 'scp_css/cv_mix_css_v1.scp'
cv_s1_scp = 'scp_css/cv_s1_css_v1.scp'
cv_s2_scp = 'scp_css/cv_s2_css_v1.scp'
cv_n_scp = 'scp_css/cv_n_css_v1.scp'
cv_mix_n_scp = 'scp_css/cv_mix_n_css_v1.scp'

cv_mix = '/home/nas/DB/wsj0-mix/2speakers/wav16k/css/cv/mix'
cv_s1 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/css/cv/s1'
cv_s2 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/css/cv/s2'
cv_n_path = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/css/cv/n'
cv_mix_n_path = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/css/cv/mix_n'


tr_mix = open(train_mix_scp,'w')
for root, dirs, files in os.walk(train_mix):
    files.sort()
    for file in files:
        tr_mix.write(file+" "+root+'/'+file)
        tr_mix.write('\n')


tr_s1 = open(train_s1_scp,'w')
for root, dirs, files in os.walk(train_s1):
    files.sort()
    for file in files:
        tr_s1.write(file+" "+root+'/'+file)
        tr_s1.write('\n')


tr_s2 = open(train_s2_scp,'w')
for root, dirs, files in os.walk(train_s2):
    files.sort()
    for file in files:
        tr_s2.write(file+" "+root+'/'+file)
        tr_s2.write('\n')

# generate white noise
tr_n = open(train_n_scp,'w')
tr_mix_n = open(train_mix_n_scp,'w')
os.makedirs(train_n, exist_ok=True)
os.makedirs(train_mix_n, exist_ok=True)
for root, dirs, files in os.walk(train_mix):
    files.sort()
    for file in files:
        a, fs = sp.read(train_mix+'/'+file)
        power_mix = np.std(a)*np.std(a) + np.mean(a)*np.mean(a)
        SNR = np.random.uniform(-25,-15)
        n = np.random.normal(0,1,size=a.shape[0]) * math.sqrt(power_mix) * pow(10,SNR/20)
        sp.write(train_n+'/'+file,n,fs)
        sp.write(train_mix_n+'/'+file,a+n,fs)
        tr_n.write(file+" "+train_n+'/'+file)
        tr_n.write('\n')
        tr_mix_n.write(file+" "+train_mix_n+'/'+file)
        tr_mix_n.write('\n')



tt_mix = open(test_mix_scp,'w')
for root, dirs, files in os.walk(test_mix):
    files.sort()
    for file in files:
        tt_mix.write(file+" "+root+'/'+file)
        tt_mix.write('\n')


tt_s1 = open(test_s1_scp,'w')
for root, dirs, files in os.walk(test_s1):
    files.sort()
    for file in files:
        tt_s1.write(file+" "+root+'/'+file)
        tt_s1.write('\n')


tt_s2 = open(test_s2_scp,'w')
for root, dirs, files in os.walk(test_s2):
    files.sort()
    for file in files:
        tt_s2.write(file+" "+root+'/'+file)
        tt_s2.write('\n')


# generate white noise
# tt_n = open(test_n_scp,'w')
# tt_mix_n = open(test_mix_n_scp,'w')
os.makedirs(test_n,exist_ok=True)
os.makedirs(test_mix_n, exist_ok=True)
for root, dirs, files in os.walk(test_mix):
    files.sort()
    for file in files:
        a, fs = sp.read(test_mix+'/'+file)
        power_mix = np.std(a)*np.std(a) + np.mean(a)*np.mean(a)
        SNR = np.random.uniform(-25,-15)
        n = np.random.normal(0,1,size=a.shape[0]) * math.sqrt(power_mix) * pow(10,SNR/20)
        sp.write(test_n+'/'+file,n,fs)
        sp.write(test_mix_n+'/'+file,a+n,fs)
        # tt_n.write(file+" "+test_n+'/'+file)
        # tt_n.write('\n')
        # tt_mix_n.write(file+" "+test_mix_n+'/'+file)
        # tt_mix_n.write('\n')


# cv_mix_file = open(cv_mix_scp,'w')
# for root, dirs, files in os.walk(cv_mix):
#     files.sort()
#     for file in files:
#         cv_mix_file.write(file+" "+root+'/'+file)
#         cv_mix_file.write('\n')


# cv_s1_file = open(cv_s1_scp,'w')
# for root, dirs, files in os.walk(cv_s1):
#     files.sort()
#     for file in files:
#         cv_s1_file.write(file+" "+root+'/'+file)
#         cv_s1_file.write('\n')


# cv_s2_file = open(cv_s2_scp,'w')
# for root, dirs, files in os.walk(cv_s2):
#     files.sort()
#     for file in files:
#         cv_s2_file.write(file+" "+root+'/'+file)
#         cv_s2_file.write('\n')

# generate white noise
# cv_n = open(cv_n_scp,'w')
# cv_mix_n = open(cv_mix_n_scp,'w')
os.makedirs(cv_n_path, exist_ok=True)
os.makedirs(cv_mix_n_path, exist_ok=True)
for root, dirs, files in os.walk(cv_mix):
    files.sort()
    for file in files:
        a, fs = sp.read(cv_mix+'/'+file)
        power_mix = np.std(a)*np.std(a) + np.mean(a)*np.mean(a)
        SNR = np.random.uniform(-25,-15)
        n = np.random.normal(0,1,size=a.shape[0]) * math.sqrt(power_mix) * pow(10,SNR/20)
        sp.write(cv_n_path+'/'+file,n,fs)
        sp.write(cv_mix_n_path+'/'+file,a+n,fs)
        # cv_n.write(file+" "+cv_n_path+'/'+file)
        # cv_n.write('\n')
        # cv_mix_n.write(file+" "+cv_mix_n_path+'/'+file)
        # cv_mix_n.write('\n')