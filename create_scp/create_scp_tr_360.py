import os



train_mix_scp = 'scp_librimix_css_v1_segment/tr_360_mix.scp'
train_s1_scp = 'scp_librimix_css_v1_segment/tr_360_s1.scp'
train_s2_scp = 'scp_librimix_css_v1_segment/tr_360_s2.scp'


train_mix = '/home/work/data_Uihyeop/data/Libri2MIX/train-360/mix'
train_s1 = '/home/work/data_Uihyeop/data/Libri2MIX/train-360/s1'
train_s2 = '/home/work/data_Uihyeop/data/Libri2MIX/train-360/s2'

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
