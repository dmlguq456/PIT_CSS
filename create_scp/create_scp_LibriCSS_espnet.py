import os

test_mix_scp = 'LibriCSS_monoaural_utterance_v3.scp'
# test_s1_scp = 'tt_s1.scp'
# test_s2_scp = 'tt_s2.scp'
test_mix = '/home/nas/user/Uihyeop/DB/LibriCSS/exp/data/monaural/utterances'
# test_s1 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/min/tt/s1'
# test_s2 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/min/tt/s2'

all_lines = []
with open(os.path.join('meeting_info_ESPNET.txt'),'r') as f:
    for line in f.readlines():
            name,spk_id = line.rstrip().split('\t')
            tmp = name.rstrip().split('_')
            subdir = '_'.join(tmp[1:-2])
            file = '_'.join(tmp[-2:])
            all_lines.append(name+' '+test_mix+'/'+subdir + '/' + file + '.wav')
            # all_lines.append(name+'\t'+test_mix+'/'+subdir + '/' + file + '.wav')

with open(os.path.join('meeting_info_ESPNET_v2.scp'),'w') as f:
    for item in all_lines:
        f.write(item+'\n')


# tt_mix = open(test_mix_scp,'w')
# for root, dirs, files in os.walk(test_mix):
#     dirs.sort()
#     for subdir in (test_mix + '/' + element for element in dirs):
#         for _, _, subfiles in os.walk(subdir):
#             subfiles.sort()
#             for file in subfiles:
#                 tt_mix.write(subdir.split('/')[-1]+'_'+file[:-4])
#                 tt_mix.write(" ")
#                 tt_mix.write(subdir+'/'+file)
#                 tt_mix.write('\n')


# tt_s1 = open(test_s1_scp,'w')
# for root, dirs, files in os.walk(test_s1):
#     files.sort()
#     for file in files:
#         tt_s1.write(file+" "+root+'/'+file)
#         tt_s1.write('\n')


# tt_s2 = open(test_s2_scp,'w')
# for root, dirs, files in os.walk(test_s2):
#     files.sort()
#     for file in files:
#         tt_s2.write(file+" "+root+'/'+file)
#         tt_s2.write('\n')