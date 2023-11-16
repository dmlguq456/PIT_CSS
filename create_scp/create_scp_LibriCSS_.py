import os

test_mix_scp = 'LibriCSS_mix_mono_segments.scp'
# test_s1_scp = 'tt_s1.scp'
# test_s2_scp = 'tt_s2.scp'
# test_mix = '/home/nas/user/Uihyeop/DB/LibriCSS_with_Noise_All/exp/data/7ch/utterances'
test_mix = '/home/nas/user/Uihyeop/DB/LibriCSS/exp/data/monaural/segments'
# test_mix = '/home/nas/user/Uihyeop/DB/LibriCSS/exp/data/7ch/utterances'
# test_s1 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/min/tt/s1'
# test_s2 = '/home/nas/DB/wsj0-mix/2speakers/wav16k/min/tt/s2'

tt_mix = open(test_mix_scp,'w')
for root, dirs, files in os.walk(test_mix):
    dirs.sort()
    for subdir in (test_mix + '/' + element for element in dirs):
        for _, _, subfiles in os.walk(subdir):
            subfiles.sort()
            for file in subfiles:
                tt_mix.write(subdir.split('/')[-1]+'/'+file)
                # tt_mix.write(subdir.split('/')[-1]+'/'+file[:-4])
                tt_mix.write(" ")
                tt_mix.write(subdir+'/'+file)
                tt_mix.write('\n')