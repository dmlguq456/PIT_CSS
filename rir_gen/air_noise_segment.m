
root = '/home/nas/user/Uihyeop/DB/Reverb_air_noise/NOISE'; % YOUR_PATH/, the folder containing wsj0/
output_dir16k='/home/nas/user/Uihyeop/DB/Reverb_air_noise/NOISE_seg_7ch';
% output_dir8k='/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav8k/css_v3_segment';

samp_idx_tr = 0;
samp_idx_cv = 0;
samp_idx_tt = 0;
fs = 16000;
for i = 1:3
	for j = 1:10
		x = audioread([root,'/Noise_SimRoom',num2str(i),'_',num2str(j),'.wav']);
		for time = 1:5
			time_period = 6*fs*(time-1)+1:6*fs*(time);
			if j < 9
				audiowrite([output_dir16k,'/tr/Noise_src_sample_',num2str(samp_idx_tr),'.wav'],x(time_period,1:7),16000);
				samp_idx_tr = samp_idx_tr + 1;
			elseif j == 9
				audiowrite([output_dir16k,'/cv/Noise_src_sample_',num2str(samp_idx_cv),'.wav'],x(time_period,1:7),16000);
				samp_idx_cv = samp_idx_cv + 1;
			elseif j == 10
				audiowrite([output_dir16k,'/tt/Noise_src_sample_',num2str(samp_idx_tt),'.wav'],x(time_period,1:7),16000);
				samp_idx_tt = samp_idx_tt + 1;
			end
		end	
	end
end
