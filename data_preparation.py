import os
import soundfile as sf
import random
random.seed(3)

idx = 1

save_path1 = "demand_ms_snsd_train/"
save_path2 = "demand_ms_snsd_val/"

path = "ANC"
dir_list = os.listdir(path)

for dir_nm in dir_list:
    dir_path = path+"/"+dir_nm
    
    file_list = os.listdir(dir_path)
    for file_nm in file_list:
        file_path = path+"/"+dir_nm+"/"+file_nm
        x, fs = sf.read(file_path, dtype="float32")
        
        itr = x.shape[0]
        cut = 6 * fs
        num = itr // cut
        
        for k in range(num):
            
            x_cut = x[ k*cut : (k+1)*cut ]
            norm = max(abs(x_cut))
            x_cut = x_cut / norm
            
            idx_str = str(idx)
            idx_str_len = len(idx_str)
            idx_max_len = 7
            save_idx = "0"*(idx_max_len - idx_str_len) + idx_str
            save_nm = "noise_" + save_idx + ".wav"
            
            if idx % 10 == 0:
                rand_num = random.uniform(0.3, 0.99)
                x_cut = x_cut * rand_num
                sf.write("demand_ms_snsd_val/"+save_nm, x_cut, fs)
                
            else:
                sf.write("demand_ms_snsd_train/"+save_nm, x_cut, fs)
            idx += 1
            
            
            
        
    
