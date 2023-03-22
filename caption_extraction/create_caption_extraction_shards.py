import os 
import pandas as pd 
import numpy as np
import json

folder="/scratch1/dbose/ads_repo/shot_folder/PySceneDetect"
caption_split_folder="/project/shrikann_35/dbose/codes/ads_codes/caption_split_file"
subfolder_list=os.listdir(folder)
caption_split_subfolder=os.listdir(caption_split_folder)

#create multiple groups of 2000 files each
group_size=2000
num_groups=int(np.ceil(len(subfolder_list)/group_size))
print("num_groups",num_groups)

for i in range(num_groups):
    print("i",i)
    start_index=i*group_size
    end_index=(i+1)*group_size
    if end_index>len(subfolder_list):
        end_index=len(subfolder_list)
    subfolder_list_group=subfolder_list[start_index:end_index]


    print("subfolder_list_group",subfolder_list_group)


    caption_split_file=os.path.join(caption_split_folder,"caption_split_"+str(i)+".txt")
    with open(caption_split_file,"w") as f:
        for subfolder in subfolder_list_group:
            f.write(subfolder+"\n")

    #create a file for each group

    