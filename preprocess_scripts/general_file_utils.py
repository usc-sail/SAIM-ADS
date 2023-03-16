#helper scripts for general file utils 
import os 
import numpy as np 
import pandas as pd 
import json 
import pickle

#folder path and option .. lomond locations
ads_of_world_video_folder="/data/digbose92/ads_complete_repo/ads_videos/ads_of_world_videos"
jwt_video_folder="/data/digbose92/ads_complete_repo/ads_videos/jwt_videos/videos"
cvpr_video_folder="/data/digbose92/ads_complete_repo/ads_videos/cvpr_videos/videos"

ads_of_world_jwt_video_list=[os.path.join(ads_of_world_video_folder,i) for i in os.listdir(ads_of_world_video_folder)] +[os.path.join(jwt_video_folder,i) for i in os.listdir(jwt_video_folder) if i.endswith(".mp4")]
cvpr_video_list=[os.path.join(cvpr_video_folder,i) for i in os.listdir(cvpr_video_folder)]
total_file_list=ads_of_world_jwt_video_list+cvpr_video_list
file_keys=[f.split("/")[-1] for f in total_file_list]
print(len(file_keys))

#read the csv file
csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_ads_data_message_tone_train_test_val_clip_features.csv"
csv_data=pd.read_csv(csv_file)
video_file_list=list(set(csv_data['video_file']))

cnt_video_file_num=0
available_file_path=[]
#create video file list
for video_file in video_file_list:
    if(video_file in file_keys):
        cnt_video_file_num+=1
        
        index=file_keys.index(video_file)
        video_file_path=total_file_list[index]
        if(os.path.exists(video_file_path)==True):
            #print(video_file_path
            available_file_path.append(video_file_path)

#save the file list in a txt file 

with open("../data/lomond_available_file_path.txt","w") as f:

    for file_path in available_file_path:
        f.write(file_path)
        f.write("\n")
