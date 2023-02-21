import os 
import pandas as pd 
import numpy as np
import pickle 
from statistics import mean, median
from tqdm import tqdm 

file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_ads_data_message_tone_train_test_val_clip_features.csv"
ads_of_world_video_folder="/data/digbose92/ads_complete_repo/ads_videos/ads_of_world_videos"
jwt_video_folder="/data/digbose92/ads_complete_repo/ads_videos/jwt_videos/videos"
ads_of_world_video_list=os.listdir(ads_of_world_video_folder)
ads_of_world_video_keys=[os.path.splitext(x)[0] for x in ads_of_world_video_list]
jwt_video_list=os.listdir(jwt_video_folder)
jwt_video_keys=[os.path.splitext(x)[0] for x in jwt_video_list]
#print(video_keys[0:5])

SAIM_ads_tone_clip_features=pd.read_csv(file)

clip_feature_path_list=SAIM_ads_tone_clip_features['clip_feature_path'].tolist()
clip_feature_shape_list=[]
clip_features_zero_list=[]

for clip_feature_file in tqdm(clip_feature_path_list):

    if(os.path.exists(clip_feature_file)):

        with open(clip_feature_file, 'rb') as f:
            clip_features = pickle.load(f)
        #print(clip_feature_file,clip_features.shape)
        clip_feature_shape_list.append(clip_features['Features'].shape[0])
        if(clip_features['Features'].shape[0]==0):
            #print(clip_feature_file,clip_features['Features'].shape[0]
            clip_features_zero_list.append(clip_feature_file)
    else:
        print("File not found",clip_feature_file)

clip_feature_list=[x/6 for x in clip_feature_shape_list]
print("Mean",mean(clip_feature_list))
print("Median",median(clip_feature_list))
print("Max",max(clip_feature_list))
print("Min",min(clip_feature_list))
print('75 percentile',np.percentile(clip_feature_list, 75))
print('25 percentile',np.percentile(clip_feature_list, 25))
print('90 percentile',np.percentile(clip_feature_list, 90))
print('50 percentile',np.percentile(clip_feature_list, 50))

#print(clip_features_zero_list)

#list the zero files from the video folder and use that for feature extraction again 

# zero_file_list=[]

# for zero_file in clip_features_zero_list:
    
#     zero_file_key=os.path.splitext(zero_file.split("/")[-1])[0]
#     try:
#         index=ads_of_world_video_keys.index(zero_file_key)
#         print(ads_of_world_video_list[index],ads_of_world_video_keys[index],zero_file)
#         zero_file_list.append(os.path.join(ads_of_world_video_folder,ads_of_world_video_list[index]))

#     except:
#         index=jwt_video_keys.index(zero_file_key)
#         print(jwt_video_list[index],jwt_video_keys[index],zero_file)
#         zero_file_list.append(os.path.join(jwt_video_folder,jwt_video_list[index]))
    
   
# #check existence of files in zero_file_list
# for file in zero_file_list:
#     if(os.path.exists(file)):
#         print(file)
#     else:
#         print("File not found",file)

# #save the list of zero clip features files
# with open("/data/digbose92/ads_complete_repo/ads_codes/updated_pkl_files/corrected_files/zero_clip_features_files.pkl", 'wb') as f:
#     pickle.dump(zero_file_list, f)

# 75 percentile 333.29166666666663
# 25 percentile 125.16666666666667
# 90 percentile 507.7166666666667
# 50 percentile 221.41666666666669