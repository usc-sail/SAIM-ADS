import os 
import pandas as pd 
import numpy as np
import pickle 
from tqdm import tqdm 

#csv file path containing the split 
csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_ads_data_message_tone_train_test_val.csv"
SAIM_ads_data=pd.read_csv(csv_file)
#print(SAIM_ads_data.shape)

#path to the folder containing the ads files
jwt_ads_of_world_folder="/data/digbose92/ads_complete_repo/ads_features/clip_embeddings/jwt_ads_of_world"
jwt_ads_of_world_feature_file_list=os.listdir(jwt_ads_of_world_folder)

cvpr_folder="/data/digbose92/ads_complete_repo/ads_features/clip_embeddings/cvpr_ads"
cvpr_feature_file_list=os.listdir(cvpr_folder)

cvpr_video_folder="/data/digbose92/ads_complete_repo/ads_videos/cvpr_videos/videos"
jwt_ads_of_world_video_folder="/data/digbose92/ads_complete_repo/ads_videos/ads_of_world_videos"

video_file_list=SAIM_ads_data['video_file'].tolist()
file_test_list=[]

cnt_files=0
feature_file_path=[]

for video_file in tqdm(video_file_list):

    if((video_file.endswith(".mp4")) or (video_file.endswith(".mkv"))):
        pkl_feat_name = video_file[:-4]+".pkl"
    elif(video_file.endswith(".webm")):
        pkl_feat_name = video_file[:-5]+".pkl"
    else:
        pkl_feat_name = video_file+".pkl"
    
    if pkl_feat_name in jwt_ads_of_world_feature_file_list:
        cnt_files=cnt_files+1
        feature_file_path.append(os.path.join(jwt_ads_of_world_folder,pkl_feat_name))
    else:
        if(pkl_feat_name in cvpr_feature_file_list):
            cnt_files=cnt_files+1
            feature_file_path.append(os.path.join(cvpr_folder,pkl_feat_name))
        else:
            cvpr_video_file_name=os.path.join(cvpr_video_folder,video_file)
            if(os.path.exists(cvpr_video_file_name)):
                #cnt_files=cnt_files+1
                file_test_list.append(cvpr_video_file_name)
            else:
                #check if the file exists in the ads of the world and jwt videos combined folder 
                jwt_ads_of_world_file_name=os.path.join(jwt_ads_of_world_video_folder,video_file)
                if(os.path.exists(jwt_ads_of_world_file_name)):
                    file_test_list.append(jwt_ads_of_world_file_name)
                else:
                    print("File not found",video_file)
            #file_test_list.append(video_file)

print(cnt_files,SAIM_ads_data.shape[0])
print((file_test_list))

#add the feature path to the csv file 
SAIM_ads_data['clip_feature_path']=feature_file_path

#save the csv file
SAIM_ads_data.to_csv("/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_ads_data_message_tone_train_test_val_clip_features.csv",index=False)


# recheck if the files are present in the folder

# for file in file_test_list:
#     if(os.path.exists(file)):
#         print("File exists",file)
#     else:
#         print("File does not exist",file)

# with open("/data/digbose92/ads_complete_repo/ads_codes/updated_pkl_files/corrected_files/file_list_clip_features_extraction_remaining.pkl","wb") as f:
#     pickle.dump(file_test_list,f)


# train_pkl_file="/data/digbose92/ads_complete_repo/ads_codes/updated_pkl_files/corrected_files/train_clip_visual_features_corrected.pkl"
# with open(train_pkl_file,"rb") as f:
#     train_data=pickle.load(f)
# print((train_data))