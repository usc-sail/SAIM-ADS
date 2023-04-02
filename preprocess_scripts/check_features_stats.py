import os 
import pandas as pd 
import numpy as np
import pickle 
from statistics import mean, median
from tqdm import tqdm 

def extract_clip_features_stats(clip_feature_path_list):

    clip_feature_shape_list=[]
    clip_features_zero_list=[]
    for clip_feature_file in tqdm(clip_feature_path_list):

        if(os.path.exists(clip_feature_file)):

            with open(clip_feature_file, 'rb') as f:
                clip_features = pickle.load(f)
            
            clip_feature_shape_list.append(clip_features['Features'].shape[0])
            if(clip_features['Features'].shape[0]==0):
                
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

def extract_shot_features_stats(shot_folder):

    shot_file_list=os.listdir(shot_folder)
    shot_file_list=[s for s in shot_file_list if s.endswith('.pkl')]

    shot_feature_shape_list=[]
    shot_zero_shape_list=[]

    for shot_file in tqdm(shot_file_list):
        #print(shot_file)
        shot_feature_file=os.path.join(shot_folder,shot_file)

        with open(shot_feature_file, 'rb') as f:
            shot_features = pickle.load(f)

        key_list=list(shot_features.keys())
        sorted_key_list=key_list.sort()
        #print(sorted_key_list)
        
        if(len(shot_features)==0):
            shot_zero_shape_list.append(shot_feature_file)
        else:
            shot_avg_features=[]
            for key in list(shot_features.keys()):

                shot_feat_temp=shot_features[key]
                if(len(shot_feat_temp)>0):
                    shot_feat_avg=np.mean(shot_feat_temp,axis=0)
                    shot_avg_features.append(shot_feat_avg)

            #shot_feature_shape_list.append(shot_feat_avg.shape[0])
            shot_avg_features=np.array(shot_avg_features)

            if(shot_avg_features.shape[0]==0):
                shot_zero_shape_list.append(shot_feature_file)

            else:
                #print(shot_avg_features.shape)
                shot_feature_shape_list.append(shot_avg_features.shape[0])

    #statistics 
    # Mean 25.75679829746985
    # Median 19.0
    # Max 240
    # Min 1
    # 75 percentile 35.0
    # 25 percentile 10.0
    # 90 percentile 55.0
    # 50 percentile 19.0
    #shot_feature_shape_list.append(shot_feat_avg.shape[0])
    print(shot_zero_shape_list)
    print(len(shot_feature_shape_list))
    print("Mean",mean(shot_feature_shape_list))
    print("Median",median(shot_feature_shape_list))
    print("Max",max(shot_feature_shape_list))
    print("Min",min(shot_feature_shape_list))
    print('75 percentile',np.percentile(shot_feature_shape_list, 75))
    print('25 percentile',np.percentile(shot_feature_shape_list, 25))
    print('90 percentile',np.percentile(shot_feature_shape_list, 90))
    print('50 percentile',np.percentile(shot_feature_shape_list, 50))


def check_ast_features_stats(ast_file):

    with open(ast_file, "rb") as f:
        ast_features = pickle.load(f)

    #ast features here 
    #print(ast_features.keys())
    ast_dataset=ast_features['data']
    ast_embeds=ast_dataset['embeddings']

    ast_shape_list=[]

    for key in tqdm(list(ast_embeds.keys())):
        ast_emb=ast_embeds[key]
        ast_shape_list.append(ast_emb.shape[0])

    print("Mean",mean(ast_shape_list))
    print("Median",median(ast_shape_list))
    print("Max",max(ast_shape_list))
    print("Min",min(ast_shape_list))
    print('75 percentile',np.percentile(ast_shape_list, 75))
    print('25 percentile',np.percentile(ast_shape_list, 25))
    print('90 percentile',np.percentile(ast_shape_list, 90))
    print('50 percentile',np.percentile(ast_shape_list, 50))

    #Mean 10.768095238095238
    # Median 9.0
    # Max 86
    # Min 1
    # 75 percentile 14.0
    # 25 percentile 4.0
    # 90 percentile 22.0
    # 50 percentile 9.0
#print(ast_embeds.keys())
ast_file="/data/digbose92/ads_complete_repo/ads_features/ast_embeddings/ast_embs_0.5.pkl"
check_ast_features_stats(ast_file)

# shot_folder="/data/digbose92/ads_complete_repo/ads_features/shot_embeddings/vit_features"
# extract_shot_features_stats(shot_folder)

#clip feature extraction segment
# file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_ads_data_message_tone_train_test_val_clip_features.csv"
# ads_of_world_video_folder="/data/digbose92/ads_complete_repo/ads_videos/ads_of_world_videos"
# jwt_video_folder="/data/digbose92/ads_complete_repo/ads_videos/jwt_videos/videos"
# ads_of_world_video_list=os.listdir(ads_of_world_video_folder)
# ads_of_world_video_keys=[os.path.splitext(x)[0] for x in ads_of_world_video_list]
# jwt_video_list=os.listdir(jwt_video_folder)
# jwt_video_keys=[os.path.splitext(x)[0] for x in jwt_video_list]
# #print(video_keys[0:5])

# SAIM_ads_tone_clip_features=pd.read_csv(file)

# clip_feature_path_list=SAIM_ads_tone_clip_features['clip_feature_path'].tolist()
# clip_feature_shape_list=[]
# clip_features_zero_list=[]
# extract_clip_features_stats(clip_feature_path_list)



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