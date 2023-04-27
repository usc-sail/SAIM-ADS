import os 
import pandas as pd 
import numpy as np
import pickle 
from statistics import mean, median
from tqdm import tqdm 
import cv2

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

def check_total_duration_video_files(csv_file,video_file_name):

    csv_data=pd.read_csv(csv_file)
    clip_feature_path=csv_data['clip_feature_path'].tolist()
    clip_feature_keys=[os.path.splitext(x.split("/")[-1])[0] for x in clip_feature_path]

    with open(video_file_name, "r") as f:
        video_file_list = f.readlines()

    video_file_list=[x.strip().split("\n")[0] for x in video_file_list]
    video_file_keys=[os.path.splitext(x.split("/")[-1])[0] for x in video_file_list]

    intersect_keys=list(set(clip_feature_keys) & set(video_file_keys))

    file_duration_list=[]
    num_files=0

    for key in tqdm(intersect_keys):

        video_file_index=video_file_keys.index(key)
        video_file_path=video_file_list[video_file_index]

        #compute the duration of the file 
        video_data=cv2.VideoCapture(video_file_path)
        fps = video_data.get(cv2.CAP_PROP_FPS)
        num_frames = int(video_data.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = num_frames/fps

        file_duration_list.append(duration)
        num_files=num_files+1

    print("Mean",mean(file_duration_list))
    print("Median",median(file_duration_list))
    print("Max",max(file_duration_list))
    print("Min",min(file_duration_list))
    print('75 percentile',np.percentile(file_duration_list, 75))
    print('25 percentile',np.percentile(file_duration_list, 25))
    print('Sum of all durations',sum(file_duration_list))
    print('Number of files',num_files)

def count_number_of_shots(folder,csv_file):

    csv_data=pd.read_csv(csv_file)
    clip_feature_path=csv_data['clip_feature_path'].tolist()
    clip_feature_keys=[os.path.splitext(x.split("/")[-1])[0] for x in clip_feature_path]

    shot_file_list=os.listdir(folder)
    shot_file_keys=[os.path.splitext(x.split("/")[-1])[0] for x in shot_file_list]

    intersect_keys=list(set(clip_feature_keys) & set(shot_file_keys))
    num_files=0
    for key in tqdm(intersect_keys):
        shot_file_index=shot_file_keys.index(key)
        shot_file_path=os.path.join(folder,shot_file_list[shot_file_index])
        num_files=num_files+len(os.listdir(shot_file_path))

    print("Number of files",num_files)
    return(num_files)

def count_SBERT_features(folder):

    pkl_file_list=os.listdir(folder)
    feat_list=[]

    for file in tqdm(pkl_file_list):

        file_path=os.path.join(folder,file)
        with open(file_path, "rb") as f:
            shot_features = pickle.load(f)

        keys=sorted(list(shot_features.keys()))
        shot_feature_avg=[]
        for key in keys:
            shot_feat_temp=shot_features[key]
            if(len(shot_feat_temp)>0):
                #shot_feat_avg=np.mean(shot_feat_temp,axis=0)
                shot_feature_avg.append(shot_feat_temp[0])

        shot_feature_avg=np.array(shot_feature_avg)
        feat_list.append(shot_feature_avg.shape[0])
        #print(shot_feature_avg.shape)

    #count number of zeros 
    print("Number of zeros",feat_list.count(0))

    print("Mean",mean(feat_list))
    print("Median",median(feat_list))
    print("Max",max(feat_list))
    print("Min",min(feat_list))
    print('75 percentile',np.percentile(feat_list, 75))
    print('25 percentile',np.percentile(feat_list, 25))
    print('Sum of all durations',sum(feat_list))

def count_share_of_ads_sources(file,csv_data):

    with open(file, "r") as f:
        video_file_list = f.readlines()

    video_file_list=[x.strip().split("\n")[0] for x in video_file_list]

    clip_feature_file=csv_data['clip_feature_path'].tolist()

    clip_keys=[os.path.splitext(x.split("/")[-1])[0] for x in clip_feature_file]

    video_keys=[os.path.splitext(x.split("/")[-1])[0] for x in video_file_list]

    AOW_list=[]
    jwt_list=[]
    cvpr_list=[]

    for key in clip_keys:

        index=video_keys.index(key)
        video_file=video_file_list[index]

        if('ads_of_world_videos' in video_file):
            AOW_list.append(key)

        elif('jwt' in video_file):
            jwt_list.append(key)

        elif('cvpr' in video_file):
            cvpr_list.append(key)

    print("AOW",len(AOW_list))
    print("JWT",len(jwt_list))
    print("CVPR",len(cvpr_list))


    #Mean 24.74686687160085
    # Median 18.0
    # Max 240
    # Min 0
    # 75 percentile 34.0
    # 25 percentile 9.0
    # Sum of all durations 209309

    #print(file,data['data'].shape)

# video_folder=""
# folder="/data/digbose92/ads_complete_repo/ads_features/shot_embeddings/shot_caption_SBERT_embeddings"
# count_SBERT_features(folder)
lomond_file="/proj/digbose92/ads_repo/ads_codes/SAIM-ADS/data/lomond_available_file_path.txt"
csv_file="/proj/digbose92/ads_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_no_zero_files.csv"
csv_data=pd.read_csv(csv_file)
count_share_of_ads_sources(lomond_file,csv_data)

# csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_no_zero_files.csv"
# video_file_name="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/lomond_available_file_path.txt"
# folder="/data/digbose92/ads_complete_repo/ads_videos/shot_folder/PySceneDetect"
# count_number_of_shots(folder,csv_file)
#check_total_duration_video_files(csv_file,video_file_name)
# #print(ast_embeds.keys())
# ast_file="/data/digbose92/ads_complete_repo/ads_features/ast_embeddings/ast_embs_0.5.pkl"
# check_ast_features_stats(ast_file)

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