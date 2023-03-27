import os 
import numpy as np 
import pandas as pd 
import json

def generate_wav_file_shards(folder,shard_folder):

    file_list=[os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".wav")]
    file_list_chunks=np.array_split(file_list,10)
    for i in range(len(file_list_chunks)):

        print("Processing chunk ",i)
        with open(os.path.join(shard_folder,"shard_"+str(i)+".txt"),"w") as f:
            for file in file_list_chunks[i]:
                f.write(file+"\n")

def generate_wav_file_dict(jwt_ads_of_world_folder,cvpr_ads_folder,csv_data):

    jwt_ads_of_world_file_list=os.listdir(jwt_ads_of_world_folder)
    cvpr_ads_file_list=os.listdir(cvpr_ads_folder)

    list_files=[]
    for wav_file_name in wav_file_names:

        if(wav_file_name in jwt_ads_of_world_file_list):
            wav_file_path=os.path.join(jwt_ads_of_world_folder,wav_file_name)
        elif(wav_file_name in cvpr_ads_file_list):
            wav_file_path=os.path.join(cvpr_ads_folder,wav_file_name)

        list_files.append({'wav':wav_file_path})
    
    data_dict={"data":list_files}
    with open(os.path.join("/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/","jwt_ads_of_world_cvpr_wav_files.json"),"w") as f:
        json.dump(data_dict,f, indent=4)

# folder="/scratch2/dbose/ads_complete_dir/ads_wav_files/jwt_ads_of_world_wav_files"
# shard_folder="/scratch2/dbose/ads_complete_dir/ads_wav_shards"
# file_list=[os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".wav")]

#split the list into chunks of 1000
# generate_wav_file_shards(folder,shard_folder)

jwt_ads_of_world_folder="/data/digbose92/ads_complete_repo/ads_wav_files/jwt_ads_of_world_wav_files"
cvpr_ads_folder="/data/digbose92/ads_complete_repo/ads_wav_files/cvpr_wav_files"
csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_transcripts_augmented.csv"
csv_data=pd.read_csv(csv_file)
transcript_file_list=list(csv_data["Transcript"])
wav_file_names=[os.path.splitext(f.split("/")[-1])[0]+".wav" for f in transcript_file_list]
generate_wav_file_dict(jwt_ads_of_world_folder,cvpr_ads_folder,wav_file_names)




    