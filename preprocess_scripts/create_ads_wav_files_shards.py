import os 
import numpy as np 
import json
import pandas as pd 
# def create_file_list_shard(shard_folder,file_list):
#     file_list_chunks=np.array_split(file_list,10)
#     print(file_list_chunks[0])

#     for i in range(len(file_list_chunks)):

#         print("Processing chunk ",i)

#         with open(os.path.join(shard_folder,"shard_"+str(i)+".txt"),"w") as f:
#             for file in file_list_chunks[i]:
#                 f.write(file+"\n")
def create_file_list_json(file_list):

    dict_tot=dict()
    wav_file_list=[]
    for file in file_list:
        temp_dict={"wav":file}
        wav_file_list.append(temp_dict)

    dict_tot["data"]=wav_file_list
    #print(dict_tot)
    return(dict_tot)

def generate_wav_file_name(base_folder,video_file_list):


    dict_tot=dict()
    wav_file_list=[]

    for video_file in video_file_list:
        wav_file_name=os.path.splitext(video_file)[0]+".wav"
        wav_file_path=os.path.join(base_folder,wav_file_name)
        if(os.path.exists(wav_file_path)):
            temp_dict={"wav":wav_file_path}
            wav_file_list.append(temp_dict)

    dict_tot["data"]=wav_file_list
    return(dict_tot)
    
folder="/scratch2/dbose/ads_complete_dir/ads_wav_files/jwt_ads_of_world_wav_files"
# file_list=[os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".wav")]
csv_file="/project/shrikann_35/dbose/codes/ads_codes/SAIM-ADS/data/SAIM_ads_data_message_tone_train_test_val_clip_features.csv"
csv_data=pd.read_csv(csv_file)
video_file_list=csv_data["video_file"].tolist()
#dict_tot=create_file_list_json(file_list)
#save the json file 
# filename=folder.split("/")[-1]
# with open("../data/"+filename+".json","w") as f:
#     json.dump(dict_tot,f,indent=4)
dict_tot=generate_wav_file_name(folder,video_file_list)
filename=folder.split("/")[-1]
with open("../data/"+filename+".json","w") as f:
    json.dump(dict_tot,f,indent=4)




#create_file_list_shard(shard_folder,file_list)

#split the list into chunks of 1000


#print(file_list[0:5])
#def create_shards(folder,shard_folder):

    