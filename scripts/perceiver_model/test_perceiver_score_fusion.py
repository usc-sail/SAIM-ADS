#score fusion for perceiver model
import os 
import json 
import numpy as np
import argparse
import torch.nn as nn 
import torch
import pandas as pd
import pickle
from tqdm import tqdm 

def single_max_fusion(av_logits_i,tv_logits_i):
    av_max_ind=torch.max(av_logits_i,1)[1]
    tv_max_ind=torch.max(tv_logits_i,1)[1]

    if(av_logits_i[av_max_ind]>tv_logits_i[tv_max_ind]):
        return av_max_ind
    else:
        return tv_max_ind
    
def multi_max_fusion(av_logits_i,tv_logits_i):

    max_values, ind_values = torch.max(torch.stack([av_logits_i, tv_logits_i]), dim=0)
    #find maximum over the max values
    max_ind=torch.max(max_values,1)[1]

    return(max_ind)



av_file="/proj/digbose92/ads_repo/model_files/predictions/multi_run_Audio_visual_perceiver_topic_single_task_classifier_shot_level_multiple_seeds_Topic_20230422-010404_logits.pkl" #audio visual file 
tv_file="/proj/digbose92/ads_repo/model_files/predictions/multi_run_Perceiver_text_visual_single_task_classifier_shot_level_multiple_seeds_Topic_20230409-220327.pkl" #text visual file
csv_file="/proj/digbose92/ads_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_no_zero_files.csv"
topic_file="/proj/digbose92/ads_repo/ads_codes/SAIM-ADS/data/topic_list_18.json"
#there will be 5 x 5 values for predictions for each task 

#read the predictions and the ground truth values
with open(av_file,'rb') as f:
    av_data=pickle.load(f)

with open(tv_file,'rb') as f:
    tv_data=pickle.load(f)

csv_data=pd.read_csv(csv_file)
test_data=csv_data[csv_data['Split']=='test']
task_name="Topic"
fusion_strategy="single_max"

### task name contains social message, Topic, transition_val
if(task_name=='social_message'):
    label_map={'No':0,'Yes':1}

elif(task_name=='Transition_val'):
    label_map={'No transition':0,'Transition':1}


elif(task_name=='Topic'):
    with open(topic_file) as f:
        label_map=json.load(f)

num_classes=len(label_map.keys())

#clip files 
clip_feature_list=test_data['clip_feature_path'].tolist()
clip_keys=[os.path.splitext(file.split("/")[-1])[0] for file in clip_feature_list]

#create the mapping between the clip keys and the ground truth labels
clip_label_map=dict(zip(clip_keys,test_data[task_name].tolist()))
print(clip_label_map)

#for i in tqdm(np.arange(test_data.shape[0])):

#start with number of runs of av and tv
# 
key_list=['run_'+str(i) for i in np.arange(5)]

for av_key in key_list:
    av_logits=av_data[av_key]['logits']
    clip_key_av=av_data[av_key]['clip_key']
    av_seed=av_data[av_key]['seed']

    for tv_key in key_list:
        #print(tv_key)
        tv_logits=tv_data[tv_key]['logits']
        clip_key_tv=tv_data[tv_key]['clip_key']
        tv_seed=tv_data[tv_key]['seed']
        gt_list=[]
        pred_list=[]

        #LOOP THROUGH THE CLIP KEYS AND CHECK IF THEY ARE SAME
        for i,clip_key in enumerate(clip_key_tv): #loop over every sample 

            if(task_name=='Topic'):
                gt_list.append(label_map[clip_label_map[clip_key]])

                #combine the predictions from av and tv using maximum value after applying softmax

                av_logits_i=av_logits[i]
                tv_logits_i=tv_logits[i]

                av_logits_i=nn.Softmax(dim=0)(torch.from_numpy(av_logits_i))
                tv_logits_i=nn.Softmax(dim=0)(torch.from_numpy(tv_logits_i))

                if(fusion_strategy=='single_max'):
                    pred_label=single_max_fusion(av_logits_i,tv_logits_i)

                elif(fusion_strategy=='multi_max'):
                    pred_label=multi_max_fusion(av_logits_i,tv_logits_i)

                pred_list.append(pred_label.item())



            elif((task_name=='Transition_val') or (task_name=='social_message')):
                label_c=label_map[clip_label_map[clip_key]]
                ret_label=np.zeros((num_classes))
                ret_label[label_c]=1
                gt_list.append(ret_label)

                

            

        #now we have the logits for both av and tv

#task name will be from argparse 

# parser=argparse.ArgumentParser()
# parser.add_argument("--task_name",type=str,default="Topic",help="Topic")

# args=parser.parse_args()

# task_name=args.task_name


# csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_no_zero_files.csv"
# embedding_file="/data/digbose92/ads_complete_repo/ads_features/ast_embeddings/ast_embs_0.5.pkl"
# transcript_file="/data/digbose92/ads_complete_repo/ads_transcripts/combined_transcripts/en_combined_transcripts.json"
# topic_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/topic_list_18.json"


# #load the csv file
# csv_data=pd.read_csv(csv_file)
# test_data=csv_data[csv_data['Split']=='test']
# max_text_length=256
# max_audio_length=14
# max_video_length=35

# #label map for each task
# if(task_name=="Topic"):
#     with open(topic_file) as f:
#         label_map=json.load(f)

# elif(task_name=='Transition_val'):
#     label_map={'No transition':0,'Transition':1}

# elif(task_name=='social_message'):
#     label_map={'No':0,'Yes':1}



    

