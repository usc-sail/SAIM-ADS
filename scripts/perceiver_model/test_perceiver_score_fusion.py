#score fusion for perceiver model
import os 
import json 
import numpy as np
import argparse
import torch.nn as nn 
import torch
import pandas as pd
#task name will be from argparse 

parser=argparse.ArgumentParser()
parser.add_argument("--task_name",type=str,default="Topic",help="Topic")

args=parser.parse_args()

task_name=args.task_name


csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_no_zero_files.csv"
embedding_file="/data/digbose92/ads_complete_repo/ads_features/ast_embeddings/ast_embs_0.5.pkl"
transcript_file="/data/digbose92/ads_complete_repo/ads_transcripts/combined_transcripts/en_combined_transcripts.json"
topic_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/topic_list_18.json"


#load the csv file
csv_data=pd.read_csv(csv_file)
test_data=csv_data[csv_data['Split']=='test']
max_text_length=256
max_audio_length=14
max_video_length=35

#label map for each task
if(task_name=="Topic"):
    with open(topic_file) as f:
        label_map=json.load(f)

elif(task_name=='Transition_val'):
    label_map={'No transition':0,'Transition':1}

elif(task_name=='social_message'):
    label_map={'No':0,'Yes':1}



    

