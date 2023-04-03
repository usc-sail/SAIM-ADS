#combine english and translated transcripts into a single json file with the following format:
#key : transcript and empty if not present 

import json
import os
import pandas as pd 
import numpy as np
import argparse
from tqdm import tqdm 

#actual csv file 
csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_transcripts_augmented.csv"
csv_data=pd.read_csv(csv_file)
#print(csv_data.keys())
transcript_list=csv_data['Transcript'].tolist()
transcript_keys=[os.path.splitext(f.split("/")[-1])[0] for f in transcript_list]

#language id file 
language_id_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/language_metadata/language_detection_whisper.json"
with open(language_id_file) as f:
    language_id = json.load(f)

#non english translated files
file="/data/digbose92/ads_complete_repo/ads_transcripts/translated_english/gpt-4_non_english_transcripts_translated.json"
with open(file,"r") as f:
    non_english_transcripts = json.load(f)

#print(language_id)
#
num_non_en_files=0
num_en_files=0
num_zero_files=0
num_non_zero_files=0
transcript_dict=dict()
#filter english and non english transcripts
for file in tqdm(transcript_list):

    key=os.path.splitext(file.split("/")[-1])[0]
    language_detect=language_id[key]['label']

    #read the file and check the length of the transcript
    with open(file) as f:
        transcript=f.readlines()

    if len(transcript)==0:
        num_zero_files+=1
    else:
        if language_detect=="en":
            num_en_files+=1
            transcript_dict[key]=transcript[0]
        else:
            num_non_en_files+=1

            #sample from the non english translated files
            transcript_dict[key]=non_english_transcripts[key]['answer']

        num_non_zero_files+=1


print("num_en_files:",num_en_files) #6849
print("num_non_en_files:",num_non_en_files)  #1551
print("num_zero_files:",num_zero_files) #0
print("num_non_zero_files:",num_non_zero_files) #8400

#print(transcript_dict)


#save the dictionary to a json file
with open("/data/digbose92/ads_complete_repo/ads_transcripts/combined_transcripts/en_combined_transcripts.json","w") as f:
    json.dump(transcript_dict,f,indent=4)








