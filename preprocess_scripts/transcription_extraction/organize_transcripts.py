import json 
import numpy as np 
from tqdm import tqdm 
import pandas as pd
import os
from collections import Counter  
import re

def contains_only_music_symbols(text):
    pattern = r'^[\u266d\u266f\u266e\u266a\u266c\u266b\u2669\u2668\u2665\u2661\u2662\u2660\u2664\u2667\u2666\u2640\u2642\u2605\u2606\u263c\u263d\u2663\u2661\u2664\u2667\u2662\u2665\u2666\u2660]*$'
    return re.match(pattern, text) is not None

#combined csv file
csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_transcripts_augmented.csv"
csv_data=pd.read_csv(csv_file)
transcript_list=csv_data['Transcript'].tolist()
transcript_keys=[os.path.splitext(f.split("/")[-1])[0] for f in transcript_list]

#language id file 
language_id_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/language_metadata/language_detection_whisper.json"
with open(language_id_file) as f:
    language_id = json.load(f)

#filter english and non english transcripts
en_dict={}
non_en_dict={}

for key in tqdm(language_id.keys()):
    if language_id[key]['label'] == 'en':
        en_dict[key]=language_id[key]
    else:
        non_en_dict[key]=language_id[key]

print(len(en_dict),len(non_en_dict)) #6849 (English), 1551 (Non english)

len_non_english_transcripts=[]
non_english_dict=dict()
#create a list of non english transcripts and find the length of the transcripts in the list
for key in tqdm(non_en_dict.keys()):
    transcript_index=transcript_keys.index(key)
    transcript_file=transcript_list[transcript_index]

    with open(transcript_file,'r') as f:
        transcript=f.readlines()

    if(len(transcript) > 0):

        len_non_english_transcripts.append(len(transcript[0]))
        non_english_dict[key]=transcript[0]
    else:
        len_non_english_transcripts.append(0)

dest_folder="/data/digbose92/ads_complete_repo/ads_transcripts/non_english"

#save the non english transcripts in a folder
with open(os.path.join(dest_folder,"non_english_transcripts.json"),'w') as f:
    json.dump(non_english_dict,f,indent=4)


# len_counter=Counter(len_non_english_transcripts)
# print(len_counter.most_common(20))


