from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import os 
import pandas as pd
from tqdm import tqdm
import json

#model and tokenizer 
model_name = 'qanastek/51-languages-classifier'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, truncation=True)

#csv file 
file="/data/digbose92/codes/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_transcripts_augmented.csv"
SAIM_data=pd.read_csv(file)

#source folder
source_folder="/bigdata/digbose92/ads_data/ads_complete_repo/ads_transcripts"

#transcipt file list
transcript_list=SAIM_data['Transcript'].tolist()

#language list
language_list=dict()

#number of files
num_files=0

for transcript in tqdm(transcript_list):
    #print(transcript)
    filename=os.path.join(source_folder,transcript.split("/")[-1])
    with open(filename, 'r') as f:
        text=f.read()
    
    result=classifier(text)
    #print('Language:', result)
    language_list[os.path.splitext(transcript.split("/")[-1])[0]]=result

    num_files=num_files+1

    # if(num_files==100):
    #     break

#save language list
with open("../data/SAIM_data/language_list.json", "w") as f:
    json.dump(language_list, f, indent=4)

        
