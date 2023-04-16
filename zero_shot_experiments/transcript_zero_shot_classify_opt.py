import os
import pdb
import yaml
import json
import torch
import argparse 

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from transformers import pipeline, set_seed
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

parser=argparse.ArgumentParser()
parser.add_argument("--dest_folder",type=str,help="destination folder to save the translated transcripts")
parser.add_argument('--task_name',type=str,help="name of the task")

# arguments 
args = parser.parse_args()
task_name = args.task_name

device='cuda:0' if torch.cuda.is_available() else 'cpu'
# Generator model
generator_model = pipeline(
    'text-generation', 
    model="facebook/opt-iml-1.3b", 
    device=0, 
    max_new_tokens=5
)


#load the transcripts file 
with open("/media/data/public-data/ads_complete_repo/ads_transcripts/combined_transcripts/en_combined_transcripts.json", "r") as f:
    data = json.load(f)

prediction_dict = dict()
step_files = 0

# Iterate over data
for key in tqdm(data):
    text = data[key]
    text_list = text.split(' ')
    text = ' '.join(text_list[:400])
    
    # Prompt message
    if (task_name=="topic"):
        # prompt_msg = f"{text}\nAssociate a single topic label with the transcript from the given set: \nOPTIONS:\n-Games\n-Household\n-Services\n-Sports\n-Banking\n-Clothing\n-Industrial and agriculture\n-Leisure\n-Publications media\n-Health\n-Car\n-Electronics\n-Cosmetics\n-Food and drink\n-Awareness\n-Travel and transport\n-Retail\nANSWER: "
        prompt_msg = f"In this task, you are given a transcription of an advertisement. Your task is to associate a single topic label with the transcript from the given set. \nTranscription: {text}\n \nOPTIONS:\n-Games\n-Household\n-Services\n-Sports\n-Banking\n-Clothing\n-Industrial and agriculture\n-Leisure\n-Publications media\n-Health\n-Car\n-Electronics\n-Cosmetics\n-Food and drink\n-Awareness\n-Travel and transport\n-Retail\nAnswer:"
    # logging.info(f"{prompt_msg}")
    elif (task_name=="social_message"):
        prompt_msg = f"In this task, you are given a transcription of an advertisement. An advertisement video has a social message if it provides awareness about any social issue. Example of social issues: gender equality, drug abuse, police brutality, workplace harassment, domestic violence, child labor, environmental damage, homelessness, hate crimes, racial inequality etc. Your task is to give label \"Yes\" if the advertisement given has any social message, otherwise give label \"No\". \nTranscription:{text}\nAnswer:"
    # Token
    logging.info(text)

    with torch.no_grad():
        response = generator_model(prompt_msg)
        
    # pdb.set_trace()
    # Getting the Answer
    prediction_dict[key] = response[0]['generated_text'].split('Answer:')[1].strip()
    logging.info(f"Prompt Answer: {prediction_dict[key]}")
    step_files += 1

    if (step_files % 100 == 0):
        filename = f"predicted_labels_opt_iml_{task_name}.json"
        with open(filename, "w") as f:
            json.dump(prediction_dict, f,indent=4)

filename = f"predicted_labels_opt_iml_{task_name}.json"
with open(filename, "w") as f:
    json.dump(prediction_dict, f,indent=4)



