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
parser.add_argument(
    "--dest_folder", 
    type=str, 
    help="destination folder to save the translated transcripts"
)
parser.add_argument(
    '--task_name', 
    type=str, 
    help="name of the task"
)
parser.add_argument(
    '--model_name', 
    type=str, 
    help="name of the model"
)
parser.add_argument(
    '--cache_folder', 
    type=str, 
    help="name of the model"
)

# arguments 
args = parser.parse_args()
task_name = args.task_name
# pdb.set_trace()

device='cuda:0' if torch.cuda.is_available() else 'cpu'
# Generator model
generator_model = AutoModelForSeq2SeqLM.from_pretrained(
    f"google/{args.model_name}", 
    cache_dir=args.cache_folder,
    torch_dtype=torch.float16
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    f"google/{args.model_name}", 
    cache_dir=args.cache_folder
)

#load the transcripts file 
with open(args.dest_folder, "r") as f:
    data = json.load(f)

prediction_dict = dict()
step_files = 0

# Iterate over data
for key in tqdm(data):
    text = data[key]

    # Prompt message
    if (task_name=="topic"):
        prompt_msg = f"{text}\nAssociate a single topic label with the transcript from the given set: \nOPTIONS:\n-Games\n-Household\n-Services\n-Sports\n-Banking\n-Clothing\n-Industrial and agriculture\n-Leisure\n-Publications media\n-Health\n-Car\n-Electronics\n-Cosmetics\n-Food and drink\n-Awareness\n-Travel and transport\n-Retail\nANSWER: "
    elif (task_name=="transition"):
        prompt_msg = f"{text}\nBased on the given text transcript from the advertisement, determine if the advertisement has any transitions. \nOPTIONS:\n-Transition\n-No transition\nANSWER:"
    # logging.info(f"{prompt_msg}")
    elif (task_name=="social_message"):
        prompt_msg = f"{text}\nAn advertisement video has a social message if it provides awareness about any social issue. Example of social issues: gender equality, drug abuse, police brutality, workplace harassment, domestic violence, child labor, environmental damage, homelessness, hate crimes, racial inequality etc. Based on the given text transcript, determine if the advertisement has any social message. \nOPTIONS:\n-Yes\n-No\nANSWER: "

    # Token
    inputs = tokenizer(prompt_msg, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = generator_model.generate(**inputs, max_new_tokens=10)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Getting the Answer
    prediction_dict[key] = response
    logging.info(f"Prompt Answer: {response}")
    step_files += 1

    if (step_files % 100 == 0):
        filename = f"predicted_labels_{args.model_name}_{task_name}.json"
        with open(filename, "w") as f:
            json.dump(prediction_dict, f,indent=4)

filename = f"predicted_labels_{args.model_name}_{task_name}.json"
with open(filename, "w") as f:
    json.dump(prediction_dict, f,indent=4)



