#codebase to use GPT-4 to translate non english transcripts using openai api
import os 
import json
import pandas as pd 
import argparse 
import openai
from tqdm import tqdm 
import numpy as np

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

parser=argparse.ArgumentParser()
parser.add_argument("--api_key_file",type=str,help="file containing the openai api key")
parser.add_argument("--transcript_file",type=str,help="file containing the non english transcripts")
parser.add_argument("--dest_folder",type=str,help="destination folder to save the translated transcripts")
#parser.add_argument("--split_file",type=str,help="file containing the train/test/val split") #not using split file for now since we need to run multiple evaluations
parser.add_argument('--task_name',type=str,help="name of the task")
parser.add_argument('--csv_file',type=str,help="csv file to use")

#arguments 
args=parser.parse_args()
key_file=args.api_key_file
transcript_file=args.transcript_file
dest_folder=args.dest_folder
task_name=args.task_name
csv_file=args.csv_file

csv_data=pd.read_csv(csv_file)
test_data=csv_data[csv_data['Split']=='test']
file_paths=list(test_data['clip_feature_path'])
file_keys=[os.path.splitext(f.split("/")[-1])[0] for f in file_paths]


with open(key_file) as f:
    api_key=f.readlines()
openai.api_key=api_key[0].strip().split("\n")[0]

#assign the parameters for the translation
model_option="gpt-4"
temperature=0.02
max_tokens=2048
top_p=1

if(task_name=="Topic"):
    system_string="Associate a single topic label with the transcript from the given set: Games, Household, Services, Sports, Banking, Clothing, Industrial and agriculture, Leisure, Publications media, Health, Car, Electronics, Cosmetics, Food and drink, Awareness, Travel and transport, Retail"

elif(task_name=="Social_message"):
    system_string="An advertisement video has a social message if it provides awareness about any social issue. Example of social issues: gender equality, drug abuse, police brutality, workplace harassment, domestic violence, child labor, environmental damage, homelessness, hate crimes, racial inequality etc. Based on the given text transcript, determine if the advertisement has any social message. Please provide answers in Yes and No."

elif(task_name=="Tone_transition"):
    system_string="Based on the given text transcript from the advertisement, determine if the advertisement has any transitions in tones. Possible tone labels are: positive, negative, and neutral. Please respond saying \"Transition\" or \"No transition\"."

#load the transcript file
with open(transcript_file) as f:
    transcript_data=json.load(f)

#intersect_keys=list(set(file_keys) & set(list(transcript_data.keys())))

#difference keys
diff_keys=list(set(list(transcript_data.keys()))-set(file_keys))

#print(len(intersect_keys))
output_dict={}

num_files=0
for key in tqdm(diff_keys):

    prompt=transcript_data[key]

    #messages list
    messages=[{"role": "system", "content": system_string},
              {"role": "user", "content": prompt}]
    
    #chat completion response
    response=completion_with_backoff(
        model=model_option,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )

    #answer 
    answer=response['choices'][0]['message']['content']

    #token usage 
    token_usage= response['usage']['total_tokens']

    #temp dict
    temp_dict={"answer":answer,"token_usage":token_usage}

    #save the output
    output_dict[key]=temp_dict

    #number of files 
    num_files=num_files+1

    if(num_files%50==0):
        #save a running snapshot of the file 
        destination_file=dest_folder+"/"+model_option+"_"+transcript_file.split("/")[-1].split(".")[0]+"_zero_shot_"+str(task_name)+"_"+str(num_files)+".json"

        with open(destination_file, 'w') as fp:
            json.dump(output_dict, fp)

#save the output
output_file=dest_folder+"/"+model_option+"_"+transcript_file.split("/")[-1].split(".")[0]+"_zero_shot_"+str(task_name)+".json"

with open(output_file,'w') as f:
    json.dump(output_dict,f,indent=4)


