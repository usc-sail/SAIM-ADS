#codebase to use GPT-4 to translate non english transcripts using openai api
import os 
import json
import pandas as pd 
import argparse 
import openai
from tqdm import tqdm 

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

args=parser.parse_args()
key_file=args.api_key_file
transcript_file=args.transcript_file
dest_folder=args.dest_folder

with open(key_file) as f:
    api_key=f.readlines()


openai.api_key=api_key[0].strip().split("\n")[0]

#assign the parameters for the translation
model_option="gpt-4"
temperature=0.02
max_tokens=2048
top_p=1
translate_string="Please provide an English translation of this transcript"

#load the transcript file
with open(transcript_file) as f:
    transcript_data=json.load(f)

output_dict={}

num_files=0
#keys in transcript data 
for key in tqdm(transcript_data.keys()):

    prompt=transcript_data[key]

    #messages list
    messages=[{"role": "system", "content": translate_string},
              {"role": "user", "content": prompt}]
    
    #chat completion response
    response=completion_with_backoff(
        model=model_option,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )

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
        destination_file=dest_folder+"/"+model_option+"_"+transcript_file.split("/")[-1].split(".")[0]+"_zero_shot_"+str(num_files)+".json"

        with open(destination_file, 'w') as fp:
            json.dump(output_dict, fp)


#save the output
output_file=dest_folder+"/"+model_option+"_"+transcript_file.split("/")[-1].split(".")[0]+"_translated.json"

with open(output_file,'w') as f:
    json.dump(output_dict,f,indent=4)

