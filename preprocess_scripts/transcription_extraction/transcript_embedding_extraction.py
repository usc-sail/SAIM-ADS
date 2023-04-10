import os 
import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse 
from tqdm import tqdm 
import pickle
import json 
import re

source_file="/scratch1/dbose/ads_repo/transcripts/en_combined_transcripts.json"
destination_folder="/scratch1/dbose/ads_repo/transcripts"


#device and model here 
device='cuda:0'
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)

#read the file
with open(source_file,'r') as f:
    data = json.load(f)

#extract the keys
key_list=list(data.keys())

#regular expression to break the text into sentences
regex = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

dict_embeddings=dict()

for key in tqdm(key_list):
    
    #current string
    string_c=data[key]

    #split into list of sentences 
    sentences = re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", string_c)

    #embed the sentences
    embeddings = model.encode(sentences)

    #print(embeddings.shape)
    dict_embeddings[key]=embeddings

#save the embeddings
with open(os.path.join(destination_folder,"en_combined_transcripts.pkl"),'wb') as f:
    pickle.dump(dict_embeddings,f)

