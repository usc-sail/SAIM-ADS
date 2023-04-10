import os 
import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse 
from tqdm import tqdm 
import pickle 

#source folder here 
source_folder="/scratch1/dbose/ads_repo/captions/csv_files"
file_list=os.listdir(source_folder)

#destination folder here
dest_folder="/scratch1/dbose/ads_repo/captions/embeddings/SBERT_embeddings"
dest_file_list=os.listdir(dest_folder)

#device and model here 
device='cuda:0'
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)

for file in tqdm(file_list):

    csv_file=os.path.join(source_folder,file)
    df=pd.read_csv(csv_file)
    dict_embeddings={}

    for i in range(df.shape[0]):

        caption=df['Caption'].iloc[i]
        shot_key=df['Shot_key'].iloc[i]
        sentence=[caption]

        embeddings = model.encode(sentence)
        dict_embeddings[shot_key]=embeddings

    dest_file=os.path.join(dest_folder,os.path.splitext(file)[0]+".pkl")

    with open(dest_file,'wb') as f:
        pickle.dump(dict_embeddings,f)

        

