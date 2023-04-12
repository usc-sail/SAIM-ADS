import pickle 
import os 
import pandas as pd 
import numpy as np
from tqdm import tqdm

#load the non english transcripts
pkl_file="/data/digbose92/ads_complete_repo/ads_transcripts/combined_transcripts/en_combined_transcripts.pkl"
with open(pkl_file,'rb') as f:
    en_transcripts=pickle.load(f)

#load the non english transcripts
transcript_keys=list(en_transcripts.keys())
shape_list=[]
for key in tqdm(list(transcript_keys)):
    shape_list.append(en_transcripts[key].shape[0])
    #print(en_transcripts[key].shape)

print(np.mean(shape_list)) # 111.0
print(np.median(shape_list)) # 111.0
print(np.max(shape_list)) # 111.0
print(np.min(shape_list)) # 111.0
#75 and 25 percentile
print(np.percentile(shape_list,75)) # 111.0
print(np.percentile(shape_list,25)) # 111.0
