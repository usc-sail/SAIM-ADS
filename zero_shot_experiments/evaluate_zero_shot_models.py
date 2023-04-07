#evaluate zero shot model results using the non zero transcripts files 
import os 
import pandas as pd 
import numpy as np 
import json 
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix 
from tqdm import tqdm 

#argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--transcript_file', type=str, default='/bigdata/digbose92/ads_data/ads_complete_repo/ads_transcripts/translated_transcripts/en_combined_transcripts.json', help='path to the transcript file')
parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='name of the model')
parser.add_argument('--split_file', type=str, default='/data/digbose92/codes/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_no_zero_files.csv', help='path to the split file')
parser.add_argument('--label_file',type=str,required=True,help='path to the label mapping file')

#load the transcripts file
parse_args=parser.parse_args()
model=parse_args.model_name
transcript_file=parse_args.transcript_file
split_file=parse_args.split_file
label_file=parse_args.label_file

with open(transcript_file,'r') as f:
    pred_labels=json.load(f)

#load the label file
with open(label_file,'r') as f:
    label_map=json.load(f)

#load the split file
split_df=pd.read_csv(split_file)

test_data=split_df[split_df['Split']=='test']
topic_labels=list(test_data['Topic'])
test_data_keys=[os.path.splitext(test_data['video_file'].iloc[i])[0] for i in np.arange(test_data.shape[0])]

pred_keys=list(pred_labels.keys())
intersect_pred_keys=list(set(test_data_keys) & set(pred_keys))

#load the predictions
gt_topic_list=[]
pred_topic_list=[]

for key in tqdm(intersect_pred_keys):

    intersect_key=test_data_keys.index(key)
    gt_topic=topic_labels[intersect_key]
    pred_topic=pred_labels[key]['label']

    gt_topic_list.append(label_map[gt_topic])
    pred_topic_list.append(label_map[pred_topic])

#compute the accuracy

accuracy=accuracy_score(gt_topic_list,pred_topic_list)
print('Accuracy for the model {} is {}'.format(model,accuracy))

#compute the f1 score
f1=f1_score(gt_topic_list,pred_topic_list,average='weighted')
print('F1 score for the model {} is {}'.format(model,f1))

#compute the precision score





