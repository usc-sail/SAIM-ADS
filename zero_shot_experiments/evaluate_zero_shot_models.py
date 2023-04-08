#evaluate zero shot model results using the non zero transcripts files 
import os 
import pandas as pd 
import numpy as np 
import json 
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix 
from tqdm import tqdm 

def generate_topic_classification_report(intersect_keys,total_keys,label_map,topic_labels,pred_labels):

    gt_topic_list=[]
    pred_topic_list=[]
    cnt_num=0

    for key in tqdm(intersect_keys):

        intersect_key=total_keys.index(key)
        gt_topic=topic_labels[intersect_key]

        pred_topic=pred_labels[key]['answer']
        if(pred_topic not in label_map.keys()):

            #assign topic at random
            pred_topic_list.append(label_map[np.random.choice(list(label_map.keys()))])
            cnt_num+=1
            #pred_topic=pred_labels[key]['label']
        else:
            pred_topic_list.append(label_map[pred_topic])
        gt_topic_list.append(label_map[gt_topic])

    return(gt_topic_list,pred_topic_list)
    
def generate_social_message_classification_report(intersect_keys,total_keys,label_map, sm_labels,pred_labels):

    gt_sm_list=[]
    pred_sm_list=[]
    cnt_num=0

    for key in tqdm(intersect_keys):

        intersect_key=total_keys.index(key)
        gt_sm=sm_labels[intersect_key]

        pred_sm=pred_labels[key]['answer']
        if(pred_sm not in label_map.keys()):

            #assign topic at random
            pred_sm_list.append(label_map[np.random.choice(list(label_map.keys()))])
            cnt_num+=1
        else:
            pred_sm_list.append(label_map[pred_sm])
            
        gt_sm_list.append(label_map[gt_sm])

    return(gt_sm_list,pred_sm_list)
    

#argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--transcript_file', type=str, default='/bigdata/digbose92/ads_data/ads_complete_repo/ads_transcripts/translated_transcripts/en_combined_transcripts.json', help='path to the transcript file')
parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='name of the model')
parser.add_argument('--split_file', type=str, default='/data/digbose92/codes/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_no_zero_files.csv', help='path to the split file')
parser.add_argument('--task_name',type=str,required=True,help='name of the task')

#load the transcripts file
parse_args=parser.parse_args()

#model bname, transcript file, split file, label file, task name
model=parse_args.model_name
transcript_file=parse_args.transcript_file
split_file=parse_args.split_file
task_name=parse_args.task_name

with open(transcript_file,'r') as f:
    pred_labels=json.load(f)


#load the split file
split_df=pd.read_csv(split_file)
test_data=split_df[split_df['Split']=='test']
test_data_keys=[os.path.splitext(test_data['video_file'].iloc[i])[0] for i in np.arange(test_data.shape[0])]

#task specific name
if(task_name=="Topic"):

    topic_labels=list(test_data['Topic']) #Topic list
    label_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/topic_list_18.json"
    with open(label_file, 'r') as f:
        label_map=json.load(f)
    
elif(task_name=="social_message"):

    social_message_labels=list(test_data['social_message']) #social message list
    label_map={'No':0,'Yes':1}

#predicted and intersecting keys
pred_keys=list(pred_labels.keys())
intersect_pred_keys=list(set(test_data_keys) & set(pred_keys))
print(len(intersect_pred_keys))
#load the predictions
gt_topic_list=[]
pred_topic_list=[]
cnt_num=0

#task names
if(task_name=="Topic"):
    gt_list,pred_list=generate_topic_classification_report(intersect_pred_keys,test_data_keys,label_map,topic_labels,pred_labels)

elif(task_name=="social_message"):
    gt_list,pred_list=generate_social_message_classification_report(intersect_pred_keys,test_data_keys,label_map,social_message_labels,pred_labels)

print(len(gt_list),len(pred_list))
#compute the accuracy
accuracy=accuracy_score(gt_list,pred_list)
print('Accuracy for the model {} is {}'.format(model,accuracy))

#compute the f1 score
f1=f1_score(gt_list,pred_list,average='weighted')
print('F1 score for the model {} is {}'.format(model,f1))

#compute the precision score





