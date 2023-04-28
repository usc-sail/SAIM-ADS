#score fusion for perceiver model
import os 
import json 
import numpy as np
import argparse
import torch.nn as nn 
import torch
import pandas as pd
import pickle
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import json

def double_max_fusion(av_logits_i,tv_logits_i):
    av_max_ind=torch.max(av_logits_i,0)[1]
    tv_max_ind=torch.max(tv_logits_i,0)[1]

    if(av_logits_i[av_max_ind]>tv_logits_i[tv_max_ind]):
        return av_max_ind
    else:
        return tv_max_ind
    
def avg_max_fusion(av_logits_i,tv_logits_i):

    avg_logits= (av_logits_i+tv_logits_i)/2
    #find maximum over the max values
    max_ind=torch.max(avg_logits,0)[1]

    return(max_ind)

def flatten_list(lst):

    lst_set=[]

    for l in lst:
        lst_set.extend(l)

    return lst_set


av_file="/proj/digbose92/ads_repo/model_files/predictions/multi_run_Audio_visual_perceiver_single_task_classifier_shot_level_multiple_seeds_social_message_20230422-004711_logits.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Audio_visual_perceiver_topic_single_task_classifier_shot_level_multiple_seeds_Topic_20230422-010404_logits.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Audio_visual_perceiver_single_task_classifier_shot_level_multiple_seeds_social_message_20230422-004711_logits.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Audio_visual_perceiver_single_task_classifier_shot_level_multiple_seeds_Transition_val_20230422-012239_logits.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Audio_visual_perceiver_single_task_classifier_shot_level_multiple_seeds_social_message_20230422-004711_logits.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Audio_visual_perceiver_topic_single_task_classifier_shot_level_multiple_seeds_Topic_20230422-010404_logits.pkl" #audio visual file 
tv_file="/proj/digbose92/ads_repo/model_files/predictions/multi_run_Perceiver_single_task_classifier_shot_level_multiple_seeds_social_message_20230409-150706.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Perceiver_text_visual_single_task_classifier_shot_level_multiple_seeds_Topic_20230409-220327.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Perceiver_single_task_classifier_shot_level_multiple_seeds_social_message_20230409-150706.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Perceiver_single_task_classifier_shot_level_multiple_seeds_Transition_val_20230409-035050.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Perceiver_single_task_classifier_shot_level_multiple_seeds_social_message_20230409-150706.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Perceiver_text_visual_single_task_classifier_shot_level_multiple_seeds_Topic_20230409-220327.pkl" #text visual file
csv_file="/proj/digbose92/ads_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_no_zero_files.csv"
topic_file="/proj/digbose92/ads_repo/ads_codes/SAIM-ADS/data/topic_list_18.json"
#there will be 5 x 5 values for predictions for each task 

#read the predictions and the ground truth values
with open(av_file,'rb') as f:
    av_data=pickle.load(f)

with open(tv_file,'rb') as f:
    tv_data=pickle.load(f)

csv_data=pd.read_csv(csv_file)
test_data=csv_data[csv_data['Split']=='test']
task_name="social_message"
fusion_strategy="double_max"

### task name contains social message, Topic, transition_val
if(task_name=='social_message'):
    label_map={'No':0,'Yes':1}

elif(task_name=='Transition_val'):
    label_map={'No transition':0,'Transition':1}


elif(task_name=='Topic'):
    with open(topic_file) as f:
        label_map=json.load(f)

num_classes=len(label_map.keys())

#clip files 
clip_feature_list=test_data['clip_feature_path'].tolist()
clip_keys=[os.path.splitext(file.split("/")[-1])[0] for file in clip_feature_list]

#create the mapping between the clip keys and the ground truth labels
clip_label_map=dict(zip(clip_keys,test_data[task_name].tolist()))
print(clip_label_map)

#for i in tqdm(np.arange(test_data.shape[0])):

#start with number of runs of av and tv
# 
key_list=['run_'+str(i) for i in np.arange(5)]
total_dict_list={}
num_runs=0
sig=nn.Sigmoid()

for av_key in tqdm(key_list):
    av_logits=av_data[av_key]['logits']
    clip_key_av=av_data[av_key]['clip_key']
    av_seed=av_data[av_key]['seed']

    for tv_key in tqdm(key_list):
        #print(tv_key)
        tv_logits=tv_data[tv_key]['logits']
        clip_key_tv=tv_data[tv_key]['clip_key']
        if(len(clip_key_tv)==106): #very specific 
            #flattenthe list
            clip_key_tv=flatten_list(clip_key_tv)
            
        tv_seed=tv_data[tv_key]['seed']
        gt_list=[]
        pred_list=[]

        #LOOP THROUGH THE CLIP KEYS AND CHECK IF THEY ARE SAME
        for i,clip_key in tqdm(enumerate(clip_key_tv)): #loop over every sample 

            av_logits_i=av_logits[i]
            tv_logits_i=tv_logits[i]

            if(task_name=='Topic'):
                gt_list.append(label_map[clip_label_map[clip_key]])

                #combine the predictions from av and tv using maximum value after applying softmax
                av_logits_i=nn.Softmax(dim=0)(torch.from_numpy(av_logits_i))
                tv_logits_i=nn.Softmax(dim=0)(torch.from_numpy(tv_logits_i))

                if(fusion_strategy=='double_max'):
                    pred_label=double_max_fusion(av_logits_i,tv_logits_i)

                elif(fusion_strategy=='avg_max'):
                    pred_label=avg_max_fusion(av_logits_i,tv_logits_i)

                pred_list.append(pred_label.item())


            elif((task_name=='Transition_val') or (task_name=='social_message')):

                av_logits_i=sig(torch.from_numpy(av_logits_i))
                tv_logits_i=sig(torch.from_numpy(tv_logits_i))

                if(fusion_strategy=='double_max'):
                    pred_label=double_max_fusion(av_logits_i,tv_logits_i)

                elif(fusion_strategy=='avg_max'):
                    pred_label=avg_max_fusion(av_logits_i,tv_logits_i)

                
                # print(av_logits_i)
                # print(tv_logits_i)
                #print(clip_label_map[clip_key])
                label_c=label_map[clip_label_map[clip_key]]
                # ret_label=np.zeros((num_classes))
                # ret_label[label_c]=1

                # pred_vect=np.zeros((num_classes))
                # pred_vect[pred_label.item()]=1

                gt_list.append(label_c)
                pred_list.append(pred_label.item())

        #evaluate the predictions

        if(task_name=='Topic'):
            _acc=accuracy_score(gt_list,pred_list)
            _f1=f1_score(gt_list,pred_list,average='macro')

            #generate class wise f1 score and accuracy
            _f1_score_class_wise= f1_score(gt_list,pred_list,average=None)
            _acc_score_confusion=confusion_matrix(gt_list,pred_list)
            _acc_score_class_wise=_acc_score_confusion.diagonal()/np.sum(_acc_score_confusion,axis=1)

            #convert the class wise f1 score and accuracy to dictionary
            _f1_score_class_wise_dict=dict(zip(label_map.keys(),_f1_score_class_wise))
            _acc_score_class_wise_dict=dict(zip(label_map.keys(),_acc_score_class_wise))

        elif((task_name=='Transition_val') or (task_name=='social_message')):

            pred_array=np.array(pred_list)
            gt_array=np.array(gt_list)
            print(pred_array.shape,gt_array.shape)

            _acc=accuracy_score(gt_list,pred_list)
            _f1=f1_score(gt_list,pred_list,average='macro')

            #generate class wise f1 score and accuracy
            _f1_score_class_wise= f1_score(gt_list,pred_list,average=None)
            _acc_score_confusion=confusion_matrix(gt_list,pred_list)
            _acc_score_class_wise=_acc_score_confusion.diagonal()/np.sum(_acc_score_confusion,axis=1)

            #convert the class wise f1 score and accuracy to dictionary
            _f1_score_class_wise_dict=dict(zip(label_map.keys(),_f1_score_class_wise))
            _acc_score_class_wise_dict=dict(zip(label_map.keys(),_acc_score_class_wise))


        print('Av seed:',av_seed)
        print('Tv seed:',tv_seed)
        print('Accuracy:',_acc)
        print('F1 score:',_f1)

        dict_temp={
            'av_seed':av_seed,
            'tv_seed':tv_seed,
            'accuracy':_acc,
            'f1_score':_f1,
            'f1_score_class_wise':_f1_score_class_wise_dict,
            'acc_score_class_wise':_acc_score_class_wise_dict
        }

        num_runs+=1

        total_dict_list['run_'+str(num_runs)]=dict_temp


#save the total_dict_list
dest_filename='/proj/digbose92/ads_repo/model_files/predictions/class_wise/'+task_name+'_'+fusion_strategy+'_class_wise_recheck.json'
with open(dest_filename,'w') as f:
    json.dump(total_dict_list,f,indent=4)







            

        #now we have the logits for both av and tv

#task name will be from argparse 

# parser=argparse.ArgumentParser()
# parser.add_argument("--task_name",type=str,default="Topic",help="Topic")

# args=parser.parse_args()

# task_name=args.task_name


# csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_no_zero_files.csv"
# embedding_file="/data/digbose92/ads_complete_repo/ads_features/ast_embeddings/ast_embs_0.5.pkl"
# transcript_file="/data/digbose92/ads_complete_repo/ads_transcripts/combined_transcripts/en_combined_transcripts.json"
# topic_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/topic_list_18.json"


# #load the csv file
# csv_data=pd.read_csv(csv_file)
# test_data=csv_data[csv_data['Split']=='test']
# max_text_length=256
# max_audio_length=14
# max_video_length=35

# #label map for each task
# if(task_name=="Topic"):
#     with open(topic_file) as f:
#         label_map=json.load(f)

# elif(task_name=='Transition_val'):
#     label_map={'No transition':0,'Transition':1}

# elif(task_name=='social_message'):
#     label_map={'No':0,'Yes':1}



    

