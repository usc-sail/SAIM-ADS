#read every logits performance of the av and tv model 
#for each seed run the evaluation metrics (total and class wise)
#then take average of the metrics and save it as av dict and tv dict

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


def flatten_list(lst):

    lst_set=[]

    for l in lst:
        lst_set.extend(l)

    return lst_set


def generate_class_wise_performance_av_data(av_data,label_map, clip_label_map, dest_folder,task_name):

    key_list=['run_'+str(i) for i in np.arange(5)]
    f1_score_av_list=[]
    acc_score_av_list=[]
    run_wise_av_dict=dict()

    #first run for the av file 
    for av_key in tqdm(key_list):

        av_logits=av_data[av_key]['logits']
        av_seed=av_data[av_key]['seed']
        clip_key_av=av_data[av_key]['clip_key']
        av_seed=av_data[av_key]['seed']
        gt_list=[]
        pred_list=[]
        
        for i,clip_key in tqdm(enumerate(clip_key_av)): #loop over every sample 

            gt_list.append(label_map[clip_label_map[clip_key]])
            av_logits_i=av_logits[i]
            #gt_list.append(label_map[clip_label_map[clip_key]])

            if(task_name=='Topic'):
                
                    #combine the predictions from av and tv using maximum value after applying softmax
                    av_logits_i=nn.Softmax(dim=0)(torch.from_numpy(av_logits_i))

                    #take the maximum value and the corresponding index
                    av_max_val,av_max_idx=torch.max(av_logits_i,dim=0)

                    pred_list.append(av_max_idx.item())

            elif(task_name=='Transition_val' or task_name=='social_message'):

                #apply sigmoid 
                av_logits_i=sig(torch.from_numpy(av_logits_i))
                
                
                av_logits_i=av_logits_i.cpu().detach().numpy()


                #greater than equal to 0.5 to 1 else 0
                av_logits_i=np.where(av_logits_i>=0.5,1,0)

                #take the maximum value and the corresponding index from numpy array

                av_max_idx=np.argmax(av_logits_i)

                pred_list.append(av_max_idx)

        #convert the gt and pred list to numpy array
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
        print('F1 score:',_f1)
        print('Acc score:',_acc)

        #append to list
        f1_score_av_list.append(_f1)
        acc_score_av_list.append(_acc)

        #append to dictionary

        dict_temp={
                'seed':av_seed,
                'accuracy':_acc,
                'f1_score':_f1,
                'f1_score_class_wise':_f1_score_class_wise_dict,
                'acc_score_class_wise':_acc_score_class_wise_dict
            }
        
        run_wise_av_dict[av_key]=dict_temp

    #print mean and std 
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ AV results +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Mean F1 score:',np.mean(f1_score_av_list)*100)
    print('Std F1 score:',np.std(f1_score_av_list)*100)
    print('Mean Acc score:',np.mean(acc_score_av_list)*100)
    print('Std Acc score:',np.std(acc_score_av_list)*100)

    #save the av results
    av_result_file=os.path.join(dest_folder,'av_result_'+task_name+'.json')
    with open(av_result_file,'w') as f:
        json.dump(run_wise_av_dict,f,indent=4)


def generate_class_wise_performance_tv_data(tv_data,label_map, clip_label_map, dest_folder,task_name):


    #then run for the tv file 
    key_list=['run_'+str(i) for i in np.arange(5)]
    f1_score_tv_list=[]
    acc_score_tv_list=[]
    run_wise_tv_dict=dict()

    #first run for the av file 
    for tv_key in tqdm(key_list):

        tv_logits=tv_data[tv_key]['logits']
        tv_seed=tv_data[tv_key]['seed']
        clip_key_tv=tv_data[tv_key]['clip_key']
        if(len(clip_key_tv)==106): #very specific 
            #flattenthe list
            clip_key_tv=flatten_list(clip_key_tv)
        tv_seed=tv_data[tv_key]['seed']
        gt_list=[]
        pred_list=[]
        
        for i,clip_key in tqdm(enumerate(clip_key_tv)): #loop over every sample 

            gt_list.append(label_map[clip_label_map[clip_key]])
            tv_logits_i=tv_logits[i]
            #gt_list.append(label_map[clip_label_map[clip_key]])

            if(task_name=='Topic'):
                
                    #combine the predictions from av and tv using maximum value after applying softmax
                    tv_logits_i=nn.Softmax(dim=0)(torch.from_numpy(tv_logits_i))

                    #take the maximum value and the corresponding index
                    tv_max_val,tv_max_idx=torch.max(tv_logits_i,dim=0)

                    pred_list.append(tv_max_idx.item())

            elif(task_name=='Transition_val' or task_name=='social_message'):

                #apply sigmoid 
                tv_logits_i=sig(torch.from_numpy(tv_logits_i))
                
                
                tv_logits_i=tv_logits_i.cpu().detach().numpy()


                #greater than equal to 0.5 to 1 else 0
                tv_logits_i=np.where(tv_logits_i>=0.5,1,0)

                #take the maximum value and the corresponding index from numpy array

                tv_max_idx=np.argmax(tv_logits_i)

                pred_list.append(tv_max_idx)

        #convert the gt and pred list to numpy array
        _acc=accuracy_score(gt_list,pred_list)
        _f1=f1_score(gt_list,pred_list,average='macro')

        #generate class wise f1 score and accuracy
        _f1_score_class_wise= f1_score(gt_list,pred_list,average=None)
        _acc_score_confusion=confusion_matrix(gt_list,pred_list)
        _acc_score_class_wise=_acc_score_confusion.diagonal()/np.sum(_acc_score_confusion,axis=1)

        #convert the class wise f1 score and accuracy to dictionary
        _f1_score_class_wise_dict=dict(zip(label_map.keys(),_f1_score_class_wise))
        _acc_score_class_wise_dict=dict(zip(label_map.keys(),_acc_score_class_wise))

        print('TV seed:',tv_seed)
        print('F1 score:',_f1)
        print('Acc score:',_acc)

        #append to list
        f1_score_tv_list.append(_f1)
        acc_score_tv_list.append(_acc)

        dict_temp_tv={
                'seed':tv_seed,
                'accuracy':_acc,
                'f1_score':_f1,
                'f1_score_class_wise':_f1_score_class_wise_dict,
                'acc_score_class_wise':_acc_score_class_wise_dict
            }
        
        run_wise_tv_dict[tv_key]=dict_temp_tv

    #print mean and std
    #print mean and std 
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ TV results +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Mean F1 score:',np.mean(f1_score_tv_list)*100)
    print('Std F1 score:',np.std(f1_score_tv_list)*100)
    print('Mean Acc score:',np.mean(acc_score_tv_list)*100)
    print('Std Acc score:',np.std(acc_score_tv_list)*100)


    #save the av results
    tv_result_file=os.path.join(dest_folder,'tv_result_'+task_name+'.json')
    with open(tv_result_file,'w') as f:
        json.dump(run_wise_tv_dict,f,indent=4)



task_name="social_message"
csv_file="/proj/digbose92/ads_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_no_zero_files.csv"
topic_file="/proj/digbose92/ads_repo/ads_codes/SAIM-ADS/data/topic_list_18.json"
dest_folder="/proj/digbose92/ads_repo/model_files/predictions/class_wise"
#there will be 5 x 5 values for predictions for each task 


csv_data=pd.read_csv(csv_file)
test_data=csv_data[csv_data['Split']=='test']
sig=nn.Sigmoid()


av_file="/proj/digbose92/ads_repo/model_files/predictions/multi_run_Audio_visual_perceiver_single_task_classifier_shot_level_multiple_seeds_social_message_20230422-004711_logits.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Audio_visual_perceiver_topic_single_task_classifier_shot_level_multiple_seeds_Topic_20230422-010404_logits.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Audio_visual_perceiver_single_task_classifier_shot_level_multiple_seeds_Transition_val_20230422-012239_logits.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Audio_visual_perceiver_single_task_classifier_shot_level_multiple_seeds_social_message_20230422-004711_logits.pkl"
tv_file="/proj/digbose92/ads_repo/model_files/predictions/multi_run_Perceiver_single_task_classifier_shot_level_multiple_seeds_social_message_20230409-150706.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Perceiver_text_visual_single_task_classifier_shot_level_multiple_seeds_Topic_20230409-220327.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Perceiver_single_task_classifier_shot_level_multiple_seeds_Transition_val_20230409-035050.pkl"
#"/proj/digbose92/ads_repo/model_files/predictions/multi_run_Perceiver_single_task_classifier_shot_level_multiple_seeds_social_message_20230409-150706.pkl"

#read the predictions and the ground truth values
with open(av_file,'rb') as f:
    av_data=pickle.load(f)

with open(tv_file,'rb') as f:
    tv_data=pickle.load(f)


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


generate_class_wise_performance_av_data(av_data,label_map, clip_label_map, dest_folder,task_name)
generate_class_wise_performance_tv_data(tv_data,label_map, clip_label_map, dest_folder,task_name)



