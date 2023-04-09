import torch
import torch.nn as nn 
import pandas as pd 
import os 
import sys 
import time 
import pickle

#append path of datasets and models 
sys.path.append(os.path.join('..', '..', '..','datasets'))
sys.path.append(os.path.join('..', '..', '..','models'))
sys.path.append(os.path.join('..', '..', '..','configs'))
sys.path.append(os.path.join('..', '..', '..','losses'))
sys.path.append(os.path.join('..', '..','..','optimizers'))
sys.path.append(os.path.join('..', '..','..','utils'))
sys.path.append(os.path.join('..', '..'))
#import all libraries 
import random
from ast import literal_eval
import torch
import yaml
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from dataset import *
from loss_functions import *
from MHA_models import *
from optimizer import *
from metrics import calculate_stats
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm 
from evaluate_model import *
import argparse
from log_file_generate import *
from scipy.stats.stats import pearsonr
import json
from statistics import mean

def load_config(config_file):

    with open(config_file,'r') as f:
        config_data=yaml.safe_load(f)
    return(config_data)


def test_model_soc_msg_tone(model_filename,config_data,device,seed_value):

    #fix seed for reproducibility
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model=torch.load(model_filename)
    model.to(device)

    csv_file=config_data['data']['csv_file']
    csv_data=pd.read_csv(csv_file)
    base_folder=config_data['data']['base_folder']

    #test_data,train_data,val_data
    test_data=csv_data[csv_data['Split']=='test']
    train_data=csv_data[csv_data['Split']=='train']
    val_data=csv_data[csv_data['Split']=='val']

    task_name=config_data['parameters']['task_name']
    print(test_data.shape,train_data.shape,val_data.shape)


    if(task_name=='social_message'):
        label_map={'No':0,'Yes':1}
        social_list=test_data['social_message'].tolist()
        social_list=[label_map[i] for i in social_list]
        counter_social_message=dict(Counter(social_list))
        majority_class=max(counter_social_message,key=counter_social_message.get)
        majority_class_accuracy=counter_social_message[majority_class]/len(social_list)
        #print('Majority class: ',majority_class)
        #compute f1 score with majority labels
        majority_class_labels=[majority_class]*len(social_list)
        f1_majority_class=f1_score(social_list,majority_class_labels,average='macro')

        print('Majority class accuracy: ',majority_class_accuracy)
        print('F1 score with majority class labels: ',f1_majority_class)

    else:
        label_map={'No transition':0,'Transition':1}
        transition_list=test_data['Transition_val'].tolist()
        transition_list=[label_map[i] for i in transition_list]
        counter_transition=dict(Counter(transition_list))
        majority_class=max(counter_transition,key=counter_transition.get)
        majority_class_accuracy=counter_transition[majority_class]/len(transition_list)
        majority_class_labels=[majority_class]*len(transition_list)
        f1_majority_class=f1_score(transition_list,majority_class_labels,average='macro')

        print('Majority class accuracy: ',majority_class_accuracy)
        print('F1 score with majority class labels: ',f1_majority_class)

    #parameters
    num_classes=config_data['model']['n_classes']
    max_length=config_data['parameters']['max_length']
    base_folder=config_data['data']['base_folder']
    batch_size=config_data['parameters']['batch_size']
    num_workers=config_data['parameters']['num_workers']

    #test_ds and test_dl
    test_ds=SAIM_single_task_dataset_shot_level(csv_data=test_data,
                                            label_map=label_map,
                                            num_classes=num_classes,
                                            max_length=max_length,
                                            base_folder=base_folder,
                                            task_name=task_name)

    test_dl=DataLoader(test_ds,
                                batch_size=batch_size,
                                shuffle=config_data['parameters']['train_shuffle'],
                                num_workers=num_workers)
        

    #loss function
    criterion=binary_cross_entropy_loss(device)

    #test statistics
    test_loss,test_acc,test_f1=gen_validate_score_MHA_model_single_task_soc_message_tone(model,test_dl,device,criterion)

    print('Test loss: ',test_loss)
    print('Test accuracy: ',test_acc)
    print('Test f1 score: ',test_f1)

    return(test_loss,test_acc,test_f1)
    

def test_model_topic(model_filename,config_data,device,seed_value):

    #fix seed for reproducibility
    #fix seed for reproducibility
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model=torch.load(model_filename)
    model.to(device)

    #csv file and data 
    csv_file=config_data['data']['csv_file']
    csv_data=pd.read_csv(csv_file)
    base_folder=config_data['data']['base_folder']

    #test_data,train_data,val_data
    test_data=csv_data[csv_data['Split']=='test']
    train_data=csv_data[csv_data['Split']=='train']
    val_data=csv_data[csv_data['Split']=='val']

    #task_name and topic_file
    task_name=config_data['parameters']['task_name']
    topic_file=config_data['data']['topic_file']

    #load the topic file
    with open(topic_file,'r') as f:
        label_map=json.load(f)

    #parameters
    num_classes=config_data['model']['n_classes']
    max_length=config_data['parameters']['max_length']
    batch_size=config_data['parameters']['batch_size']
    num_workers=config_data['parameters']['num_workers']

    #test dataset and dataloader
    test_ds=SAIM_single_task_dataset_shot_level(csv_data=test_data,
                                        base_folder=base_folder,
                                        label_map=label_map,
                                        num_classes=num_classes,
                                        max_length=max_length,
                                        task_name=task_name)
    test_dl=DataLoader(test_ds,
                            batch_size=batch_size,
                            shuffle=config_data['parameters']['train_shuffle'],
                            num_workers=num_workers)
    
    #loss function
    criterion=multi_class_cross_entropy_loss(device)

    #test statistics
    test_loss,test_acc,test_f1=gen_validate_score_MHA_model_single_task_topic(model,test_dl,device,criterion)

    return(test_loss,test_acc,test_f1)


log_dir="/data/digbose92/ads_complete_repo/ads_codes/model_files/recent_models/multi_run_log_dir"
model_dir="/data/digbose92/ads_complete_repo/ads_codes/model_files/recent_models/multi_run_model_dir"
folder_name="MHA_attn_single_task_classifier_CLIP_features_shot_level_multiple_seeds_Topic"
#"MHA_attn_single_task_classifier_CLIP_features_shot_level_multiple_seeds_social_message"
#"MHA_attn_single_task_classifier_CLIP_features_shot_level_multiple_seeds_Transition_val"

log_subfolder=os.path.join(log_dir,folder_name)
model_subfolder=os.path.join(model_dir,folder_name)

log_file_list=os.listdir(log_subfolder)
model_file_list=os.listdir(model_subfolder)

json_file="/data/digbose92/ads_complete_repo/ads_codes/model_files/multi_run_folder/MHA_attn_single_task_classifier_CLIP_features_shot_level_multiple_seeds/multi_run_MHA_attn_single_task_classifier_CLIP_features_shot_level_multiple_seeds_Topic_20230408-020418.json"
#"/data/digbose92/ads_complete_repo/ads_codes/model_files/multi_run_folder/MHA_attn_single_task_classifier_CLIP_features_shot_level_multiple_seeds/multi_run_MHA_attn_single_task_classifier_CLIP_features_shot_level_multiple_seeds_social_message_20230408-005612.json"
#"/data/digbose92/ads_complete_repo/ads_codes/model_files/multi_run_folder/MHA_attn_single_task_classifier_CLIP_features_shot_level_multiple_seeds/multi_run_MHA_attn_single_task_classifier_CLIP_features_shot_level_multiple_seeds_Transition_val_20230408-002603.json"

with open(json_file,'r') as f:
    run_data=json.load(f)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_list_f1=[]
_list_acc=[]


for run in list(run_data.keys()):

    run_dict=run_data[run]

    seed=run_dict['seed']

    timestamp=run_dict['timestamp']

    log_timestamp_folder=[filename for filename in log_file_list if timestamp in filename][0]
    yaml_file=os.path.join(log_subfolder,log_timestamp_folder,log_timestamp_folder+".yaml")

    model_timestamp_folder=[filename for filename in model_file_list if timestamp in filename][0]
    model_filename=os.path.join(model_subfolder,model_timestamp_folder,model_timestamp_folder+"_best_model.pt")

    config_data=load_config(yaml_file)
    #print(timestamp,seed,,model_filename)
    if(os.path.exists(yaml_file) and os.path.exists(model_filename)):
        #print(yaml_file,model_filename)
        test_loss,test_acc,test_f1=test_model_topic(model_filename,config_data,device,seed)

        print('Run: %s, Seed: %d, Test loss: %.4f, Test accuracy: %.4f, Test f1 score: %.4f'%(run,seed,test_loss,test_acc,test_f1))

        _list_f1.append(test_f1)
        _list_acc.append(test_acc)

print('Mean test f1 score: ',100*np.mean(_list_f1))
print('Mean test accuracy: ',100*np.mean(_list_acc))

#standard deviation
print('Standard deviation of test f1 score: ',100*np.std(_list_f1))
print('Standard deviation of test accuracy: ',100*np.std(_list_acc))



