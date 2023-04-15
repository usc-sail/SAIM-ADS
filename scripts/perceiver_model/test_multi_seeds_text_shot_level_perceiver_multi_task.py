import torch
import torch.nn as nn 
import pandas as pd 
import os 
import sys 
import time 
import pickle

#append path of datasets and models 
sys.path.append(os.path.join('..', '..','datasets'))
sys.path.append(os.path.join('..', '..','models'))
sys.path.append(os.path.join('..', '..','configs'))
sys.path.append(os.path.join('..', '..','losses'))
sys.path.append(os.path.join('..', '..','optimizers'))
sys.path.append(os.path.join('..', '..','utils'))
sys.path.append(os.path.join('..'))
#import all libraries 
import random
from ast import literal_eval
import torch
import yaml
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from multi_task_dataset import *
from loss_functions import *
from perceiver_model import *
from optimizer import *
from metrics import calculate_stats
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm 
from evaluate_multi_task_model import *
import argparse
from log_file_generate import *
from scipy.stats.stats import pearsonr
import json
from statistics import mean, stdev
import numpy as np 
from transformers import BertTokenizer, BertModel, BertConfig

def load_config(config_file):

    with open(config_file,'r') as f:
        config_data=yaml.safe_load(f)
    return(config_data)

def test_model(model_filename,config_data,device,seed_value,label_map,task_dict,weight_dict,loss_function_dict,activation_dict):

    #fix seed for reproducibility
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #load model + device 
    model=torch.load(model_filename)
    model.to(device)

    csv_file=config_data['data']['csv_file']
    topic_file=config_data['data']['topic_file']
    csv_data=pd.read_csv(csv_file)
    test_data=csv_data[csv_data['Split']=='test']
    transcript_file=config_data['data']['transcript_file']
    base_folder=config_data['data']['base_folder']
    model_name=config_data['model']['model_name']
    tokenizer=BertTokenizer.from_pretrained(model_name)

    #task label map 
    task_dict=config_data['parameters']['task_dict']
    weight_dict=config_data['loss']['weight_dict']

    #create the label maps here 
    with open(topic_file,'r') as f:
        topic_dict=json.load(f)

    #complete label map for all tasks 
    label_map={'Topic':topic_dict,
            'Transition_val':{'No transition':0,'Transition':1},
            'social_message':{'No':0,'Yes':1}}

    sampled_label_map={k:label_map[k] for k in task_dict.keys()}
    sampled_loss_function_dict={k:loss_function_dict[k] for k in task_dict.keys()}
    sampled_activation_dict={k:activation_dict[k] for k in task_dict.keys()}

    ## general parameters 
    #parameters regarding number of classes, maximum audio length, maximum video length
    max_text_length=config_data['parameters']['text_max_length']
    max_video_length=config_data['parameters']['video_max_length']
    batch_size=config_data['parameters']['batch_size']
    num_epochs=config_data['parameters']['epochs']
    num_workers=config_data['parameters']['num_workers']

    test_ds=SAIM_multi_task_dataset_visual_text_shot_level(test_data,
                                                                transcript_file,tokenizer,
                                                                base_folder,sampled_label_map,
                                                                max_text_length,max_video_length)
    
    test_dl=DataLoader(test_ds,
                                batch_size=batch_size,
                                shuffle=config_data['parameters']['test_shuffle'],
                                num_workers=num_workers)
    
    test_loss,test_loss_dict,test_f1_dict,test_acc_dict=gen_validate_score_text_visual_perceiver_multi_task(model,test_dl,device,task_dict,sampled_activation_dict,sampled_loss_function_dict,weight_dict)

    return(test_loss,test_loss_dict,test_f1_dict,test_acc_dict)

argparser = argparse.ArgumentParser()
argparser.add_argument('--log_dir', required=True)
argparser.add_argument('--model_dir', required=True)
argparser.add_argument('--folder_name', required=True)
argparser.add_argument('--json_file', required=True)

args=argparser.parse_args()

log_dir=args.log_dir
model_dir=args.model_dir
folder_name=args.folder_name
json_file=args.json_file

log_subfolder=os.path.join(log_dir,folder_name)
model_subfolder=os.path.join(model_dir,folder_name)

log_file_list=os.listdir(log_subfolder)
model_file_list=os.listdir(model_subfolder)

with open(json_file,'r') as f:
    run_data=json.load(f)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_list_f1=[]
_list_acc=[]

loss_function_dict={'Topic': multi_class_cross_entropy_loss(device),
                    'Transition_val':binary_cross_entropy_loss(device),
                    'social_message':binary_cross_entropy_loss(device)}

activation_dict={'Topic':nn.Softmax(dim=-1),
                 'Transition_val':nn.Sigmoid(),
                'social_message':nn.Sigmoid()}

_list_f1_dict=dict()
_list_acc_dict=dict()


for run in list(run_data.keys()):

    run_dict=run_data[run]

    #seed here 
    seed=run_dict['seed']

    #timestamp
    timestamp=run_dict['timestamp']

    log_timestamp_folder=[filename for filename in log_file_list if timestamp in filename][0]
    yaml_file=os.path.join(log_subfolder,log_timestamp_folder,log_timestamp_folder+".yaml")

    model_timestamp_folder=[filename for filename in model_file_list if timestamp in filename][0]
    model_filename=os.path.join(model_subfolder,model_timestamp_folder,model_timestamp_folder+"_best_model.pt")

    #config data 
    config_data=load_config(yaml_file)
    topic_file=config_data['data']['topic_file']
    
    #topic file here and topic dictionary
    with open(topic_file,'r') as f:
        topic_dict=json.load(f)

    #label map here 
    label_map={'Topic':topic_dict,
           'Transition_val':{'No transition':0,'Transition':1},
           'social_message':{'No':0,'Yes':1}}
    
    #task and weight dictionaries
    task_dict=config_data['parameters']['task_dict']
    weight_dict=config_data['loss']['weight_dict']

    if(os.path.exists(yaml_file) and os.path.exists(model_filename)):

        test_loss,test_loss_dict,test_f1_dict,test_acc_dict=test_model(model_filename,config_data,device,seed,label_map,task_dict,weight_dict,loss_function_dict,activation_dict)

        for k in task_dict.keys():
            print('Task:{},Test loss:{:.3f},Test F1 score:{:.3f}, Test Acc score:{:.3f}'.format(k, mean(test_loss_dict[k]), test_f1_dict[k],test_acc_dict[k]))

        _list_f1_dict['run_'+str(run)]=test_f1_dict

        _list_acc_dict['run_'+str(run)]=test_acc_dict

 
for task in list(task_dict.keys()):
    list_f1_task=[]
    list_acc_task=[]
    for k in list(_list_f1_dict.keys()):
        list_f1_task.append(_list_f1_dict[k][task])
        list_acc_task.append(_list_acc_dict[k][task])

    print('Task:{},mean F1 score:{:.3f}, std F1 score:{:.3f}'.format(task, mean(list_f1_task)*100, stdev(list_f1_task)*100))
    print('Task:{},mean acc score:{:.3f}, std acc score:{:.3f}'.format(task, mean(list_acc_task)*100, stdev(list_acc_task)*100))


        
                  











