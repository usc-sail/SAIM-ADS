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
from multi_task_dataset import *
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

config_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/configs/MHA_configs/config_MHA_multi_task_classifier_shot_level_multiple_seeds.yaml"

#load the config file
with open(config_file,'r') as f:
    config_data = yaml.safe_load(f)

csv_file=config_data['data']['csv_file']
topic_file=config_data['data']['topic_file']
csv_data=pd.read_csv(csv_file)

#task label map 
task_dict=config_data['parameters']['task_dict']
loss_dict=config_data['parameters']['loss']['loss_dict']
weight_dict=config_data['parameters']['loss']['weight_dict']


#create the label maps here 
with open(topic_file,'r') as f:
    topic_dict=json.load(f)

#complete label map for all tasks 
label_map={'Topic':topic_dict,
           'Transition_val':{'No transition':0,'Transition':1},
           'social_message':{'No':0,'Yes':1}}

sampled_label_map={k:label_map[k] for k in task_dict.keys()}

#basic parameters initialize regarding number of classes, max length of the sequence, fps, base fps, batch size, number of epochs, number of workers 
num_classes=config_data['model']['n_classes']
base_folder=config_data['data']['base_folder']
max_length=config_data['parameters']['max_length']
batch_size=config_data['parameters']['batch_size']
num_epochs=config_data['parameters']['epochs']
num_workers=config_data['parameters']['num_workers']
input_dim=config_data['model']['input_dim']
model_dim=config_data['model']['model_dim']
num_heads=config_data['model']['num_heads']
num_layers=config_data['model']['num_layers']
input_dropout=config_data['model']['input_dropout']
output_dropout=config_data['model']['output_dropout']
model_dropout=config_data['model']['model_dropout']
model_type=config_data['model']['model_type']
option=model_type+"_"+config_data['parameters']['task_name']
num_runs=config_data['parameters']['num_runs']
multi_run_folder=config_data['output']['multiple_run_folder']

#create the model type folder inside the multi run folder
destination_run_folder=os.path.join(multi_run_folder,model_type)
if not os.path.exists(destination_run_folder):
    os.mkdir(destination_run_folder)

#create random seeds for multiple runs 
seed_list = [random.randint(1, 100000) for _ in range(5)]

#test and validation metrics 
test_loss_multiple_seeds_list=[]
test_f1_multiple_seeds_list=[]
test_acc_multiple_seeds_list=[]
val_best_f1_multiple_seeds_list=[]
val_best_acc_multiple_seeds_list=[]

#device definition
if(config_data['device']['is_cuda']):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#seed value set 
dict_multiple_seeds={}

for i,seed in enumerate(seed_list):

    # global fixing here 
    np.random.seed(seed) # cpu vars
    torch.manual_seed(seed) # cpu  vars
    random.seed(seed) # Python
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('Run with random seed: %d' %(seed))

    #define the datasets 
    train_data=csv_data[csv_data['Split']=='train']
    val_data=csv_data[csv_data['Split']=='val']
    test_data=csv_data[csv_data['Split']=='test']


    #define the datasets
    train_ds=Multi_Task_Shot_Dataset(train_data,max_length,sampled_label_map,base_folder)