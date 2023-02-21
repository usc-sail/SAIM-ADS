import torch
import torch.nn as nn 
import pandas as pd 
import os 
import sys 
import time 
import pickle
#append path of datasets and models 
sys.path.append(os.path.join('..', 'datasets'))
sys.path.append(os.path.join('..', 'models'))
sys.path.append(os.path.join('..', 'configs'))
sys.path.append(os.path.join('..', 'losses'))
sys.path.append(os.path.join('..', 'optimizers'))
sys.path.append(os.path.join('..', 'utils'))

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
from LSTM_models import *
from optimizer import *
from metrics import calculate_stats
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm 
from evaluate_model import *
import argparse
from log_file_generate import *
from scipy.stats.stats import pearsonr
import wandb

#fix seed for reproducibility
seed_value=123457
np.random.seed(seed_value) # cpu vars
torch.manual_seed(seed_value) # cpu  vars
random.seed(seed_value) # Python
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_config(config_file):
    with open(config_file,'r') as f:
        config_data=yaml.safe_load(f)
    return(config_data)

config_file="/data/digbose92/ads_complete_repo/ads_codes/model_files/recent_models/log_dir/LSTM_multi_layer_tone_transition_model/20230220-232417_LSTM_multi_layer_tone_transition_model/20230220-232417_LSTM_multi_layer_tone_transition_model.yaml"
config_data=load_config(config_file)
label_map={'No transition':0,'Transition':1}
csv_file=config_data['data']['csv_file']
csv_data=pd.read_csv(csv_file)
test_data=csv_data[csv_data['Split']=='test']
max_length=config_data['parameters']['max_length']
batch_size=config_data['parameters']['batch_size']
num_workers=config_data['parameters']['num_workers']
num_classes=config_data['model']['n_classes']
fps=config_data['parameters']['fps']
base_fps=config_data['parameters']['base_fps']
transition_list=test_data['Transition_val'].tolist()
transition_list=[label_map[i] for i in transition_list]
counter_transition=dict(Counter(transition_list))
majority_class=max(counter_transition,key=counter_transition.get)
majority_class_accuracy=counter_transition[majority_class]/len(transition_list)


#compute f1 score with majority labels
majority_class_labels=[majority_class]*len(transition_list)
f1_majority_class=f1_score(transition_list,majority_class_labels,average='macro')

print('Majority class accuracy: ',majority_class_accuracy)
print('F1 score with majority class labels: ',f1_majority_class)
#compute the majority accuracy over transition list 

#load the model
model_file="/data/digbose92/ads_complete_repo/ads_codes/model_files/recent_models/model_dir/LSTM_multi_layer_tone_transition_model/20230220-232417_LSTM_multi_layer_tone_transition_model/20230220-232417_LSTM_best_model.pt"
model=torch.load(model_file)
model.eval()
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion= binary_cross_entropy_loss(device,pos_weights=None)

#create test dataloader
test_ds=SAIM_ads_tone_clip_features_dataset(test_data,
                    label_map,
                    num_classes,
                    max_length,
                    fps,base_fps)


test_dl=DataLoader(test_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers)

#compute the accuracy and f1 score
test_loss,test_acc,test_f1=gen_validate_score_LSTM_tone_transition_model(model,test_dl,device,criterion)

print('Test loss: ',test_loss)
print('Test accuracy: ',test_acc)
print('Test f1 score: ',test_f1)




