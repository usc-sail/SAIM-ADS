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
from dataset import *
from loss_functions import *
from MHA_models import *
from optimizer import *
from metrics import calculate_stats
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm 
from evaluate_model import *
import argparse
from log_file_generate import *
from scipy.stats.stats import pearsonr
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json

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

parser=argparse.ArgumentParser()
parser.add_argument('--config_file', help='Location of configuration data', type=str, required=True)
parser.add_argument('--model_file', help='Location of model file', type=str, required=True)
args = vars(parser.parse_args())
config_file=args['config_file']
model_file=args['model_file']

config_data=load_config(config_file)

#csv file and data 
csv_file=config_data['data']['csv_file']
csv_data=pd.read_csv(csv_file)

#test_data,train_data,val_data
test_data=csv_data[csv_data['Split']=='test']
train_data=csv_data[csv_data['Split']=='train']
val_data=csv_data[csv_data['Split']=='val']

#task_name and topic_file
task_name=config_data['parameters']['task_name']
print(task_name)

 
if(task_name=='social_message'):
    label_map={'No':0,'Yes':1}
    social_list=test_data['social_message'].tolist()
    social_list=[label_map[i] for i in social_list]
    counter_social_message=dict(Counter(social_list))
    majority_class=max(counter_social_message,key=counter_social_message.get)
    majority_class_accuracy=counter_social_message[majority_class]/len(social_list)
    print('Majority class: ',majority_class)
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
fps=config_data['parameters']['fps']
base_fps=config_data['parameters']['base_fps']
batch_size=config_data['parameters']['batch_size']
num_workers=config_data['parameters']['num_workers']



#test_ds and test_dl
test_ds=SAIM_single_task_dataset(csv_data=test_data,
                                    label_map=label_map,
                                    num_classes=num_classes,
                                    max_length=max_length,
                                    fps=fps,
                                    base_fps=base_fps,
                                    task_name=task_name)

test_dl=DataLoader(test_ds,
                        batch_size=batch_size,
                        shuffle=config_data['parameters']['train_shuffle'],
                        num_workers=num_workers)

#load the model
model=torch.load(model_file)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#loss function
criterion=binary_cross_entropy_loss(device)

#test statistics
test_loss,test_acc,test_f1=gen_validate_score_MHA_model_single_task_soc_message_tone(model,test_dl,device,criterion)

print('Test loss: ',test_loss)
print('Test accuracy: ',test_acc)
print('Test f1 score: ',test_f1)

