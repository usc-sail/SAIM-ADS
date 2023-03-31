import torch
import torch.nn as nn 
import pandas as pd 
import os 
import sys 
import time 
import pickle
#append path of datasets and models 
sys.path.append(os.path.join('..', '..','..','datasets'))
sys.path.append(os.path.join('..', '..','..','models'))
sys.path.append(os.path.join('..', '..','..','configs'))
sys.path.append(os.path.join('..', '..','..','losses'))
sys.path.append(os.path.join('..', '..','..','optimizers'))
sys.path.append(os.path.join('..', '..','..','utils'))
sys.path.append(os.path.join('..','..'))

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


test_data=csv_data[csv_data['Split']=='test']
train_data=csv_data[csv_data['Split']=='train']
val_data=csv_data[csv_data['Split']=='val']

print(test_data.shape,train_data.shape,val_data.shape)


task_name=config_data['parameters']['task_name']
topic_file=config_data['data']['topic_file']

#load the topic file
with open(topic_file,'r') as f:
    label_map=json.load(f)

#parameters
num_classes=config_data['model']['n_classes']
max_length=config_data['parameters']['max_length']
fps=config_data['parameters']['fps']
base_fps=config_data['parameters']['base_fps']
batch_size=config_data['parameters']['batch_size']
num_workers=config_data['parameters']['num_workers']

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
criterion=multi_class_cross_entropy_loss(device)

#test statistics
test_loss,test_acc,test_f1=gen_validate_score_MHA_model_single_task_topic(model,test_dl,device,criterion)


print('Test loss: ',test_loss)
print('Test accuracy: ',test_acc)
print('Test f1 score: ',test_f1)






