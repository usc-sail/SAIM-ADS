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
from statistics import mean
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

def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X, y, lengths # transpose (batch x seq_length) to (seq_length x batch)

def load_config(config_file):

    with open(config_file,'r') as f:
        config_data=yaml.safe_load(f)
    return(config_data)


