import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd 
from collections import Counter
import math

#basic binary cross entropy loss
def binary_cross_entropy_loss(device,pos_weights=None,reduction='mean'):
  loss=nn.BCEWithLogitsLoss(reduction='mean',pos_weight=pos_weights).to(device)
  return(loss)

#multi class cross entropy loss
def multi_class_cross_entropy_loss(device,pos_weights=None,reduction='mean'):
  loss=nn.CrossEntropyLoss(reduction='mean',weight=pos_weights).to(device)
  return(loss)


