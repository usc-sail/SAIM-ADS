from tqdm import tqdm 
import numpy as np 
import torch 
from statistics import mean 
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats.stats import pearsonr
import sys
import os 

def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X, y, lengths

def gen_validate_score_LSTM_tone_transition_model(model,loader,device,criterion):

    print("starting validation")
    Sig = nn.Sigmoid()
    model.eval()
    target_labels=[]
    pred_labels=[]
    step=0
    val_loss_list=[]

    with torch.no_grad():
        for i, (vid_feat,label,lens) in enumerate(tqdm(loader)):

            vid_feat=vid_feat.float()
            label=label.float()
            vid_feat=vid_feat.to(device)
            label=label.to(device)

            vid_feat,label,lens = sort_batch(vid_feat,label,lens)
            logits=model(vid_feat,lens.cpu().numpy())
            logits_sig=Sig(logits)

            loss=criterion(logits,label)
            val_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            pred_labels.append(logits_sig.cpu())
            step=step+1

    target_label_val=torch.cat(target_labels).numpy()
    pred_label_val=torch.cat(pred_labels).numpy()
    #print(target_label_val.shape,pred_label_val.shape)
    pred_labels_discrete=np.where(pred_label_val>=0.5,1,0)
    
    #print(len(target_labels),len(pred_labels_discrete))
    val_acc=accuracy_score(target_label_val,pred_labels_discrete)
    val_f1=f1_score(target_label_val,pred_labels_discrete,average='macro')

    return(mean(val_loss_list),val_acc,val_f1)



