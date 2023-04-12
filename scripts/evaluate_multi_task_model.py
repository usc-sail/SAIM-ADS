from tqdm import tqdm 
import numpy as np 
import torch 
from statistics import mean 
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from scipy.stats.stats import pearsonr
import sys
import os 
from collections import Counter

def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X, y, lengths

def gen_validate_score_MHA_multi_task(model,loader,device,task_dict,sampled_activation_dict,sampled_loss_function_dict):

    print("starting validation")

    model.eval()
    val_loss_dict={k:[] for k in task_dict.keys()}
    target_labels_dict={k:[] for k in task_dict.keys()}
    pred_labels_dict={k:[] for k in task_dict.keys()}
    pred_labels_np={k:[] for k in task_dict.keys()}
    target_labels_np={k:[] for k in task_dict.keys()}
    val_loss_list=[]
    f1_score_dict={k:[] for k in task_dict.keys()}
    acc_score_dict={k:[] for k in task_dict.keys()}

    with torch.no_grad():

        for id,(feat,mask,label_dict) in enumerate(tqdm(loader)):

            #forward pass 
            feat=feat.to(device)
            feat=feat.float()

            #label dictionary
            for task in label_dict.keys():
                label_dict[task]=label_dict[task].to(device)

            #mask here 
            mask=mask.unsqueeze(1).unsqueeze(1)
            mask=mask.to(device)

            #task logits
            task_logits=model(feat,mask)

            #loss, val loss list
            loss=torch.tensor(0.0).to(device)
            val_loss_list.append(loss.item())

            #train loss dictionary
            for task in task_dict.keys():
                loss_task=sampled_loss_function_dict[task](task_logits[task],label_dict[task])
                loss+=loss_task
                val_loss_dict[task].append(loss_task.item())

            #obtain the logits here 
            for k in task_dict.keys():
                activation_logits_task=sampled_activation_dict[k](task_logits[k])
                pred_labels_dict[k].append(activation_logits_task.cpu())
                target_labels_dict[k].append(label_dict[k].cpu())

    #agrgegate the labels here
    for k in task_dict.keys():
            
        pred_lb_np=torch.cat(pred_labels_dict[k]).detach().numpy()
        if((k=='social_message') or (k=='Transition_val')):
            pred_lb_np=np.where(pred_lb_np>=0.5,1,0)

        #pred and target labels
        pred_labels_np[k]=pred_lb_np
        target_labels_np[k]=torch.cat(target_labels_dict[k]).detach().numpy()    

        #f1 score and accuracy score
        f1_score_dict[k]=f1_score(target_labels_np[k],pred_labels_np[k],average='macro')
        acc_score_dict[k]=accuracy_score(target_labels_np[k],pred_labels_np[k])

    return(mean(val_loss_list),val_loss_dict,f1_score_dict,acc_score_dict)

    

            

            


