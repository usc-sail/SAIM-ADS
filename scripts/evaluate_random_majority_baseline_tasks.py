import os 
import pandas as pd 
import numpy as np 
import json 
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score
import random
#computing majority and random baseline for social message 

def compute_majority_random_baseline_social_message(test_data,num_runs=5):

    label_map={'No':0,'Yes':1}
    social_list=test_data['social_message'].tolist()
    social_list=[label_map[i] for i in social_list]
    counter_social_message=dict(Counter(social_list))
    majority_class=max(counter_social_message,key=counter_social_message.get)
    # majority_class_accuracy=counter_social_message[majority_class]/len(social_list)
    majority_class_labels=[majority_class]*len(social_list)
    f1_majority_class=f1_score(social_list,majority_class_labels,average='macro')
    majority_class_accuracy=accuracy_score(social_list,majority_class_labels)
    print('++++++++++++++++++++++++++++++++++++++++++++ SOCIAL MESSAGE +++++++++++++++++++++++++++++++++++++++++++')
    print('Majority class accuracy for social message: ', 100*majority_class_accuracy)
    print('F1 score with majority class labels for social message: ',100*f1_majority_class)

    #assign a random label to each instance
    seed_list = [random.randint(1, 100000) for _ in range(5)]
    acc_random_list=[]
    f1_random_list=[]

    for i in np.arange(num_runs):

        #fix the seed 
        np.random.seed(seed_list[i])
        random.seed(seed_list[i])
        
        random_labels=np.random.randint(0,2,len(social_list))


        random_accuracy=accuracy_score(social_list,random_labels)
        random_f1=f1_score(social_list,random_labels,average='macro')

        acc_random_list.append(random_accuracy)
        f1_random_list.append(random_f1)

    print('Mean random accuracy for social message: ',100*np.mean(acc_random_list))
    print('Mean random f1 score for social message: ',100*np.mean(f1_random_list))
    #standard deviation
    print('Standard deviation for random accuracy for social message: ',100*np.std(acc_random_list))
    print('Standard deviation for random f1 score for social message: ',100*np.std(f1_random_list))


def compute_majority_random_baseline_transition(test_data,num_runs=5):

    label_map={'No transition':0,'Transition':1}
    transition_list=test_data['Transition_val'].tolist()
    transition_list=[label_map[i] for i in transition_list]
    counter_transition=dict(Counter(transition_list))
    majority_class=max(counter_transition,key=counter_transition.get)
    #majority_class_accuracy=counter_transition[majority_class]/len(transition_list)
    majority_class_labels=[majority_class]*len(transition_list)
    f1_majority_class=f1_score(transition_list,majority_class_labels,average='macro')
    majority_class_accuracy=accuracy_score(transition_list,majority_class_labels)

    print('++++++++++++++++++++++++++++++++++++++++++++ TRANSITION VAL +++++++++++++++++++++++++++++++++++++++++++')
    print('Majority class accuracy for transition: ',100*majority_class_accuracy)
    print('F1 score with majority class labels for transition: ',100*f1_majority_class)

    #assign a random label to each instance
    seed_list = [random.randint(1, 100000) for _ in range(5)]
    acc_random_list=[]
    f1_random_list=[]

    for i in np.arange(num_runs):

        #fix the seed 
        np.random.seed(seed_list[i])
        random.seed(seed_list[i])
        
        random_labels=np.random.randint(0,2,len(transition_list))


        random_accuracy=accuracy_score(transition_list,random_labels)
        random_f1=f1_score(transition_list,random_labels,average='macro')

        acc_random_list.append(random_accuracy)
        f1_random_list.append(random_f1)

    print('Mean random accuracy for transition: ',100*np.mean(acc_random_list))
    print('Mean random f1 score for transition: ',100*np.mean(f1_random_list))
    #standard deviation
    print('Standard deviation for random accuracy for transition: ',100*np.std(acc_random_list))
    print('Standard deviation for random f1 score for transition: ',100*np.std(f1_random_list))


def compute_majority_random_baseline_topic(test_data,label_map,num_runs=5):

    topic_list=test_data['Topic'].tolist()
    topic_list=[label_map[i] for i in topic_list]
    counter_topic=dict(Counter(topic_list))
    majority_class=max(counter_topic,key=counter_topic.get)
    majority_class_labels=[majority_class]*len(topic_list)

    f1_majority_class=f1_score(topic_list,majority_class_labels,average='macro')
    majority_class_accuracy=accuracy_score(topic_list,majority_class_labels)

    print('++++++++++++++++++++++++++++++++++++++++++++ TOPIC +++++++++++++++++++++++++++++++++++++++++++')
    print('Majority class accuracy for topic: ',100*majority_class_accuracy)
    print('F1 score with majority class labels for topic: ',100*f1_majority_class)

    #assign a random label to each instance

    seed_list = [random.randint(1, 100000) for _ in range(5)]
    acc_random_list=[]
    f1_random_list=[]

    for i in np.arange(num_runs):

        #fix the seed 
        np.random.seed(seed_list[i])
        random.seed(seed_list[i])
        
        random_labels=np.random.randint(0,len(label_map),len(topic_list))

        random_accuracy=accuracy_score(topic_list,random_labels)
        random_f1=f1_score(topic_list,random_labels,average='macro')

        acc_random_list.append(random_accuracy)
        f1_random_list.append(random_f1)

    print('Mean random accuracy for topic: ',100*np.mean(acc_random_list))
    print('Mean random f1 score for topic: ',100*np.mean(f1_random_list))
    #standard deviation
    print('Standard deviation for random accuracy for topic: ',100*np.std(acc_random_list))
    print('Standard deviation for random f1 score for topic: ',100*np.std(f1_random_list))



csv_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_no_zero_files.csv"
topic_file="/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/topic_list_18.json"
with open(topic_file,'r') as f:
    label_map=json.load(f)
test_data=pd.read_csv(csv_file)
compute_majority_random_baseline_social_message(test_data)
compute_majority_random_baseline_transition(test_data)
compute_majority_random_baseline_topic(test_data,label_map,num_runs=5)
    



