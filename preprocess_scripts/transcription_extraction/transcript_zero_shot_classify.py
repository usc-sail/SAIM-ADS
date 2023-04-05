from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
from transformers import pipeline
from tqdm import tqdm 
#nli model - facebook/bart-large-mnli
# nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
# tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
device='cuda:0' if torch.cuda.is_available() else 'cpu'
# nli_model=nli_model.to(device)
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli",device=device)

#labels
label_list=['Games', 'Household', 
'Services', 'Sports', 
'Banking', 'Clothing', 'Industrial and agriculture', 'Leisure', 'Publications media', 'Health', 'Car', 'Electronics', 'Cosmetics', 'Food and drink', 'Awareness', 'Travel and transport', 'Retail']

#load the transcripts file 
with open("/bigdata/digbose92/ads_data/ads_complete_repo/ads_transcripts/translated_transcripts/en_combined_transcripts.json", "r") as f:
    data = json.load(f)

prediction_dict={}
step_files=0

for key in tqdm(data):
    text=data[key]
    class_label_dict=classifier(text, label_list)

    class_labels=class_label_dict['labels']
    class_scores=class_label_dict['scores']

    class_label_pred=class_labels[0]
    class_score_pred=class_scores[0]

    #temportary dictionary
    temp_dict={'label':class_label_pred,'score':class_score_pred}

    #save the dictionary in steps of 150
    step_files+=1

    if step_files%150==0:
        filename="/bigdata/digbose92/ads_data/ads_complete_repo/ads_transcripts/translated_transcripts/predicted_labels_"+str(step_files)+".json"
        with open(filename, "w") as f:
            json.dump(prediction_dict, f,indent=4)

    prediction_dict[key]=temp_dict

    #print(key, class_label_pred, class_score_pred)

with open("/bigdata/digbose92/ads_data/ads_complete_repo/ads_transcripts/translated_transcripts/predicted_labels.json", "w") as f:
    json.dump(prediction_dict, f,indent=4)


#print(class_label)
# #text from the key
# key="bud_light_bud_lights_for_everyone"
# text=data[key]

# #preprocess the text
# #print(text)
# class_label=classifier(text, label_list)
# print(class_label)

