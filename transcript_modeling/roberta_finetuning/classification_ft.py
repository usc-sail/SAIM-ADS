import torch
import evaluate
from typing import Dict
from transformers import BertTokenizer, BertModel, AutoTokenizer, XLMRobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

# target = "Topic"
target = "Transition_val"
# target = "social_message"

data_path = "/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_transcripts_augmented.csv"
if target == "Topic":
    label_map = {"Games": 0,
        "Household": 1,
        "Services": 2,
        "Misc": 3,
        "Sports": 4,
        "Banking": 5,
        "Clothing": 6,
        "Industrial and agriculture": 7,
        "Leisure": 8,
        "Publications media": 9,
        "Health": 10,
        "Car": 11,
        "Electronics": 12,
        "Cosmetics": 13,
        "Food and drink": 14,
        "Awareness": 15,
        "Travel and transport": 16,
        "Retail": 17
    }
    num_labels = 18
elif target == "social_message":
    label_map = {
        "No": 0,
        "Yes": 1
    }
    num_labels = 2
    
elif target == "Transition_val":
    label_map = {
        "No transition": 0,
        "Transition": 1
    }
    num_labels = 2

metric_f1 = evaluate.load("f1")
metric_acc = evaluate.load("accuracy")
class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val for key, val in self.encodings[idx].items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def data_preparation():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    train_data, val_data, test_data = [], [], []
    train_label, val_label, test_label = [], [], []
    df = pd.read_csv(data_path)
    for index, row in df.iterrows():
        transcript_path = row['Transcript']
        with open(transcript_path, 'r') as file:
            text = file.read()
        if text == "":
            continue
        topic = row[target]
        tokenized_text = tokenizer(text, padding='max_length', return_tensors="pt", truncation = True)
        tokenized_text['input_ids'] = tokenized_text['input_ids'].squeeze()
        tokenized_text['attention_mask'] = tokenized_text['attention_mask'].squeeze()
        if row['Split'] == 'train':
            train_data.append(tokenized_text)
            train_label.append(label_map[topic])
        elif row['Split'] == 'val':
            val_data.append(tokenized_text)
            val_label.append(label_map[topic])
        else:
            test_data.append(tokenized_text)
            test_label.append(label_map[topic])
        # if index == 100:
        #     break
    return MyDataset(train_data, train_label), MyDataset(val_data, val_label), MyDataset(test_data, test_label)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = metric_f1.compute(predictions=predictions, references=labels, average = "macro")
    metrics.update(metric_acc.compute(predictions=predictions, references=labels))
    return metrics

if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = data_preparation()
    training_args = TrainingArguments(
        output_dir=f'./results_{target}',          # output directory
        num_train_epochs=10,              # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=4,   # batch size for evaluation
        warmup_steps=0,                # number of warmup steps for learning rate scheduler
        weight_decay=0.001,               # strength of weight decay
        learning_rate=1e-5,
        logging_dir=f'./logs_{target}',            # directory for storing logs
        logging_steps=20,
        evaluation_strategy="epoch"
    )
    model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels).to(torch.device('cuda:0'))
    
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics = compute_metrics
    )
    trainer.train()
    test_result = trainer.predict(test_dataset)
    print(test_result.metrics)
    trainer.save_model(f"./best_model_{target}")