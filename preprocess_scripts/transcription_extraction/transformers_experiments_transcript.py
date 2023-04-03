import os 
import pandas as pd 
import numpy as np
import pickle 
import transformers 
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

#check the model with t5 large 
model = T5ForConditionalGeneration.from_pretrained('t5-large')
tokenizer = T5Tokenizer.from_pretrained('t5-large')

#load the transcripts file 
with open("/bigdata/digbose92/ads_data/ads_complete_repo/ads_transcripts/translated_transcripts/en_combined_transcripts.json", "r") as f:
    data = json.load(f)

key="kitchen_world_the_letter"
text=data[key]

preprocess_text = text.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text
print ("original text preprocessed: \n", preprocess_text)


tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=30,
                                    max_length=100,
                                    early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)

