#convert the json of 25 values to a csv with 5 x 5 and columns with the av seed values and index as the tv seed values

import pandas as pd
import json as json

#read the json file
json_file="/proj/digbose92/ads_repo/model_files/predictions/Topic_double_max.json"
with open(json_file,'r') as f:
    json_data=json.load(f)

#convert json to a dataframe

df=pd.DataFrame(json_data)
df=df.T
df.to_csv("/proj/digbose92/ads_repo/model_files/predictions/Topic_double_max.csv")