from transformers import BertTokenizer, BertModel, AutoTokenizer, XLMRobertaModel
import pandas as pd
import pickle as pkl
from collections import defaultdict
from tqdm import tqdm
from os.path import join
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def extract_features(data_path: str, output_path: str):
    """
    Extract Bert features from the data and save them to the output path.
    """
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    # model = BertModel.from_pretrained("bert-base-multilingual-uncased")
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
    df = pd.read_csv(data_path)
    features = dict()
    for index, row in tqdm(df.iterrows()):
        transcript_path = row['Transcript']
        with open(transcript_path, 'r') as file:
            text = file.read()
        file_id = row['video_file'].split(".")[0]
        if text == "":
            continue
        encoded_input = tokenizer(text, return_tensors='pt', truncation = True)
        try:
            output = model(**encoded_input).last_hidden_state.squeeze()[0, :]
        except:
            print(text)
            return
        features[file_id] = output.detach().numpy()
    # save as pickle
    # with open(join(output_path, "bert_features.pkl"), 'wb') as file:
    with open(join(output_path, "xlm-roberta_features.pkl"), 'wb') as file:
        pkl.dump(features, file)

def classification(data_path: str, feature_path: str, label: str):
    """
    Load the features and perform classification.
    """
    df = pd.read_csv(data_path)
    with open(join(feature_path, "bert_features.pkl"), 'rb') as file:
        features = pkl.load(file)
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []
    label_map = dict()
    label_map["social_message"] = {"No": 0, "Yes": 1}
    label_map["Transition_val"] = {"No transition": 0, "Transition": 1}
    for index, row in df.iterrows():
        file_id = row['video_file'].split(".")[0]
        if file_id in features:
            if row['Split'] == 'train' or row['Split'] == 'val':
                X_train.append(features[file_id])
                y_train.append(label_map[label][row[label]])
            # elif row['Split'] == 'val':
            #     X_val.append(features[file_id])
            #     y_val.append(label_map[label][row[label]])
            else:
                X_test.append(features[file_id])
                y_test.append(label_map[label][row[label]])
    clf = LogisticRegression(random_state=42, class_weight="balanced").fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f1_score(y_test, y_pred, average='macro'))
    print(clf.score(X_test, y_test))


def multi_classification(data_path: str, feature_path: str):
    """
    Load the features and perform multiclass classification.
    """
    df = pd.read_csv(data_path)
    with open(join(feature_path + "bert_features.pkl"), 'rb') as file:
        features = pkl.load(file)
    X, y = [], []
    for index, row in df.iterrows():
        file_id = row['video_file'].split(".")[0]
        if file_id in features:
            X.append(features[file_id])
            y.append(row["Topic"])

    labels = sorted(list(set(y)))
    label_map = dict()
    for i, label in enumerate(labels):
        label_map[label] = i
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    y_train = [label_map[label] for label in y_train]
    y_test = [label_map[label] for label in y_test]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000))
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_base = [max(set(y_train), key=y_train.count) for _ in range(len(y_test))]
    print(f"f1 score")
    print(f1_score(y_test, y_pred, average='macro'))
    # print(pipe.score(X_test, y_test))
    print("baseline")
    print(f1_score(y_test, y_base, average='macro'))
    


def multi_classification_finetune(data_path: str, feature_path: str):
    """
    Load the features and perform multiclass classification.
    """
    df = pd.read_csv(data_path)
    with open(join(feature_path, "bert_features.pkl"), 'rb') as file:
        features = pkl.load(file)
    X, y = [], []
    for index, row in df.iterrows():
        file_id = row['video_file'].split(".")[0]
        if file_id in features:
            X_train.append(features[file_id])
            y_train.append(row["Topic"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    labels = sorted(list(set(y_train)))
    label_map = dict()
    for i, label in enumerate(labels):
        label_map[label] = i
    y_train = [label_map[label] for label in y_train]
    y_test = [label_map[label] for label in y_test]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000))
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_base = [max(set(y_train), key=y_train.count) for _ in range(len(y_test))]
    print(f"f1 score")
    print(f1_score(y_test, y_pred, average='macro'))
    # print(pipe.score(X_test, y_test))
    print("baseline")
    print(f1_score(y_test, y_base, average='macro'))

if __name__ == "__main__":
    data_path = "/data/digbose92/ads_complete_repo/ads_codes/SAIM-ADS/data/SAIM_data/SAIM_multi_task_tone_soc_message_topic_data_transcripts_augmented.csv"
    feature_path = "/data/anfengxu/SAIM-ADS/processed_data"

    extract_features(data_path, feature_path)
    # classification(data_path, feature_path, "social_message")
    # classification(data_path, feature_path, "Transition_val")
    # multi_classification(data_path, feature_path)
    # language_identification(data_path, feature_path)
    # language_id_histogram(data_path, feature_path)