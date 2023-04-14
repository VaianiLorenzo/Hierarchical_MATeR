
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json
from argparse import ArgumentParser
from Hierarchical_MATeR_Dataset import Hierarchical_MATeR_Dataset
from torch.utils.data import DataLoader

parser = ArgumentParser(description='Training hierarchical MATeR model')
parser.add_argument('--batch_size',
        help='Batch size for training',
        default=32,
        required=False,
        type=int)
args = parser.parse_args()

batch_size = args.batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_embeddings = np.load(os.path.join("data", "train_embeddings.npy"), allow_pickle=True)
train_df = pd.read_csv(os.path.join("data", "train_sentences.csv"))
val_embeddings = np.load(os.path.join("data", "val_embeddings.npy"), allow_pickle=True)
val_df = pd.read_csv(os.path.join("data", "val_sentences.csv"))
test_embeddings = np.load(os.path.join("data", "test_embeddings.npy"), allow_pickle=True)
test_df = pd.read_csv(os.path.join("data", "test_sentences.csv"))
print("Data loaded...")

train_dataset = Hierarchical_MATeR_Dataset(train_embeddings, train_df)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
val_dataset = Hierarchical_MATeR_Dataset(val_embeddings, val_df)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
test_dataset = Hierarchical_MATeR_Dataset(test_embeddings, test_df)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
print("Dataloaders created...")

model = torch.load(os.path.join("checkpoints", "hierarchical_MATeR.pt"))
model.eval()

#########
# TRIAN #
#########

df_values = []
for batch in tqdm(train_dataloader):
    embeddings, attention_mask, starts, ends, sentences, scores, podcast_id, path = batch
    
    embeddings = embeddings.to(device)
    attention_mask = attention_mask.to(device)
    scores = scores.to(device)
    
    outputs = model(embeddings, attention_mask)

    podcast_id_list = []
    for i in range(len(attention_mask)):
        for _ in range(int(torch.sum(attention_mask[i]).item())):
            podcast_id_list.append(podcast_id[i])

    attention_mask = attention_mask.view(-1)
    sentences = [item[i] for i in range(len(sentences[0])) for item in sentences]
    sentences = [sentences[i] for i in range(len(sentences)) if attention_mask[i] == 1]

    for a,b,c in zip (podcast_id_list, sentences, outputs):
        df_values.append([a,b,c.item()])

df = pd.DataFrame()
df["podcast_id"] = [value[0] for value in df_values]
df["sentence"] = [value[1] for value in df_values]
df["score"] = [value[2] for value in df_values]

dict = {}
groups = df.groupby("podcast_id")
groups = [groups.get_group(x) for x in groups.groups]
for g in tqdm(groups):
    podcast_id = list(g["podcast_id"])[0]
    sentences = g["sentence"]
    scores = g["score"]

    sentences = [x for _, x in sorted(zip(scores, sentences), reverse=True)]

    podcast_dict = {}
    for j in range(1, 6):
        if len(sentences) >= j:
            podcast_dict[str(j)] = sentences[:j]
        else:
            podcast_dict[str(j)] = sentences

    dict[podcast_id] = podcast_dict

with open(os.path.join("output", "Hierarchical_MATeR_train.json"), "w") as output_file:
    json.dump(dict, output_file)

##############
# VALIDATION #
##############

df_values = []
for batch in tqdm(val_dataloader):
    embeddings, attention_mask, starts, ends, sentences, scores, podcast_id, path = batch
    
    embeddings = embeddings.to(device)
    attention_mask = attention_mask.to(device)
    scores = scores.to(device)
    
    outputs = model(embeddings, attention_mask)

    podcast_id_list = []
    for i in range(len(attention_mask)):
        for _ in range(int(torch.sum(attention_mask[i]).item())):
            podcast_id_list.append(podcast_id[i])

    attention_mask = attention_mask.view(-1)
    sentences = [item[i] for i in range(len(sentences[0])) for item in sentences]
    sentences = [sentences[i] for i in range(len(sentences)) if attention_mask[i] == 1]

    for a,b,c in zip (podcast_id_list, sentences, outputs):
        df_values.append([a,b,c.item()])

df = pd.DataFrame()
df["podcast_id"] = [value[0] for value in df_values]
df["sentence"] = [value[1] for value in df_values]
df["score"] = [value[2] for value in df_values]

dict = {}
groups = df.groupby("podcast_id")
groups = [groups.get_group(x) for x in groups.groups]
for g in tqdm(groups):
    podcast_id = list(g["podcast_id"])[0]
    sentences = g["sentence"]
    scores = g["score"]

    sentences = [x for _, x in sorted(zip(scores, sentences), reverse=True)]

    podcast_dict = {}
    for j in range(1, 6):
        if len(sentences) >= j:
            podcast_dict[str(j)] = sentences[:j]
        else:
            podcast_dict[str(j)] = sentences

    dict[podcast_id] = podcast_dict

with open(os.path.join("output", "Hierarchical_MATeR_val.json"), "w") as output_file:
    json.dump(dict, output_file)


########
# TEST #
########

df_values = []
for batch in tqdm(test_dataloader):
    embeddings, attention_mask, starts, ends, sentences, scores, podcast_id, path = batch
    
    embeddings = embeddings.to(device)
    attention_mask = attention_mask.to(device)
    scores = scores.to(device)
    
    outputs = model(embeddings, attention_mask)

    podcast_id_list = []
    for i in range(len(attention_mask)):
        for _ in range(int(torch.sum(attention_mask[i]).item())):
            podcast_id_list.append(podcast_id[i])

    attention_mask = attention_mask.view(-1)
    sentences = [item[i] for i in range(len(sentences[0])) for item in sentences]
    sentences = [sentences[i] for i in range(len(sentences)) if attention_mask[i] == 1]

    for a,b,c in zip (podcast_id_list, sentences, outputs):
        df_values.append([a,b,c.item()])

df = pd.DataFrame()
df["podcast_id"] = [value[0] for value in df_values]
df["sentence"] = [value[1] for value in df_values]
df["score"] = [value[2] for value in df_values]

dict = {}
groups = df.groupby("podcast_id")
groups = [groups.get_group(x) for x in groups.groups]
for g in tqdm(groups):
    podcast_id = list(g["podcast_id"])[0]
    sentences = g["sentence"]
    scores = g["score"]

    sentences = [x for _, x in sorted(zip(scores, sentences), reverse=True)]

    podcast_dict = {}
    for j in range(1, 6):
        if len(sentences) >= j:
            podcast_dict[str(j)] = sentences[:j]
        else:
            podcast_dict[str(j)] = sentences

    dict[podcast_id] = podcast_dict

with open(os.path.join("output", "Hierarchical_MATeR_test.json"), "w") as output_file:
    json.dump(dict, output_file)


    

    


    






