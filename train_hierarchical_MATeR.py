
import os
import numpy as np
import pandas as pd
import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from tqdm import tqdm
import json

from argparse import ArgumentParser
from Hierarchical_MATeR_Dataset import Hierarchical_MATeR_Dataset
from Hierarchical_MATeR import Hierarchical_MATeR
from torch.utils.data import DataLoader
from torch.nn import MSELoss

parser = ArgumentParser(description='Training hierarchical MATeR model')
parser.add_argument('--epochs',
        help='Number of epochs to train the model',
        default=100,
        required=False,
        type=int)
parser.add_argument('--batch_size',
        help='Batch size for training',
        default=32,
        required=False,
        type=int)
parser.add_argument('--lr',
        help='Learning rate for training',
        default=1e-5,
        required=False,
        type=float)
parser.add_argument('--attention_layers',
        help='Number of attention layers in the hierarchical transformer',
        default=3,
        required=False,
        type=int)
args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
epochs = args.epochs
attention_layers = args.attention_layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Hierarchical_MATeR(d_model=1536, nhead=4, d_hid=1536, nlayers=attention_layers, dropout=0.25, device=device)
model = model.to(device)
print("Model created...")

optimizer = AdamW(model.parameters(), lr=lr)
criterion = MSELoss()

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

with open(os.path.join("logs", "train_hierarchical.csv"), "w") as f:
    f.write("TRAINING")

best_val_loss = 1000
best_epoch = 0

for epoch in range(epochs):
    print("Epoch: ", epoch)
    with open(os.path.join("logs", "train_hierarchical.csv"), "a+") as f:
        f.write("Epoch: " + str(epoch+1))

    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader):
        embeddings, attention_mask, starts, ends, sentences, scores, podcast_id, path = batch

        embeddings = embeddings.to(device)
        attention_mask = attention_mask.to(device)
        scores = scores.to(device)

        outputs = model(embeddings, attention_mask)

        attention_mask = attention_mask.view(-1)
        
        scores = scores.float()
        scores = scores.view(-1,1)
        scores = scores[attention_mask == 1]

        loss = criterion(outputs, scores)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print("Train loss: ", total_loss / len(test_dataloader))
    with open(os.path.join("logs", "train_hierarchical.csv"), "a+") as f:
        f.write("Train loss: " + str(total_loss / len(test_dataloader)))


    model.eval()
    total_loss = 0
    for batch in tqdm(val_dataloader):
        embeddings, attention_mask, starts, ends, sentences, scores, podcast_id, path = batch

        embeddings = embeddings.to(device)
        attention_mask = attention_mask.to(device)
        scores = scores.to(device)
        
        outputs = model(embeddings, attention_mask)

        attention_mask = attention_mask.view(-1)
        scores = scores.float()
        scores = scores.view(-1,1)
        scores = scores[attention_mask == 1]

        loss = criterion(outputs, scores)

        total_loss += loss.item()

    print("Val loss: ", total_loss / len(val_dataloader))
    with open(os.path.join("logs", "train_hierarchical.csv"), "a+") as f:
        f.write("Val loss: " + str(total_loss / len(val_dataloader)))

    if total_loss / len(val_dataloader) < best_val_loss:
        best_val_loss = total_loss / len(val_dataloader)
        best_epoch = epoch
        torch.save(model, os.path.join("checkpoints", "hierarchical_MATeR.pt"))

print("Best model found at epoch ", best_epoch, " with val loss ", best_val_loss)



model = torch.load(os.path.join("checkpoints", "hierarchical_MATeR.pt"))
model.eval()

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

with open(os.path.join("output", "Hierarchical_MATeR_output.json"), "w") as output_file:
    json.dump(dict, output_file)
    

    


    






