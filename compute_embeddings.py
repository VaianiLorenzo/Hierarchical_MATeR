import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.multiprocessing.set_sharing_strategy('file_system')


def collate_fn(batch):
        audio = [item[0] for item in batch]
        text = [item[1] for item in batch]
        target = [item[2] for item in batch]
        path = [item[3] for item in batch]
        start = [item[4] for item in batch]
        end = [item[5] for item in batch]
        sentence = [item[6] for item in batch]
        return [audio, text, target, path, start, end, sentence]


def compute_embeddings(split, model):
    dataloader = torch.load(os.path.join("dataloaders", split + "_dataloader_sbert_score.bkp"))
    print("Dataloader loaded...")

    df = pd.DataFrame()
    df_values = []
    embeddings = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            audios, texts, targets, path, start, end, sentences = data
            podcast_ids = [x.split("/")[-2][5:] + "_" + x.split("/")[-1][:-4] for x in path]
            audio_values = torch.stack(audios).to(device)
            _, embedding = model(audio_values, texts)
            for a,b,c,d,e,f,g in zip(podcast_ids, sentences, targets, embedding.tolist(), start, end, path):
                df_values.append([a,b,c.item(),e,f,g])
                embeddings.append(np.array(d))


    df["podcast_id"] = [value[0] for value in df_values]
    df["sentence"] = [value[1] for value in df_values]
    df["score"] = [value[2] for value in df_values]
    df["start"] = [value[3] for value in df_values]
    df["end"] = [value[4] for value in df_values]
    df["path"] = [value[5] for value in df_values]
    df["index"] = range(len(df))
            
    df.to_csv(os.path.join("data", split + "_sentences.csv"), index=False)
    embeddings = np.array(embeddings)
    np.save(os.path.join("data", split + "_embeddings.npy"), embeddings)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_dir = "checkpoints"
model_name = "MATeR_BERT_wav2vec2_1e-5_2.model"
mater_model = torch.load(model_dir + "/" + model_name)
mater_model = mater_model.to(device)
mater_model.eval()

compute_embeddings("val", mater_model)
compute_embeddings("test", mater_model)
#compute_embeddings("train", mater_model)

####### ADJUST



def adjust(split):
    old_df = pd.read_csv('data/' + split + '_sentences.csv')
    dataloader = torch.load(os.path.join("dataloaders", split + "_dataloader_sbert_score.bkp"))

    starts = [0]*len(old_df)
    ends = [0]*len(old_df)

    with torch.no_grad():
        for data in tqdm(dataloader):
            audios, texts, targets, path, start, end, sentences = data
            podcast_ids = [x.split("/")[-2][5:] + "_" + x.split("/")[-1][:-4] for x in path]

            #find in "old_df" the index of row with the same podcast_id and sentence
            for a,b,c,d in zip(podcast_ids, sentences, start, end):
                index = old_df[(old_df["podcast_id"] == a) & (old_df["sentence"] == b)].index[0]
                starts[index] = c
                ends[index] = d

    old_df["start"] = starts
    old_df["end"] = ends
    old_df.to_csv(os.path.join("data", "adjusted_" + split + "_sentences.csv"), index=False)
            
# adjust("val")
# adjust("test")
# adjust("train")
