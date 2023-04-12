import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
import pandas as pd
from MATeR_Dataset import MATeR_Dataset
from transformers import Wav2Vec2Processor
from transformers import AutoTokenizer

# a simple custom collate function
def collate_fn(batch):
        audio = [item[0] for item in batch]
        text = [item[1] for item in batch]
        target = [item[2] for item in batch]
        path = [item[3] for item in batch]
        start = [item[4] for item in batch]
        end = [item[5] for item in batch]
        sentence = [item[6] for item in batch]
        return [audio, text, target, path, start, end, sentence]

batch_size = 8
target_score = "sbert_score"

audio_model_name="facebook/wav2vec2-base"
audio_tokenizer = Wav2Vec2Processor.from_pretrained(audio_model_name)
text_model_name = "bert-base-cased"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

train_path = []
train_start_time = []
train_end_time = []
train_text = []

val_path = []
val_start_time = []
val_end_time = []
val_text = []

test_path = []
test_start_time = []
test_end_time = []
test_text = []

train_labels = []
val_labels = []
test_labels = []

with open("splits/train_shows.txt", "r")  as f_train, open("splits/val_shows.txt", "r")  as f_val, open("splits/test_shows.txt", "r")  as f_test:
    train_files = f_train.read().splitlines()
    val_files = f_val.read().splitlines()
    test_files = f_test.read().splitlines()


#################
# Train dataset #
#################    

for f in tqdm(train_files):
    df = pd.read_feather(f)
    list_path       = df["path_audio"].values
    list_start_time = df["start_time"].values
    list_end_time   = df["end_time"].values
    list_text       = df["sentence_text"].values
    list_labels     = df[target_score].values
    del df["podcast_id"]
    del df["start_time"]
    del df["end_time"]
    del df["sentence_text"]
    del df["path_audio"]
    del df["description"]
    del df["sbert_score"]
    del df
    gc.collect()

    for i, v in enumerate(list_labels):
        train_path.append(list_path[i])
        train_start_time.append(list_start_time[i])
        train_end_time.append(list_end_time[i])
        train_text.append(list_text[i])
        train_labels.append(float(v))
    del list_path
    del list_start_time
    del list_end_time
    del list_text
    del list_labels
    gc.collect()

train_dataloader = MATeR_Dataset(train_path, train_start_time, train_end_time, train_text, torch.tensor(train_labels), audio_tokenizer, text_tokenizer)
del train_path
del train_start_time
del train_end_time
del train_text
del train_labels

train_dataloader = DataLoader(train_dataloader, batch_size = batch_size, shuffle = True, 
    num_workers=32, pin_memory=False, collate_fn=collate_fn, prefetch_factor=4)
torch.save(train_dataloader, "dataloaders/train_dataloader_"+target_score+".bkp")
del train_dataloader
gc.collect()


###############
# Val dataset #
###############

print("Validation set..")
for f in tqdm(val_files):
    df = pd.read_feather(f)
    list_path       = df["path_audio"].values
    list_start_time = df["start_time"].values
    list_end_time   = df["end_time"].values
    list_text       = df["sentence_text"].values
    list_labels     = df[target_score].values
    del df["podcast_id"]
    del df["start_time"]
    del df["end_time"]
    del df["sentence_text"]
    del df["path_audio"]
    del df["description"]
    del df["sbert_score"]
    del df
    gc.collect()

    for i, v in enumerate(list_labels):
        val_path.append(list_path[i])
        val_start_time.append(list_start_time[i])
        val_end_time.append(list_end_time[i])
        val_text.append(list_text[i])
        val_labels.append(float(v))
    del list_path
    del list_start_time
    del list_end_time
    del list_text
    del list_labels
    gc.collect()

val_dataloader = MATeR_Dataset(val_path, val_start_time, val_end_time, val_text, torch.tensor(val_labels), audio_tokenizer, text_tokenizer)
del val_path
del val_start_time
del val_end_time
del val_text
del val_labels

val_dataloader = DataLoader(val_dataloader, batch_size = batch_size, shuffle = False, 
    num_workers=32, pin_memory=False, collate_fn=collate_fn, prefetch_factor=4)
torch.save(val_dataloader  , "dataloaders/val_dataloader_"+target_score+".bkp")
del val_dataloader
gc.collect()


################
# Test dataset #
################

print("Test set..")
for f in tqdm(test_files):
    df = pd.read_feather(f)
    list_path       = df["path_audio"].values
    list_start_time = df["start_time"].values
    list_end_time   = df["end_time"].values
    list_text       = df["sentence_text"].values
    list_labels     = df[target_score].values

    del df["podcast_id"]
    del df["start_time"]
    del df["end_time"]
    del df["sentence_text"]
    del df["path_audio"]
    del df["description"]
    del df["sbert_score"]
    del df
    gc.collect()

    for i, v in enumerate(list_labels):
        test_path.append(list_path[i])
        test_start_time.append(list_start_time[i])
        test_end_time.append(list_end_time[i])
        test_text.append(list_text[i])
        test_labels.append(float(v))
    del list_path
    del list_start_time
    del list_end_time
    del list_text
    del list_labels
    gc.collect()

test_dataloader = MATeR_Dataset(test_path, test_start_time, test_end_time, test_text, torch.tensor(test_labels), audio_tokenizer, text_tokenizer)
del test_path
del test_start_time
del test_end_time
del test_text
del test_labels

test_dataloader = DataLoader(test_dataloader, batch_size = batch_size, shuffle = False, 
    num_workers=32, pin_memory=False, collate_fn=collate_fn, prefetch_factor=4)
torch.save(test_dataloader  , "dataloaders/test_dataloader_"+target_score+".bkp")
