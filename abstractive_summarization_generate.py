import os
import argparse
import torch
import transformers
import pandas as pd
import json
from typing import List
from tqdm import tqdm

# it removes the warning for the number of threads used for data loading
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_text: List[str],
        source_keys: List[str],
        tokenizer,
        max_input_length: int = 1024,
        max_output_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
    ):

        self.source_text = source_text
        self.source_keys = source_keys
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation

    def __getitem__(self, idx):
        input = self.tokenizer(
            self.source_text[idx],
            max_length=self.max_input_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )
        item = {
            "source_texts": self.source_text[idx],
            "source_keys": self.source_keys[idx],
            "input_ids": input["input_ids"].squeeze(),
            "attention_mask": input["attention_mask"].squeeze(),
        }
        return item

    def __len__(self):
        return len(self.source_text)
    

parser = argparse.ArgumentParser()
parser.add_argument(
    "--MODEL_PATH",
    type=str,
    default="checkpoints/bart-large-128_1/checkpoint-1000",
    help="The path of the pretrained model to be used for generate summaries.",
)
parser.add_argument(
    "--MODEL_TAG",
    type=str,
    default="facebook/bart-large",
    help="The tag of the pretrained model to be used for generate summaries.",
)
parser.add_argument(
    "--MAX_INPUT_LENGTH",
    type=int,
    default=1024,
    help="The maximum length of the input sequence.",
)
parser.add_argument(
    "--MAX_OUTPUT_LENGTH",
    type=int,
    default=128,
    help="The maximum length of the output sequence.",
)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("output/Hierarchical_MATeR_test.json", "r") as input_file:
    test_data = json.load(input_file)
test_sources = []
test_keys = []
for key in test_data.keys():
    test_sources.append(" ".join(test_data[key]["1"]))
    test_keys.append(key)

model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.MODEL_PATH, local_files_only=True)
model = model.to(device)
model.eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(args.MODEL_TAG)

test_dataset = Dataset(
    source_text=test_sources,
    source_keys=test_keys,
    tokenizer=tokenizer,
    max_input_length=args.MAX_INPUT_LENGTH,
    max_output_length=args.MAX_OUTPUT_LENGTH,
    padding="max_length",
    truncation=True,
)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)


output_dict = {}
with torch.no_grad():
    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for j, (source_text, decoded_output) in enumerate(zip(data["source_texts"], decoded_outputs)):
            # print("Element: ", i * 16 + j)
            # print("\tSource text: ", source_text)
            # print("\tDecoded output: ", decoded_output)
            inner_dict = {}
            inner_dict["1"] = [decoded_output]
            output_dict[data["source_keys"][j]] = inner_dict

with open("output/abstractive-"+str(args.MODEL_PATH.split("/")[1])+"_Hierarchical_MATeR_test.json", "w") as output_file:
    json.dump(output_dict, output_file)

        
