import os
import argparse

import sklearn
import torch
import transformers
import datasets
import evaluate
from datasets import load_dataset
import pandas as pd
import json
from typing import List
from sentence_transformers import SentenceTransformer, util

# it removes the warning for the number of threads used for data loading
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Dataset(torch.utils.data.Dataset):
    """
    This class is inteded for sequence classification tasks.
    :param source_text: List of source text that is used as input to the model.
    :param target_text: List of target text that is used as expected output from the model.
    :param tokenizer: The tokenizer to be used for tokenizing the texts. It can be an instance of the transformers AutoTokenizer class or a custom tokenizer.
    :param max_input_length: The maximum length of the tokenized input text.
    :param max_output_length: The maximum length of the tokenized output text.
    :param padding: The padding strategy to be used. Available options are available in the transformers library.
    :param truncation: Whether to truncate the text or not.
    """

    def __init__(
        self,
        source_text: List[str],
        target_text: List[str],
        tokenizer,
        max_input_length: int = 1024,
        max_output_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
    ):

        self.source_text = source_text
        self.target_text = target_text
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation

    def __getitem__(self, idx):
        """
        This function is called to get the tokenized source and target text for a given index.
        :param idx: The index of the text and label to be returned.
        :return: A dictionary containing the tokenized source (`input_ids`) with attention mask (`attention_mask`) and the tokenized target (`labels`).
        """
        input = self.tokenizer(
            self.source_text[idx],
            max_length=self.max_input_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        output = self.tokenizer(
            text_target = self.target_text[idx],
            max_length=self.max_output_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt",
        )

        item = {
            "input_ids": input["input_ids"].squeeze(),
            "attention_mask": input["attention_mask"].squeeze(),
            "labels": output["input_ids"].squeeze(),
        }

        return item

    def __len__(self):
        return len(self.source_text)
    

parser = argparse.ArgumentParser()
parser.add_argument(
    "--MODEL_TAG",
    type=str,
    default="facebook/bart-large-cnn",
    help="The identifier for the model to be used. It can be an identifier from the transformers library or a path to a local model.",
)
parser.add_argument(
    "--BATCH_SIZE",
    type=int,
    default=4,
    help="The batch size to be used for training.",
)
parser.add_argument(
    "--EPOCHS",
    type=int,
    default=10,
    help="The number of epochs to be used for training.",
)
parser.add_argument(
    "--NUM_SENTENCES",
    type=int,
    default=1,
    help="The number of sentences to be used for training.",
)
parser.add_argument(
    "--METRIC_FOR_BEST_MODEL",
    type=str,
    default="SBERT",
    help="The metric to be used for saving the best model.",
)
parser.add_argument(
    "--CHECKPOINT_DIR",
    type=str,
    default="checkpoints",
    help="The directory where the checkpoints will be saved.",
)
# parser.add_argument(
#     "--TOKENIZER_DIR",
#     type=str,
#     default="tokenizer",
#     help="The directory where the tokenizer will be saved.",
# )
parser.add_argument(
    "--LOG_DIR",
    type=str,
    default="logs",
    help="The directory where the logs will be saved.",
)
parser.add_argument(
    "--LOGGING_STEPS",
    type=int,
    default=100,
    help="The number of steps after which the logs will be saved.",
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
parser.add_argument(
    "--LEARNING_RATE",
    type=float,
    default=1e-5,
    help="The learning rate to be used for training.",
)
parser.add_argument(
    "--DATALOADER_NUM_WORKERS",
    type=int,
    default=4,
    help="The number of workers to be used for the dataloaders.",
)
parser.add_argument(
    "--SAVE_TOTAL_LIMIT",
    type=int,
    default=2,
    help="The maximum number of checkpoints that will be saved. The best checkpoint will always be saved.",
)
parser.add_argument(
    "--FP16",
    default=False,
    action="store_true",
    help="Whether to use 16-bit (mixed) precision instead of 32-bit.",
)
parser.add_argument(
    "--USE_CUDA", default=False, action="store_true", help="Enable cuda computation"
)
parser.add_argument(
    "--PUSH_TO_HUB",
    default=False,
    action="store_true",
    help="Whether to push the model to the Hugging Face Hub after training.",
)
parser.add_argument(
    "--HUB_MODEL_ID",
    type=str,
    default=None,
    help="The ID of the model to be pushed to the Hugging Face Hub.",
)
args = parser.parse_args()

checkpoint_dir = os.path.join(args.CHECKPOINT_DIR, args.MODEL_TAG.split("/")[-1]+"_"+str(args.MAX_OUTPUT_LENGTH)+"_"+str(args.NUM_SENTENCES)+"_"+args.METRIC_FOR_BEST_MODEL)

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

# metadata = pd.read_csv("../podcast_dataset/podcasts-no-audio-13GB/metadata.tsv", sep="\t")
# gt_dict = {}
# for a,b,c in zip(list(metadata["show_filename_prefix"]), list(metadata["episode_filename_prefix"]), list(metadata["episode_description"])):
#     key = a.split("_")[1] + "_" + b
#     gt_dict[key] = c

gt_dict = {}
for file in os.listdir("splits"):
    with open(os.path.join("splits", file) , "r") as test_ids:
        lines = test_ids.readlines()
        for line in lines:
            df = pd.read_feather(line[:-1])
            show_id = line.split("_")[-1][:-9]

            groups = df.groupby("podcast_id")
            groups = [groups.get_group(x) for x in groups.groups]
            for g in groups:
                podcast_id = show_id + "_" + list(g["podcast_id"])[0]
                gt_dict[podcast_id] = list(g["description"])[0]


with open("output/Hierarchical_MATeR_train.json", "r") as input_file:
    train_data = json.load(input_file)

with open("output/Hierarchical_MATeR_val.json", "r") as input_file:
    val_data = json.load(input_file)

with open("output/Hierarchical_MATeR_test.json", "r") as input_file:
    test_data = json.load(input_file)


train_sources = []
train_targets = []
for key in train_data.keys():
    train_sources.append(" ".join(train_data[key][str(args.NUM_SENTENCES)]))
    train_targets.append(gt_dict[key])

val_sources = []
val_targets = []
for key in val_data.keys():
    val_sources.append(" ".join(val_data[key][str(args.NUM_SENTENCES)]))
    val_targets.append(gt_dict[key])

test_sources = []
test_targets = []
for key in test_data.keys():
    test_sources.append(" ".join(test_data[key][str(args.NUM_SENTENCES)]))
    test_targets.append(gt_dict[key])


model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.MODEL_TAG)
tokenizer = transformers.AutoTokenizer.from_pretrained(args.MODEL_TAG)
#tokenizer.save_pretrained(args.TOKENIZER_DIR)

"""
############################################################################################################
Instantiating the dataset objects for each split.
!!! The dataloaders are created inside the Trainer object !!!
############################################################################################################
"""

summarization_train_dataset = Dataset(
    source_text=train_sources,
    target_text=train_targets,
    tokenizer=tokenizer,
    max_input_length=args.MAX_INPUT_LENGTH,
    max_output_length=args.MAX_OUTPUT_LENGTH,
    padding="max_length",
    truncation=True,
)

summarization_val_dataset = Dataset(
    source_text=val_sources,
    target_text=val_targets,
    tokenizer=tokenizer,
    max_input_length=args.MAX_INPUT_LENGTH,
    max_output_length=args.MAX_OUTPUT_LENGTH,
    padding="max_length",
    truncation=True,
)

summarization_test_dataset = Dataset(
    source_text=test_sources,
    target_text=test_targets,
    tokenizer=tokenizer,
    max_input_length=args.MAX_INPUT_LENGTH,
    max_output_length=args.MAX_OUTPUT_LENGTH,
    padding="max_length",
    truncation=True,
)

"""
############################################################################################################
Creating the training arguments that will be passed to the Trainer object.
Most of the parameters are taken from the parser arguments.
############################################################################################################
"""
training_arguments = transformers.TrainingArguments(
    output_dir=checkpoint_dir,
    num_train_epochs=args.EPOCHS,
    per_device_train_batch_size=args.BATCH_SIZE,
    per_device_eval_batch_size=args.BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=args.LOG_DIR,
    logging_steps=args.LOGGING_STEPS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=args.LEARNING_RATE,
    dataloader_num_workers=args.DATALOADER_NUM_WORKERS,
    save_total_limit=args.SAVE_TOTAL_LIMIT,
    no_cuda=not (args.USE_CUDA),
    fp16=args.FP16,
    metric_for_best_model=args.METRIC_FOR_BEST_MODEL,
    greater_is_better=True,
    hub_model_id=args.HUB_MODEL_ID,
    push_to_hub=args.PUSH_TO_HUB,
)

"""
############################################################################################################
Defining the compute_metrics function that will be used to compute the metrics for the validation and testing sets.
The function takes as input a dictionary with the predictions and the labels and returns a dictionary with the metrics.
############################################################################################################
"""


rouge = evaluate.load("rouge")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
textModel = SentenceTransformer('paraphrase-mpnet-base-v2', device=device)
textModel.eval()


def compute_metrics(pred):

    labels_ids = pred.label_ids
    pred_ids = pred.predictions[0]

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str,
        references=label_str,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
    )


    with torch.no_grad():
        sentence_embedding = textModel.encode(pred_str, convert_to_tensor=True)
        sentence_embedding = sentence_embedding.detach().cpu()
        description_embedding = textModel.encode(label_str, convert_to_tensor=True)
        description_embedding = description_embedding.detach().cpu()
        cosine_scores = util.pytorch_cos_sim(sentence_embedding, description_embedding)
        sbert_score = cosine_scores[0][0]


    return {
        "R1": round(rouge_output["rouge1"], 4),
        "R2": round(rouge_output["rouge2"], 4),
        "RL": round(rouge_output["rougeL"], 4),
        "RLsum": round(rouge_output["rougeLsum"], 4),
        "SBERT": round(sbert_score.item(), 4),
    }

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


"""
############################################################################################################
Instantiating the Trainer object.
It will take care of training and validation.
############################################################################################################
"""

trainer = transformers.Trainer(
    model=model,
    args=training_arguments,
    train_dataset=summarization_train_dataset,
    eval_dataset=summarization_val_dataset,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()

"""
############################################################################################################
Evaluate the model on the test set.
############################################################################################################
"""

print (trainer.evaluate(summarization_test_dataset))

"""
############################################################################################################
If the PUSH_TO_HUB argument is True, the model is pushed to the Hugging Face Hub.
The model is pushed to the user's namespace using the HUB_MODEL_NAME argument.
############################################################################################################
"""

if args.PUSH_TO_HUB:
    trainer.push_to_hub()
    tokenizer.push_to_hub(repo_id=args.HUB_MODEL_ID)

