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
    default=32,
    help="The batch size to be used for training.",
)
parser.add_argument(
    "--EPOCHS",
    type=int,
    default=10,
    help="The number of epochs to be used for training.",
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
    default=5e-5,
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


metadata = pd.read_csv("../podcast_dataset/podcasts-no-audio-13GB/metadata.tsv", sep="\t")
gt_dict = {}
for a,b,c in zip(list(metadata["show_filename_prefix"]), list(metadata["episode_filename_prefix"]), list(metadata["episode_description"])):
    key = a.split("_")[1] + "_" + b
    gt_dict[key] = c

with open("output/Hierarchical_MATeR_train.json", "r") as input_file:
    train_data = json.load(input_file)

with open("output/Hierarchical_MATeR_val.json", "r") as input_file:
    val_data = json.load(input_file)

with open("output/Hierarchical_MATeR_test.json", "r") as input_file:
    test_data = json.load(input_file)


train_sources = []
train_targets = []
for key in train_data.keys():
    train_sources.append(" ".join(train_data[key]["1"]))
    train_targets.append(gt_dict[key])

val_sources = []
val_targets = []
for key in val_data.keys():
    val_sources.append(" ".join(val_data[key]["1"]))
    val_targets.append(gt_dict[key])

test_sources = []
test_targets = []
for key in test_data.keys():
    test_sources.append(" ".join(test_data[key]["1"]))
    test_targets.append(gt_dict[key])


model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.MODEL_TAG,)
tokenizer = transformers.AutoTokenizer.from_pretrained(args.MODEL_TAG,)
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
    output_dir=args.CHECKPOINT_DIR,
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
    metric_for_best_model="R2",
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

    return {
        "R1": round(rouge_output["rouge1"], 4),
        "R2": round(rouge_output["rouge2"], 4),
        "RL": round(rouge_output["rougeL"], 4),
        "RLsum": round(rouge_output["rougeLsum"], 4),
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


# import pandas as pd
# import json
# from transformers import AutoTokenizer
# from transformers import AutoModelForSeq2SeqLM
# from datasets import load_metric
# from datasets import Dataset
# import os

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# max_input_length = 512
# max_output_length = 128
# min_output_length = 64
# batch_size = 2


# metadata = pd.read_csv("../podcast_dataset/podcasts-no-audio-13GB/metadata.tsv", sep="\t")
# gt_dict = {}
# for a,b,c in zip(list(metadata["show_filename_prefix"]), list(metadata["episode_filename_prefix"]), list(metadata["episode_description"])):
#     key = a.split("_")[1] + "_" + b
#     gt_dict[key] = c

# with open("output/Hierarchical_MATeR_train.json", "r") as input_file:
#     train_data = json.load(input_file)

# with open("output/Hierarchical_MATeR_val.json", "r") as input_file:
#     val_data = json.load(input_file)

# with open("output/Hierarchical_MATeR_test.json", "r") as input_file:
#     test_data = json.load(input_file)

# train_dataset = pd.DataFrame()
# train_texts = []
# train_descriptions = []
# for k in train_data.keys():
#     train_texts.append(" ".join(train_data[k]["1"]))
#     train_descriptions.append(gt_dict[k])

# train_dataset["text"] = train_texts
# train_dataset["description"] = train_descriptions

# val_dataset = pd.DataFrame()
# val_texts = []
# val_descriptions = []
# for k in val_data.keys():
#     val_texts.append(" ".join(val_data[k]["1"]))
#     val_descriptions.append(gt_dict[k])

# val_dataset["text"] = val_texts
# val_dataset["description"] = val_descriptions

# test_dataset = pd.DataFrame()
# test_texts = []
# test_descriptions = []
# for k in test_data.keys():
#     test_texts.append(" ".join(test_data[k]["1"]))
#     test_descriptions.append(gt_dict[k])

# test_dataset["text"] = test_texts
# test_dataset["description"] = test_descriptions

# train_dataset = Dataset.from_pandas(train_dataset)
# val_dataset = Dataset.from_pandas(val_dataset)
# test_dataset = Dataset.from_pandas(test_dataset)

# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# def process_data_to_model_inputs(batch):
#     # tokenize the inputs and labels

#     inputs = tokenizer(
#         batch["text"],
#         padding="max_length",
#         truncation=True,
#         max_length=max_input_length,
#     )
#     outputs = tokenizer(
#         batch["description"],
#         padding="max_length",
#         truncation=True,
#         max_length=max_output_length,
#     )

#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask

#     # create 0 global_attention_mask lists
#     batch["global_attention_mask"] = len(batch["input_ids"]) * [
#         [0 for _ in range(len(batch["input_ids"][0]))]
#     ]

#     # since above lists are references, the following line changes the 0 index for all samples
#     batch["global_attention_mask"][0][0] = 1
#     batch["labels"] = outputs.input_ids

#     # We have to make sure that the PAD token is ignored
#     batch["labels"] = [
#         [-100 if token == tokenizer.pad_token_id else token for token in labels]
#         for labels in batch["labels"]
#     ]

#     return batch


# train_dataset = train_dataset.map(
#     process_data_to_model_inputs,
#     batched=True,
#     batch_size=batch_size,
#     remove_columns=["text", "description"],
# )

# val_dataset = val_dataset.map(
#     process_data_to_model_inputs,
#     batched=True,
#     batch_size=batch_size,
#     remove_columns=["text", "description"],
# )

# train_dataset.set_format(
#     type="torch",
#     columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
# )
# val_dataset.set_format(
#     type="torch",
#     columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
# )



# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn", gradient_checkpointing=True, use_cache=False)

# # set generate hyperparameters
# model.config.num_beams = 2
# model.config.max_length = max_output_length
# model.config.min_length = min_output_length
# model.config.length_penalty = 2.0
# model.config.early_stopping = True
# model.config.no_repeat_ngram_size = 3

# rouge = load_metric("rouge")

# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = tokenizer.pad_token_id
#     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

#     rouge_output = rouge.compute(
#         predictions=pred_str, references=label_str, rouge_types=["rouge2"]
#     )["rouge2"].mid

#     return {
#         "rouge2_precision": round(rouge_output.precision, 4),
#         "rouge2_recall": round(rouge_output.recall, 4),
#         "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
#     }

# from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
# import os
# output_checkpoints_dir = "checkpoints"
# if not os.path.exists(output_checkpoints_dir):
#     os.makedirs(output_checkpoints_dir)


# # enable fp16 apex training
# training_args = Seq2SeqTrainingArguments(
#     predict_with_generate=True,
#     evaluation_strategy="steps",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     fp16=True,
#     output_dir=output_checkpoints_dir,
#     logging_steps= 2027,  # Total optimization steps / (20*Num Epochs)
#     eval_steps=6081, # Total optimization steps / Num Epochs
#     save_steps=6081, # Total optimization steps / Num Epochs
#     save_total_limit=8,
#     gradient_accumulation_steps=4,
#     num_train_epochs=3,
# )

# trainer = Seq2SeqTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# trainer.train()

# #led.save_model("./LED_multinews/")





















# # TODO: update train_dataset and val_dataset with DF
# target_score = "sbert_score"
# train_dataset = pd.read_feather("train_df_"+target_score+".feather")
# val_dataset = pd.read_feather("val_df_"+target_score+".feather")
# sorting_key = "score"


# top_n = 50
# max_input_length = 8192
# max_output_length = 128
# min_output_length = 64
# batch_size = 4

# # 1. Top-N frasi
# list_df = train_dataset.groupby("podcastID")
# d_list = []
# t_list = []
# for index in list_df.groups:
#     l = list_df.get_group(index)
#     l = l.sort_values(by=[sorting_key], ascending=False)
#     l = l[:top_n]
#     # 2. Re-ranking 
#     l = l.sort_values(by=["startTime"], ascending=True)
#     desc = list(l["description"])[0]
#     text = " ".join(list(l["text"]))
#     d_list.append(desc)
#     t_list.append(text)

# train_dataset = pd.DataFrame()
# train_dataset["text"] = t_list
# train_dataset["description"] = d_list

# list_df = val_dataset.groupby("podcastID")
# d_list = []
# t_list = []
# for index in list_df.groups:
#     l = list_df.get_group(index)
#     l = l.sort_values(by=[sorting_key], ascending=False)
#     l = l[:top_n]
#     # 2. Re-ranking 
#     l = l.sort_values(by=["startTime"], ascending=True)
#     desc = list(l["description"])[0]
#     text = " ".join(list(l["text"]))
#     d_list.append(desc)
#     t_list.append(text)

# val_dataset = pd.DataFrame()
# val_dataset["text"] = t_list
# val_dataset["description"] = d_list

# train_dataset = Dataset.from_pandas(train_dataset)
# val_dataset = Dataset.from_pandas(val_dataset)

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")



# def process_data_to_model_inputs(batch):
#     # tokenize the inputs and labels

#     inputs = tokenizer(
#         batch["text"],
#         padding="max_length",
#         truncation=True,
#         max_length=max_input_length,
#     )
#     outputs = tokenizer(
#         batch["description"],
#         padding="max_length",
#         truncation=True,
#         max_length=max_output_length,
#     )

#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask

#     # create 0 global_attention_mask lists
#     batch["global_attention_mask"] = len(batch["input_ids"]) * [
#         [0 for _ in range(len(batch["input_ids"][0]))]
#     ]

#     # since above lists are references, the following line changes the 0 index for all samples
#     batch["global_attention_mask"][0][0] = 1
#     batch["labels"] = outputs.input_ids

#     # We have to make sure that the PAD token is ignored
#     batch["labels"] = [
#         [-100 if token == tokenizer.pad_token_id else token for token in labels]
#         for labels in batch["labels"]
#     ]

#     return batch

# train_dataset = train_dataset.map(
#     process_data_to_model_inputs,
#     batched=True,
#     batch_size=batch_size,
#     remove_columns=["text", "description"],
# )

# val_dataset = val_dataset.map(
#     process_data_to_model_inputs,
#     batched=True,
#     batch_size=batch_size,
#     remove_columns=["text", "description"],
# )

# train_dataset.set_format(
#     type="torch",
#     columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
# )
# val_dataset.set_format(
#     type="torch",
#     columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
# )


# from transformers import AutoModelForSeq2SeqLM

# led = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384", gradient_checkpointing=True, use_cache=False)

# # set generate hyperparameters
# led.config.num_beams = 2
# led.config.max_length = max_output_length
# led.config.min_length = min_output_length
# led.config.length_penalty = 2.0
# led.config.early_stopping = True
# led.config.no_repeat_ngram_size = 3

# rouge = load_metric("rouge")

# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = tokenizer.pad_token_id
#     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

#     rouge_output = rouge.compute(
#         predictions=pred_str, references=label_str, rouge_types=["rouge2"]
#     )["rouge2"].mid

#     return {
#         "rouge2_precision": round(rouge_output.precision, 4),
#         "rouge2_recall": round(rouge_output.recall, 4),
#         "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
#     }

# from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
# import os
# led_checkpoints_dir = "./checkpoints_LED_min64_"+str(top_n)+"/"
# if not os.path.exists(led_checkpoints_dir):
#     os.makedirs(led_checkpoints_dir)


# # enable fp16 apex training
# training_args = Seq2SeqTrainingArguments(
#     predict_with_generate=True,
#     evaluation_strategy="steps",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     fp16=True,
#     output_dir=led_checkpoints_dir,
#     logging_steps= 2027,  # Total optimization steps / (20*Num Epochs)
#     eval_steps=6081, # Total optimization steps / Num Epochs
#     save_steps=6081, # Total optimization steps / Num Epochs
#     save_total_limit=8,
#     gradient_accumulation_steps=4,
#     num_train_epochs=3,
# )

# trainer = Seq2SeqTrainer(
#     model=led,
#     tokenizer=tokenizer,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )


# trainer.train()

# #led.save_model("./LED_multinews/")
