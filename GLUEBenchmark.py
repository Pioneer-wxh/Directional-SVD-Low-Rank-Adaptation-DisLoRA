from torch.nn.utils.rnn import pad_sequence

from Direc_LoRA import *
import wandb
from evaluate import load
import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import (
    LoraConfig,
    #BottleneckConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    #prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EvalPrediction
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import json

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

torch.manual_seed(42)

# %%
GLUE_TASK = ["cola", "mnli", "mrpc", "qnli",
             "qqp", "rte", "sst2", "stsb", "wnli"]
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="qnli",
                    choices=GLUE_TASK, help="choose dataset in GLUE benchmark")
parser.add_argument("--method", type=str, default="DisLoRA", choices=[
                    "DisLoRA", "LoRA", "DoRA", "PiSSA"], help="Choose which LoRA method to train") 
parser.add_argument("--lr", type=float, default=10e-4,
                    help="Choose learning rate")
parser.add_argument("--r", type=int, default=16, help="Choose Lora rank")
parser.add_argument("--lora_alpha", type=int,
                    default=24, help="Choose lora alpha")
parser.add_argument("--batch_size", type=int,
                    default=8, help="Choose batch size")
parser.add_argument("--gradient_accumulation_steps", type=int,
                    default=1, help="Choose gradient accumulation steps")
parser.add_argument("--eval_size", type=int,
                    default=128, help="Choose eval size")
parser.add_argument("--ortho_lambda", type=float, default=1,
                    help="How important the orthognality is")

args = parser.parse_args()
print(args)
# %%
if args.task in ["cola", "mrpc", "qnli", "qqp", "rte", 'sst2', 'wnli']:
    num_labels = 2
elif args.task in ["mnli"]:
    num_labels = 3
elif args.task in ["stsb"]:
    num_labels = 1
model_name = "microsoft/deberta-v3-base"  
cache_dir = "./huggingface"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels, cache_dir=cache_dir)
ds = load_dataset("nyu-mll/glue", args.task)

# %%


def process_fn(example):
    if args.task in ['cola', 'sst2']:
        tokenized_example = tokenizer(
            example['sentence'],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    elif args.task in ['mnli']:
        tokenized_example = tokenizer(
            example['premise'],
            example['hypothesis'],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    elif args.task in ['mrpc', 'rte', 'stsb', 'wnli']:
        tokenized_example = tokenizer(
            example['sentence1'],
            example['sentence2'],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    elif args.task in ['qnli']:
        tokenized_example = tokenizer(
            example['question'],
            example['sentence'],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    elif args.task in ['qqp']:
        tokenized_example = tokenizer(
            example['question1'],
            example['question2'],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    input_ids = tokenized_example['input_ids']
    attention_mask = tokenized_example['attention_mask']
    labels = example['label']
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids_padded = pad_sequence([torch.tensor(ids) for ids in input_ids],
                                    batch_first=True,
                                    padding_value=tokenizer.pad_token_id)
    attention_masks_padded = pad_sequence([torch.tensor(mask) for mask in attention_masks],
                                          batch_first=True,
                                          padding_value=0)
  
    labels_tensor = torch.tensor(labels) 

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels_tensor
    }


transformed_ds = {}
transformed_ds['train'] = ds['train'].map(
    process_fn, remove_columns=ds['train'].column_names, batched=True)
if args.task in ['mnli']:
    transformed_ds['validation'] = ds['validation_matched'].map(
        process_fn, remove_columns=ds['validation_matched'].column_names, batched=True)
    # transformed_ds['test'] = ds['test_matched'].map(process_fn, remove_columns=ds['test_matched'].column_names, batched=True)
else:
    transformed_ds['validation'] = ds['validation'].map(
        process_fn, remove_columns=ds['validation'].column_names, batched=True)
    # transformed_ds['test'] = ds['test'].map(process_fn, remove_columns=ds['test'].column_names, batched=True)
# %%
print(f"Use {args.method} to train {args.task}")
if args.method in ['LoRA', 'DoRA', 'PiSSA']:
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=["query_proj", "key_proj", "value_proj"],
        r=args.r,
        lora_dropout=0.1,  # Dropout for LoRA layers
        bias="none",
        use_dora=True if args.method == 'DoRA' else False,
        init_lora_weights="pissa" if args.method == 'PiSSA' else True,
    )
    model = get_peft_model(model, config)
    model.train()

elif args.method == 'DisLoRA':
    config = Direc_config(
        target_modules=["query_proj", "key_proj", "value_proj"],
        r=args.r,
        lora_dropout=0.1,
        lora_alpha=args.lora_alpha,
        warmup_steps=len(transformed_ds) * 4 // (args.batch_size*3),
    )
    model = Direc_Model(config, model)
    model.my_init()
    model.set_trainable(True)
    model = model.model.train()
print(model)
# %%
wandb.init(
    project="GLUE-benchmark",
    name=f"{args.task}-{args.method}",
    config={
        "task": args.task,
        "method": args.method,
        "model": model_name,
        "learning_rate": args.lr,
    }
)


def compute_metrics(eval_pred: EvalPrediction):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    if args.task == "stsb":
       
        predictions = logits.squeeze()
    else:
        
        predictions = np.argmax(logits, axis=-1)

   
    glue_metric = load('glue', args.task)
    results = glue_metric.compute(predictions=predictions, references=labels)

    return results



if args.method in ["LoRA", "DoRA", "PiSSA"]:
    training_args = TrainingArguments(
        output_dir="./cola_deberta_lora",
        num_train_epochs=4,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=args.eval_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        report_to='wandb',
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=transformed_ds["train"],
        eval_dataset=transformed_ds["validation"],
        data_collator=collate_fn,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  
    )

elif args.method == "DisLoRA":
    training_args = Direc_TrainingArguments(
        ortho_lambda=args.ortho_lambda,
        output_dir="./cola_deberta_lora",
        num_train_epochs=4,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.eval_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        report_to='wandb',
    )
    trainer = Direc_Trainer(
        model,
        training_args,
        train_dataset=transformed_ds["train"],
        eval_dataset=transformed_ds["validation"],
        # tokenizer=tokenizer,
        compute_metrics=compute_metrics,  
    )

trainer.train()


eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)


save_dir = "./training_results"
os.makedirs(save_dir, exist_ok=True)
results = {
    "task": args.task,
    "model": model_name,
    "lr": args.lr,
    "r": args.r,
    "lora_alpha": args.lora_alpha,
    "batch_size": args.batch_size,
    "method": args.method,
    "metrics": eval_results,
    "ortho_lambda": training_args.ortho_lambda if args.method == "DisLoRA" else None,
}


save_path = os.path.join(
    save_dir,
    f"{model_name.replace('/', '_')}_task-{args.task}_method-{args.method}_lr-{args.lr}_r-{args.r}_alpha-{args.lora_alpha}_bs-{args.batch_size}_results.json"
)
with open(save_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"\nThe results have been saved to: {save_path}")

#use trainer to save model
# weights_dir = f"./weights/{model_name}_{args.task}_{args.method}"
# trainer.save_model(weights_dir)
# tokenizer.save_pretrained(weights_dir)

# print(f"The model weights and tokenizer have been saved to: {weights_dir}")
wandb.log({"final_results": eval_results})
wandb.finish()
