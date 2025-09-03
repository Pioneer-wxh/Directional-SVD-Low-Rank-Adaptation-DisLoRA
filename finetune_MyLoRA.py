import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_CACHE'] = './model'
os.environ["TOKENIZERS_PARALLELISM"] = "false"# Modify
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union

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
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from Direc_LoRA import *

def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        adapter_name: str = "lora",
        load_8bit : bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 1000,
        save_step: int = 2000, # How many steps to save checkpoint
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # MyLoRA hyperparams
        ortho_lambda: float = 1.0,
        # llm hyperparams
        train_on_inputs: bool = True,
        group_by_length: bool = False,
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",
        wandb_log_model: str = "",
        resume_from_checkpoint: str = None,
        local_rank: int = -1,
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"ortho_lambda: {ortho_lambda}\n"
    )
    assert base_model, "Please specify a --base_model"
    
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,# Modify originally bfloat16
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )

    if model.config.model_type == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = (
                [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
            )
        return tokenized_full_prompt

    #model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)

    # Load dataset to calculate warmup_steps for MyLoRA
    if data_path.endswith(".json"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules or ["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    #elif adapter_name == "bottleneck":
    #    config = BottleneckConfig(
     #       bottleneck_size=bottleneck_size,
     #       non_linearity=non_linearity,
     #       adapter_dropout=adapter_dropout,
     #       use_parallel_adapter=use_parallel_adapter,
      #      use_adapterp=use_adapterp,
      #      target_modules=target_modules,
      #      scaling=scaling,
      #      bias="none",
     #       task_type="CAUSAL_LM",
     #  )
      #  model = get_peft_model(model, config)
    elif adapter_name == "prefix-tuning":
        config = PrefixTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    elif adapter_name == "mylora":
        # Configure MyLoRA parameters
        print(lora_target_modules)
        print(len(data["train"])*num_epochs // (batch_size*gradient_accumulation_steps*3))
        config = Direc_config(
            target_modules=lora_target_modules or ["q_proj", "v_proj"],
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            warmup_steps=len(data["train"])*num_epochs // (batch_size*gradient_accumulation_steps*3),  # Set to one-third of dataset length
            ortho_lambda=ortho_lambda,
            task_type="CAUSAL_LM",
            s_tsd=8,
            prefer_small_sigma=True,
        )
        # Initialize MyLoRA model
        model = Direc_Model(
            model=model,
            config=config,
            adapter_name="default",
        )
        print(model)

    if adapter_name == "prefix-tuning":
        model.to('cuda')

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            if adapter_name != "mylora":
                model = set_peft_model_state_dict(model, adapters_weights)
            else:
                # Load MyLoRA model weights
                model.load_state_dict(adapters_weights, strict=False)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    if adapter_name == "mylora":
        # Use MyLoRA specific training parameters and trainer
        trainer_args = Direc_TrainingArguments(
            ortho_lambda=ortho_lambda,
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            fp16=True,# Modify originally bf16=True
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            logging_steps=10,
            save_total_limit=1,
            load_best_model_at_end=False,
            # report_to=None,  # Disable wandb
            # run_name=None,   # Disable wandb
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            deepspeed="ds_config.json",
        )
        trainer = Direc_Trainer(
            model=model.model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=trainer_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
    else:
        trainer_args = transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,# Modify bf16=True
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=1,# Checkpoint save strategy
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            # report_to=None,  # Disable wandb
            # run_name=None,   # Disable wandb
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            deepspeed="ds_config.json"
        )
        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=trainer_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )

    model.config.use_cache = False

    if adapter_name != "mylora":
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))
    else:
        # Keep original state_dict processing for MyLoRA
        pass

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save the final adapter state
    print("Saving final PEFT adapter weights after training completion.")
    model.save_pretrained(output_dir)

    # Save full model parameters only for MyLoRA
    if adapter_name == "mylora":
        # Only save complete model weights for MyLoRA
        model.save_module(output_dir)

        print(
            f"\nTraining finished. Full model parameters saved to: {output_dir}"
        )
    else:
        print(
            f"\nTraining finished. Only adapter weights saved to: {output_dir}"
        )

def generate_prompt(data_point): # Modify: Generate prompt template
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
if __name__ == "__main__":
    fire.Fire(train)
