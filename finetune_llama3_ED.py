import copy
import os
os.environ["WANDB_MODE"] = "disabled"
import pdb
import sys
from typing import List

import fire
import torch
local_transformers_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/internship/wlr_test/pkgs/transformers/src"
sys.path.insert(0, local_transformers_path)
import transformers
print(transformers.__file__)
# import transformers
from datasets import load_dataset
from transformers import set_seed

set_seed(42)
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.backends.cuda.matmul.allow_tf32 = True

print_prompt_flag = 0


def generate_prompt(data_point, test=False, comet=False):
    # knowledge
    cause = data_point['cause']
    emo = data_point['emo']
    subs = data_point['subs']
    intent = data_point['intent']
    if "Answer:" == cause[:len("Answer:")]:
        cause = cause[len("Answer:"):].strip()
    elif "Inference:" == cause[:len("Inference:")]:
        cause = cause[len("Inference:"):].strip()
    if "The cause of the listener's last utterance" in cause:
        cause = cause.replace("The cause of the listener's last utterance", "The cause of the listener's next utterance")
    if "Answer:" == emo[:len("Answer:")]:
        emo = emo[len("Answer:"):].strip()
    elif "Inference:" == emo[:len("Inference:")]:
        emo = emo[len("Inference:"):].strip()

    if subs[:len("Answer:")] == "Answer:":
        subs = subs[len("Answer:"):].strip()
    elif "Inference:" == subs[:len("Inference:")]:
        subs = subs[len("Inference:"):].strip()
    # if 'supporter' in subs:
    #     subs = subs.replace('supporter','listener')

    if "Answer:" == intent[:len("Answer:")]:
        intent = intent[len("Answer:"):].strip()
    elif "Inference:" == intent[:len("Inference:")]:
        intent = intent[len("Inference:"):].strip()

    knowledge = \
'''
The underlying cause of the listener's next utterance (the reason contributing to response) is: {}

The subsequent event about the listener that happens or could happen following the last the utterance stated by the listener: {}

The possible emotional reaction of the speaker in response to the last utterance stated by the speaker is: {}

The listener's intent to post the last utterance according to the emotion reaction of the speaker is: {}
'''.format(cause, subs, emo, intent)

    messages = []

    system_prompt = "Assuming that you are a highly empathetic person, there is a dyadic dialogue clip between a listener and a speaker. You should first identify emotion of the speaker in the dyadic dialogue clip, and then generate a concise, relevant and empathetic response for the following conversation.\n" \
                    "Please generate a response that incorporates relevant common-sense knowledge: " + knowledge

    messages.append({"role": "system", "content": system_prompt})
    context = data_point['context']
    context.append(data_point['target'])
    for i, dia in enumerate(context):
        if i == len(context) - 1:
            output = dia
            break
        if i % 2 == 0:  # speaker
            messages.append({"role": "user", "content": dia})
        else:
            messages.append({"role": "assistant", "content": dia})

    if not test:
        messages.append({"role": "assistant", "content": output})

    # prompt = prompt.format_map(data)
    return messages, output


def train(
        # model/data params
        base_model: str = "decapoda-research/llama-7b-hf",  # the only required argument
        train_data_path: str = "data/train.json",
        dev_data_path: str = "data/valid.json",
        test_data_path: str = "data/test.json",
        add_set: bool = False,
        # data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 32,
        micro_batch_size: int = 8,
        num_epochs: int = 25,
        # learning_rate: float = 1e-5,
        learning_rate: float = 3e-5,
        cutoff_len: int = 2000,
        val_set_size: int = 190,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        # lora_target_modules: List[str] = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_target_modules: List[str] = ['q_proj', 'k_proj'],
        # llm hyperparams
        train_on_inputs: bool = False,  # if False, masks out inputs in loss
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"train_data_path: {train_data_path}\n"
            f"dev_data_path: {dev_data_path}\n"
            f"test_data_path: {test_data_path}\n"
            f"add_set: {add_set}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )


    # pdb.set_trace()
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        # pdb.set_trace()
        result = tokenizer(prompt, add_special_tokens=False)
        # truncation=True,
        # max_length=cutoff_len,
        # padding=False,
        # return_tensors=None,
        # add_special_tokens=False
        # )
        # if(
        #     result["input_ids"][-1] != tokenizer.eos_token_id
        #     and len(result["input_ids"]) < cutoff_len
        #     and add_eos_token
        # ):
        #     result["input_ids"].append(tokenizer.eos_token_id)
        #     result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def save_prompt(prompt):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        open(os.path.join(output_dir, 'train_prompt_example.txt'), 'w').write(prompt)

    def generate_and_tokenize_prompt(data_point):
        chat_template = open('./chat_templates/chat_templates/llama-3-instruct.jinja').read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        tokenizer.chat_template = chat_template

        messages, output = generate_prompt(data_point)
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized_full_prompt = tokenizer(full_prompt, return_tensors="pt")
        for key in tokenized_full_prompt.keys():
            tokenized_full_prompt.update({key: list(tokenized_full_prompt[key][0])})
        input_prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=False)

        tokenized_input_prompt = tokenizer(input_prompt, return_tensors="pt")

        for key in tokenized_input_prompt.keys():
            tokenized_input_prompt.update({key: list(tokenized_input_prompt[key][0])})
        tokenized_full_prompt["labels"] = [-100] * len(tokenized_input_prompt['input_ids']) + tokenized_full_prompt["input_ids"][len(tokenized_input_prompt['input_ids']):]  # could be sped up, probably

        global print_prompt_flag
        if print_prompt_flag == 0:
            print(messages)
            print(full_prompt)
            save_prompt(full_prompt)
            print_prompt_flag += 1
        assert len(tokenized_full_prompt['input_ids']) == len(tokenized_full_prompt['labels'])
        return tokenized_full_prompt

    # model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config, )
    data_files = {}
    data_files["train"] = train_data_path
    extension = train_data_path.split(".")[-1]
    data_files["validation"] = dev_data_path
    # data_files["test"] = test_data_path

    raw_datasets = load_dataset(extension, data_files=data_files)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    train_data = train_dataset.shuffle().map(generate_and_tokenize_prompt)
    val_data = eval_dataset.shuffle().map(generate_and_tokenize_prompt)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    print(f"micro_batch_size:{micro_batch_size}")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            tf32=True,
            bf16=True,
            logging_steps=100,
            optim="adamw_torch",
            evaluation_strategy="epoch" if val_set_size > 0 else "no",
            save_strategy="epoch",
            eval_steps=500 if val_set_size > 0 else None,
            save_steps=500,
            output_dir=output_dir,
            save_total_limit=100,
            save_safetensors=False,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
            run_name=None,

        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # callbacks=[MyCallback],
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))
    #
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
