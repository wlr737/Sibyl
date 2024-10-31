
import sys
local_transformers_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/internship/wlr_test/pkgs/transformers/src"
sys.path.insert(0, local_transformers_path)
import transformers
print(transformers.__file__)
import os
from typing import List

import fire
import torch
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
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
print_prompt_flag = 0
torch.backends.cuda.matmul.allow_tf32 = True
def generate_prompt( data_point, cci, test=False):
    prompt = ""
    data_input = ""
    data = {}

    for j in range(len(data_point['dialog']) - 1, -1, -1):
        if data_point['dialog'][j]['speaker'] == 'usr':
            continue
        break
    data_point['dialog'] = data_point['dialog'][:j + 1]
    if cci == 'ChatGPT_intent':
        prompt += \
'''Given a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress, the objective is to comprehend the dialogue in order to identify the emotional reaction of the help seeker in their last utterance. Subsequently, you need to infer and analyze the supporter's intent, taking into consideration the emotion reaction of the help seeker.

I will provide an example, which is as follows:

(1)Help seeker: I am laid off .
(2)Supporter: Oh I am really sorry to hear that , Did you have the same job for a long time ? That sounds very difficult to deal with .
(3)Help seeker: I was attending a customer . He was having a grievance which I sorted out . But the management did not like that . For the past 10 years I have been with this job .
(4)Supporter: 10 years is a very long time and I would think you have probably learned a lot working at the same place for that long . You ' re a dedicated employee .
(5)Help seeker: I was okay with the previous manager . But recently a new young chap joined the duty . He was not experienced . He only is the reason for so .
(6)Supporter: That is really unfair and hard to deal with are you close to any family ?
(7)Help seeker: I am not in any close to any family related to job .
(8)Supporter: Do you have any close friends to talk to about any new job prospects ?
(9)Help seeker: I have few friends , I have been talking to them . They also tell I was not any wrong . For the experience and qualification , I will be getting a better job .

Emotion reaction of the help seeker: Optimism: The help seeker expresses optimism about their job prospects, feeling hopeful and confident in their skills and qualifications to secure a better job.

What is the supporter's intent to post the last utterance according to the emotion reaction of the help seeker? Please infer and analyze the supporter's intent conditioned on helper seeker's emotion. 

Answer: The supporter's intent is to reinforce the help seeker's optimism and positive outlook. By suggesting joining a new group at a church or something similar, the supporter aims to provide additional emotional support and offer opportunities for the help seeker to connect with others who may provide guidance and new job prospects. 

Now, generate one concise and relevant inference (no more than 40 words) of the following conversation clip. The conversation clip is: 

{context}

Emotion reaction of the help seeker: {emo}

What is the supporter's intent to post the last utterance according to the emotion reaction of the help seeker? Please infer and analyze the supporter's intent conditioned on helper seeker's emotion reaction.

'''
    elif cci == 'ChatGPT_cause':
        prompt += \
'''Given a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress, the objective is to comprehend the dialogue and make inferences to identify the underlying cause of the latest utterance stated by the supporter (the reason contributing to the utterance stated by the supporter).

I will provide an example of a conversation clip and the explanation of causes, which is as follows:

(1)Help seeker: I am laid off .
(2)Supporter: Oh I am really sorry to hear that , Did you have the same job for a long time ? That sounds very difficult to deal with .
(3)Help seeker: I was attending a customer . He was having a grievance which I sorted out . But the management did not like that . For the past 10 years I have been with this job .
(4)Supporter: 10 years is a very long time and I would think you have probably learned a lot working at the same place for that long . You ' re a dedicated employee .
(5)Help seeker: I was okay with the previous manager . But recently a new young chap joined the duty . He was not experienced . He only is the reason for so .
(6)Supporter: That is really unfair and hard to deal with are you close to any family ?
(7)Help seeker: I am not in any close to any family related to job .
(8)Supporter: Do you have any close friends to talk to about any new job prospects ?
(9)Help seeker: I have few friends , I have been talking to them . They also tell I was not any wrong . For the experience and qualification , I will be getting a better job .

What is the cause of supporter to post the last utterance? Please make inference based on the utterances before the last utterance of the conversation. Please generate the answer like this: Answer: The supporter recognizes that the help seeker is facing unfair treatment at work due to the new inexperienced manager, which is causing emotional distress. The suggestion of joining a new group at a church might be a way to provide a supportive environment outside of the workplace.

Now, generate one concise and relevant inference (no more than 40 words) of the cause of the last utterance. The conversation clip is: 

{context}
'''
    elif cci == 'ChatGPT_emo':
        prompt += \
'''Given a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress, the objective is to comprehend the dialogue in order to identify the emotional reaction of the help seeker in their last utterance.

I will provide an example, which is as follows:

(1)Help seeker: I am laid off .
(2)Supporter: Oh I am really sorry to hear that , Did you have the same job for a long time ? That sounds very difficult to deal with .
(3)Help seeker: I was attending a customer . He was having a grievance which I sorted out . But the management did not like that . For the past 10 years I have been with this job .
(4)Supporter: 10 years is a very long time and I would think you have probably learned a lot working at the same place for that long . You ' re a dedicated employee .
(5)Help seeker: I was okay with the previous manager . But recently a new young chap joined the duty . He was not experienced . He only is the reason for so .
(6)Supporter: That is really unfair and hard to deal with are you close to any family ?
(7)Help seeker: I am not in any close to any family related to job .
(8)Supporter: Do you have any close friends to talk to about any new job prospects ?
(9)Help seeker: I have few friends , I have been talking to them . They also tell I was not any wrong . For the experience and qualification , I will be getting a better job .

What is the emotional reaction of the help seeker in their last utterance? Please generate the answer like this: Answer: Optimism: The help seeker expresses optimism about their job prospects, feeling hopeful and confident in their skills and qualifications to secure a better job.

Now, generate one concise and relevant inference (no more than 40 words) about the emotional reaction of the help seeker of the last utterance. The conversation clip is:  

{context}
'''
    elif cci == 'ChatGPT_subs':
        prompt += \
'''Given a dyadic dialogue clip between an emotional supporter and a help seeker who seeks for help in relieving emotional distress, the objective is to comprehend the dialogue context and make inferences about potential subsequent events involving the supporter that may occur after the help seeker's last utterance.

I will provide an example, which is as follows:

(1)Help seeker: I am laid off .
(2)Supporter:: Oh I am really sorry to hear that , Did you have the same job for a long time ? That sounds very difficult to deal with .
(3)Help seeker: I was attending a customer . He was having a grievance which I sorted out . But the management did not like that . For the past 10 years I have been with this job .
(4)Supporter: 10 years is a very long time and I would think you have probably learned a lot working at the same place for that long . You ' re a dedicated employee .
(5)Help seeker: I was okay with the previous manager . But recently a new young chap joined the duty . He was not experienced . He only is the reason for so .
(6)Supporter: That is really unfair and hard to deal with are you close to any family ?
(7)Help seeker: I am not in any close to any family related to job .
(8)Supporter: Do you have any close friends to talk to about any new job prospects ?
(9)Help seeker:I have few friends , I have been talking to them . They also tell I was not any wrong . For the experience and qualification , I will be getting a better job .
(10)Supporter: Do you have any close friends to talk to about any new job prospects ?
(11)Help seeker: I have few friends , I have been talking to them . They also tell I was not any wrong . For the experience and qualification , I will be getting a better job .
(12)Supporter: That is a positive outlook and is good to hear that you know you have skills to offer . Would you consider joining a new group at a church or something like that ?
(13)Help seeker: I am so worried of the management taken action on me relying a new inexperienced manager . .
(14)Supporter: I am sorry that you ' re feeling stress . Have you ever used writing as a tool to relax ?
(15)Help seeker: My colleagues are also in contact with me . They are also having similar inconvenience as to how to perform ? I have some other relaxation like listening to music , gardening etc . ,

What is the subsequent event potential subsequent events involving the supporter that may occur after the help seeker's last utterance? Please generate the answer like this: Answer: The supporter could recommend specific relaxation techniques involving music or gardening to further enhance the help seeker's coping mechanisms.

Now, generate one concise and relevant inference (no more than 40 words) about subsequent events involving the supporter that may occur after the help seeker's last utterance. The conversation clip is:

{context}
'''
    for j in range(len(data_point['dialog']) - 1, -1, -1):
        if data_point['dialog'][j]['speaker'] == 'usr':
            continue
        break

    data_point['dialog'] = data_point['dialog'][:j + 1]
    output = data_point['dialog'][-2][cci]

    if "Answer:" == output[:len("Answer:")]:
        output = output[len("Answer:"):].strip()
    elif "Inference:" == output[:len("Inference:")]:
        output = output[len("Inference:"):].strip()

    emo = data_point['dialog'][-2]["ChatGPT_emo"]
    for i, dia in enumerate(data_point['dialog']):
        if i == len(data_point['dialog']) - 1:
            break
        if dia['speaker'] == 'usr':
            data_input += '({})Help seeker: '.format(i + 1)
        else:
            data_input += '({})Supporter: '.format(i + 1)  # + 'Strategy chosen: ' + dia['annotation']['strategy']
        data_input += dia['text'].strip('\n') + '\n'


    # if test:
    #     prompt += "Answer: "
    #     if cci == 'ChatGPT_intent':
    #         data.update({'context': data_input, "emo":emo})
    #     else:
    #         data.update({'context': data_input})
    # else:
    #     prompt += "Answer: {output}"
    #     if cci == 'ChatGPT_intent':
    #         data.update({'context': data_input, 'output': output,"emo":emo})
    #     else:
    #         data.update({'context': data_input,'output': output})
    if cci == 'ChatGPT_intent':
        data.update({'context': data_input, "emo": emo})
    else:
        data.update({'context': data_input})

    prompt += "Answer: "
    prompt = prompt.format_map(data)
    messages = []
    messages.append({"role": "system", "content": "You are an expert in the theory of emotional support and conversational contextual reasoning. "})
    messages.append({"role": "user", "content": prompt})
    if not test:
        messages.append({"role": "assistant", "content": output})
    return messages, output

def train(
        # model/data params
        base_model: str = "decapoda-research/llama-7b-hf",  # the only required argument
        train_data_path: str = "data/train.json",
        dev_data_path: str = "data/valid.json",
        test_data_path: str = "data/test.json",
        # data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 32,
        micro_batch_size: int = 8,
        num_epochs: int = 5,
        cci_type: str = "ChatGPT_cause",
        learning_rate: float = 3e-5,
        cutoff_len: int = 2000,
        val_set_size: int = 190,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        # lora_target_modules: List[str] = ['q_proj', 'v_proj'],
        lora_target_modules: List[str] = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        # llm hyperparams
        train_on_inputs: bool = False,  # if False, masks out inputs in loss
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"cci_type: {cci_type}\n"
            f"base_model: {base_model}\n"
            f"train_data_path: {train_data_path}\n"
            f"dev_data_path: {dev_data_path}\n"
            f"test_data_path: {test_data_path}\n"
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
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
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
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    # model = prepare_model_for_int8_training(model)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 128002


    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=False):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=False
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result
    def save_prompt(prompt):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        open(os.path.join(output_dir, 'train_prompt_example.txt'),'w').write(prompt)

    def generate_and_tokenize_prompt(data_point):
        chat_template = open('./chat_templates/chat_templates/llama-3-instruct.jinja').read()
        chat_template = chat_template.replace('    ', '').replace('\n', '')
        tokenizer.chat_template = chat_template

        messages, output = generate_prompt(data_point,cci_type)
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

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
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

    # if resume_from_checkpoint:
    #     # Check the available weights and load them
    #     checkpoint_name = os.path.join(
    #         resume_from_checkpoint, "pytorch_model.bin"
    #     )  # Full checkpoint
    #     if not os.path.exists(checkpoint_name):
    #         checkpoint_name = os.path.join(
    #             resume_from_checkpoint, "adapter_model.bin"
    #         )  # only LoRA model - LoRA config above has to fit
    #         resume_from_checkpoint = (
    #             False  # So the trainer won't try loading its state
    #         )
    #     # The two files above have a different name depending on how they were saved, but are actually the same.
    #     if os.path.exists(checkpoint_name):
    #         print(f"Restarting from {checkpoint_name}")
    #         adapters_weights = torch.load(checkpoint_name)
    #         set_peft_model_state_dict(model, adapters_weights)
    #     else:
    #         print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            tf32=True,
            bf16=True,
            logging_steps=250,
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
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
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
