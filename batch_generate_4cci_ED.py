import json
import os,sys
import pdb
local_transformers_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/internship/wlr_test/pkgs/transformers/src"
sys.path.insert(0, local_transformers_path)
import transformers
print(transformers.__file__)
import fire
import torch

from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    PeftModel
)

from typing import List
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from finetune_llama3_generate_4cci import generate_prompt
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print_prompt_flag = 0

def main(
        load_8bit: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        batch_size: int = 4,
        cci_type: str = "ChatGPT_cause",
        lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
        # lora_target_modules: List[str] = ["q_proj", "v_proj"],
        lora_dropout: float = 0.05,
        add_set: bool = False,
        save_file_name='eval_list.json',
        test_file='',
        base_model: str = "/apdcephfs/share_47076/moyumyu/model_dir/llama-7b",
        lora_weights: str = "/apdcephfs/share_47076/jiangnanli/personet_model_save/save_ddp_4gpus_no_history/checkpoint-1360/pytorch_model.bin",
        temperature: float = 1.0,
        do_sample: bool = False,  # set True to activate sampling which will use the temperature/topk/topp.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"cci_type: {cci_type}\n"
            f"save_file_name: {save_file_name}\n"
            f"temperature: {temperature}\n"
            f"do_sample: {do_sample}\n"
            f"add_set: {add_set}\n"
            f"load_8bit: {load_8bit}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
        )
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model_state_dict = torch.load(lora_weights, map_location='cuda:0')
        set_peft_model_state_dict(model, model_state_dict)
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    tokenizer.pad_token_id = 128002
    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    def save_prompt(prompt):
        open(os.path.join('/'.join(lora_weights.split('/')[:-1]), 'prompt_example.txt'), 'w').write(prompt)

    def evaluate(
            prompts,
            top_p=1,
            top_k=50,
            num_beams=1,
            max_new_tokens=100,
            **kwargs,
    ):

        # answers = completions
        new_prompts = []
        new_answers = []
        for prompt in prompts:
            messages, answer = generate_prompt(prompt,cci_type,test=True)
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            global print_prompt_flag
            if print_prompt_flag == 0:
                print(full_prompt)
                save_prompt(full_prompt)
                print_prompt_flag += 1
            new_prompts.append(full_prompt)
            new_answers.append(answer)
        # pdb.set_trace()
        inputs = tokenizer(new_prompts, add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        if input_ids.shape[0] == 1:
            attention_mask = None
        else:
            attention_mask = inputs["attention_mask"].cuda()
        #
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            # eos_token_id=2,
            # bos_token_id=1,
            pad_token_id=128002,
            # no_repeat_ngram_size=6,
            # repetition_penalty=1.8,
            do_sample=do_sample,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_mask,
            )
        sequences = generation_output.sequences
        outputs = []
        for s in sequences:
            output = tokenizer.decode(s)
            outputs.append(output)

        preds = outputs
        preds_golds = [[p, g] for p, g in zip(preds, new_answers)]

        return preds_golds

    eval_data = json.load(open(test_file, 'r', encoding='utf-8'))
    save_path = '/'.join(lora_weights.split('/')[:-1]) + "/" + save_file_name
    print(save_path)
    save_list = []
    # eval_data = eval_data[:20]
    data_iter_num = len(eval_data) // batch_size
    left_num = len(eval_data) % batch_size
    for iter_id in tqdm(range(data_iter_num)):
        data_points = eval_data[iter_id * batch_size: (iter_id + 1) * batch_size]
        pred_answer = evaluate(data_points)
        save_list += pred_answer
        # if iter_id % 5:
        #     sp = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #     out_str = sp.communicate()
        #     for out_element in out_str:
        #         for line in str(out_element).split('\\n'):
        #             print(line, file=sys.stderr)
        json.dump(save_list, open(save_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    if left_num != 0:
        data_points = eval_data[-left_num:]
        pred_answer = evaluate(data_points)
        save_list += pred_answer
        json.dump(save_list, open(save_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(save_list, open(save_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
