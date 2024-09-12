import os
import re
import sys
import time
import json
import openai 
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from apr_prompt import build_apr_prompt_auto

model_base_path = './Model'
model_path_dict = {
    'magicoder'     :   f'{model_base_path}/ise-uiuc/Magicoder-S-CL-7B',
    'cl7b'          :   f'{model_base_path}/codellama/CodeLlama-7b-Instruct-hf',
    'cl13b'         :   f'{model_base_path}/codellama/CodeLlama-13b-Instruct-hf',
    'cl34b'         :   f'{model_base_path}/codellama/CodeLlama-34b-Instruct-hf',
}

model_format_prompt = {
    'magicoder'     : '''Return your fixed function surrounded with ```java\\n```.\n@@ Instruction\n{apr_prompt}\n@@ Response''',
    'cl7b'          : '''<s> <<SYS>> Return your fixed function surrounded with ```java\\n```. <</SYS>> [INST] {apr_prompt} [/INST]''',
    'cl13b'         : '''<s> <<SYS>> Return your fixed function surrounded with ```java\\n```. <</SYS>> [INST] {apr_prompt} [/INST]''',
    'cl34b'         : '''<s> <<SYS>> Return your fixed function surrounded with ```java\\n```. <</SYS>> [INST] {apr_prompt} [/INST]''',
}

def extract_patched_method(testcase_lst):
    method_info_lst = []
    
    bracket_cnt = 0
    bracket_flag = False
    for line in testcase_lst:
        method_info_lst.append(line)
        bracket_cnt += line.count('{')
        if bracket_cnt:
            bracket_flag = True
        bracket_cnt -= line.count('}')
        if bracket_cnt == 0 and bracket_flag:
            return method_info_lst
    return None

def extract_all_patch_codes(orig_patch, dataset, bug_name):
    patch_code_lst = []
    code_patch_pattern = r'```(?:java\n)?(.*?)\n```'
    extracted_lst = re.findall(code_patch_pattern, orig_patch, re.DOTALL)
    method_name = ' ' + dataset[bug_name]['method_signature']['method_name'] + '('
    method_return_type = dataset[bug_name]['method_signature']['return_type']

    if extracted_lst:
        for patch_code in extracted_lst:
            if method_name in patch_code and method_return_type in patch_code:
                patch_code_lst.append(patch_code)
        if len(patch_code_lst) > 0:
            return patch_code_lst
    
    orig_patch_lines = orig_patch.split('\n')
    len_orig_patch_lines = len(orig_patch_lines)
    for idx in range(len_orig_patch_lines - 1, -1, -1):
        curr_rline = orig_patch_lines[idx]
        if method_name not in curr_rline or method_return_type not in curr_rline:
            continue
        patch_code = extract_patched_method(orig_patch_lines[idx:])
        if patch_code:
            patch_code_lst.append('\n'.join(patch_code))
            break
    return patch_code_lst

def extract_patch(apr_info, raw_patch_result):
    if apr_info.model_name not in model_format_prompt:
        return raw_patch_result
    for bug_name in raw_patch_result:
        extracted_patches = []
        for raw_patch in raw_patch_result[bug_name]['patches']:
            extracted_patches.extend(extract_all_patch_codes(raw_patch, apr_info.dataset, bug_name))
        raw_patch_result[bug_name]['patches'] = extracted_patches
    return raw_patch_result

def api_query_model(prompt, buggy_code, sample_size, model_name):
    delay = 10
    while(True):
        try:
            if model_name == 'codex-edit':
                response = api_codex_response(prompt, buggy_code, sample_size)
            elif model_name == 'chatgpt':
                response = api_gpt3_5_response(prompt, sample_size)
            else:
                raise ValueError("Model type not supported")
            break
        except openai.APIError as e:
            if "Please reduce " in str(e):
                raise ValueError("Over Length")
            print(f"OpenAI API returned an API Error: {e}")
            time.sleep(delay)
        except Exception as e:
            print(f"Exception in api_query_model {e}")
            if "Please reduce " in str(e):
                raise ValueError("Over Length")
            time.sleep(delay)

    curr_patches_lst = []
    if model_name == 'codex-edit':
        for choice in response.choices:
            if choice.text:
                curr_patches_lst.append(choice.text)
        return curr_patches_lst
    elif model_name == 'chatgpt':
        for choice in response.choices:
            if choice.message:
                curr_patches_lst.append(choice.message.content)
        return curr_patches_lst


def api_codex_response(prompt, buggy_code, n):
    response = openai.edits.create(
        input=buggy_code,
        instruction=prompt,
        temperature=0.8,
        model="code-davinci-edit-001",
        n=n)
    return response


def api_gpt3_5_response(prompt, n):
    response = openai.chat.completions.create(    
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo-1106",
        n=n,
        temperature=0.8)
    return response

class AprInfo():
    def __init__(self, dataset_path, out_path, model_name, sample_size, target_bug, k_shot, enable_ce, enable_fl, repair_info_type):
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        if target_bug is not None:
            self.dataset = {target_bug: self.dataset[target_bug]}
        self.out_path = out_path
        self.model_name = model_name
        self.sample_size = sample_size
        self.target_bug = target_bug
        self.k_shot = k_shot
        self.enable_ce = enable_ce
        self.enable_fl = enable_fl
        self.repair_info_type = repair_info_type
        return
    

def api_model_apr(apr_info):
    os.environ["OPENAI_API_KEY"] = 'OPENAI_API'
    patches = {}
    for bug_name, bug_info in apr_info.dataset.items():
        curr_patch = {}
        curr_patch['patches'] = []
        prompt = build_apr_prompt_auto(apr_info, bug_name, apr_info.dataset)
        curr_patch['prompt'] = prompt
        buggy_code = bug_info['buggy']
        curr_patch['patches'] += api_query_model(prompt, buggy_code, apr_info.sample_size, apr_info.model_name)
        patches[bug_name] = curr_patch
    return patches


def opensrc_model_apr(apr_info):
    dataset = apr_info.dataset
    model_name = apr_info.model_name
    
    tokenizer = AutoTokenizer.from_pretrained(model_path_dict[model_name], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path_dict[model_name], local_files_only=True, torch_dtype=torch.float16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    patches = {}
    for bug_name in dataset:
        apr_prompt = build_apr_prompt_auto(apr_info, bug_name, dataset)
        prompt = model_format_prompt[model_name].format(apr_prompt=apr_prompt.strip())
        prompt_token_len = len(tokenizer.encode(prompt))
        if prompt_token_len > 4000:
            continue

        buggy_token_len = len(tokenizer.encode(dataset[bug_name]['buggy']))
        curr_max_new_token = max(256, 2*buggy_token_len)

        curr_patch = {}
        curr_patch['prompt'] = apr_prompt
        curr_patch['patches'] = []

        batch_size = min(5, apr_info.sample_size)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda") 
        
        while len(curr_patch['patches']) < apr_info.sample_size:
            try:
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=curr_max_new_token,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=pad_token_id,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.8,
                    num_return_sequences=batch_size,
                )
                for _, output in enumerate(outputs):
                    generated_text = tokenizer.decode(output, skip_special_tokens=True)
                    prompt_end_idx = generated_text.find(apr_prompt.strip()) + len(apr_prompt.strip())
                    fixed_result = generated_text[prompt_end_idx:].strip() if prompt_end_idx != -1 else generated_text
                    fixed_result = fixed_result.replace('[/INST]  ','')
                    curr_patch['patches'].append(fixed_result)

                curr_patch_cnt = len(curr_patch['patches'])
                print(f'### [APR]:bug_name:{bug_name:25}  |  curr_patch_cnt:{curr_patch_cnt:>3}  |  batch_size:{batch_size:2}  |  patches_cnt:{len(patches):3} ###')

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    batch_size = batch_size//2
                    if batch_size == 0:
                        break
                else:
                    raise e
        patches[bug_name] = curr_patch

    return patches


def main():
    apr_result = {}
    apr_info = AprInfo(args.d, args.o, args.m, args.s, args.bug, \
                       args.k, args.ce, args.fl, args.info)
    apr_result = opensrc_model_apr(apr_info) if apr_info.model_name in \
        model_format_prompt else api_model_apr(apr_info)
    apr_result = extract_patch(apr_info, apr_result)
    with open(apr_info.out_path, 'w') as f:
        json.dump(apr_result, f, indent=2)
    return
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True, help='dataset path')
    parser.add_argument('-o', type=str, required=True, help='patch_result path')
    parser.add_argument('-m', type=str, required=True, help='model')
    parser.add_argument('-s', type=int, required=False, help='sample_size', default=1)
    parser.add_argument('-bug', type=str, required=False, help='bug')
    parser.add_argument('-k', type=str, required=False, help='k-shot learning', default=0)
    parser.add_argument('-ce', type=bool, required=False, help='use crafted example', default=False)
    parser.add_argument('-fl', type=bool, required=False, help='use fault localization information', default=False)
    parser.add_argument('-info', type=str, required=False, help='repair info: 1.it:issue title\n2.id:issue description\n3.br:bug report\n\
                        4.bc:buggy comment\n5.tt:trigger test\n6.em:error message\n7.pi:project info', default='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main()

    
