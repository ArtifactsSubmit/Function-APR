import re
import random
import tiktoken


binary_search_buggy = '''int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r + l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}'''
binary_search_fixed = '''int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r - l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}'''
binary_search_buggy_fl = '''int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r + l) / 2; /* bug is here */
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}'''

apr_label = '// Provide a fix for the buggy function.'
buggy_label = '// Buggy Function'
fixed_label = '// Fixed Function'


def get_example_bugs(bugs, curr_bug):
    curr_proj_example_lst = []
    other_proj_example_lst = []
    project = curr_bug.split("-")[0]
    for file_name, bug in bugs.items():
        if file_name == curr_bug:
            continue
        if file_name.startswith(project + "-"):
            curr_proj_example_lst.append((len(bug['buggy']) + len(bug['fix']), file_name))
        else:
            other_proj_example_lst.append((len(bug['buggy']) + len(bug['fix']), file_name))
    curr_proj_example_lst.sort(key=lambda x: x[0])
    other_proj_example_lst.sort(key=lambda x: x[0])

    curr_proj_file_names = [item[1] for item in curr_proj_example_lst]
    other_proj_file_names = [item[1] for item in other_proj_example_lst]
    example_bug_lst = curr_proj_file_names + other_proj_file_names

    return example_bug_lst

def build_apr_prompt_auto(apr_info, curr_bug, dataset):
    binary_search_codes = {
        'buggy': binary_search_buggy,
        'fixed': binary_search_fixed,
        'buggy_fl': binary_search_buggy_fl
    }

    k_value = apr_info.k_shot
    fl_flag = apr_info.enable_fl
    ce_flag = apr_info.enable_ce
    
    example_bug_lst = get_example_bugs(dataset, curr_bug)
    
    prompt_parts = []
    if ce_flag:
        prompt_parts.append(apr_label)
        prompt_parts.append(buggy_label)
        prompt_parts.append(binary_search_codes['buggy_fl'] if fl_flag else binary_search_codes['buggy'])
        prompt_parts.append(fixed_label)
        prompt_parts.append(binary_search_codes['fixed'])
        k_value -= 1
        
    for example_bug_name in example_bug_lst[:k_value]:
        example_bug_info = dataset[example_bug_name]
        prompt_parts.extend(add_info_to_prompt(example_bug_info, apr_info))

    current_bug_info = dataset[curr_bug]
    prompt_parts.extend(add_info_to_prompt(current_bug_info, apr_info, include_fix=False))
    prompt = '\n'.join(prompt_parts)
    return prompt

def add_info_to_prompt(bug_info, apr_info, include_fix=True):
    fl_flag = apr_info.enable_fl
    br_it_flag = apr_info.repair_info_type == 'it' or apr_info.repair_info_type == 'br'
    br_id_flag = apr_info.repair_info_type == 'id' or apr_info.repair_info_type == 'br'
    pi_tt_flag = apr_info.repair_info_type == 'tt' or apr_info.repair_info_type == 'pi'
    pi_em_flag = apr_info.repair_info_type == 'em' or apr_info.repair_info_type == 'pi'
    pi_bc_flag = apr_info.repair_info_type == 'bc' or apr_info.repair_info_type == 'pi'


    parts = []
    parts.append(apr_label)
    if br_it_flag:
        parts.append(bug_info['issue_title'])
    if br_id_flag:
        parts.append(bug_info['issue_description'])
    if pi_bc_flag:
        parts.append(bug_info['buggy_code_comment'])
    if pi_tt_flag or pi_em_flag:
        random_trigger_test = random.choice(list(bug_info['trigger_test'].keys()))
        if pi_tt_flag:
            trigger_src = bug_info['trigger_test'][random_trigger_test]['src']
            parts.append(trigger_src)
        if pi_em_flag:
            err_msg = bug_info['trigger_test'][random_trigger_test]['error_msg']
            err_msg = slim_content_token(err_msg)
            parts.append(err_msg)
    
    parts.append(buggy_label)
    parts.append(bug_info['buggy_fl'] if fl_flag else bug_info['buggy'])
    parts.append(fixed_label)
    if include_fix:
        parts.append(bug_info['fix'])
    return parts

def slim_content_token(err_msg):
    token_upper_limit = 200
    err_msg_lst = err_msg.split('\n')
    slim_err_msg_lst = []
    for line in err_msg_lst:
        curr_line_token_cnt = num_tokens_from_string(line)
        token_upper_limit -= curr_line_token_cnt
        if token_upper_limit <= 0:
            break
        slim_err_msg_lst.append(line)
    return '\n'.join(slim_err_msg_lst)


def num_tokens_from_string(string: str) -> int:
    encoding_name = "cl100k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens